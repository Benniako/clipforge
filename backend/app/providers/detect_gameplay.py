"""Gameplay highlight detection from audio energy.

A game's best moments — a Valorant ace, an EA FC goal, a clutch — don't show up
in the transcript, but they almost always spike the audio: gunfire, ability
sounds, crowd roar, commentary hype, your own reaction. So we find loud,
transient moments and build clips around them, ranked by intensity.

This is game-agnostic and needs no extra models (just ffmpeg + numpy). It's the
foundation; per-game vision detectors (kill-feed / scoreboard OCR) can layer on
top behind this same interface.

Where a game's scoreboard can be tracked (``providers.hud``) its goals are the
primary highlights: a digit that went up by one is proof, a loud moment or the
word "GOAL" somewhere on screen is not. Stadium LED boards literally read GOAL
and CHAMPIONS, which the keyword OCR took for a goal and for a menu screen. So
with a healthy scoreboard result the keyword sweep is skipped, the remaining
detectors are held to the time the scoreboard was on screen, and nothing they
find may outrank a goal read off the scoreboard. Without one, detection works
exactly as it did before the tracker existed.
"""
from __future__ import annotations

import logging
import os
import wave
from dataclasses import dataclass, field

from ..config import get_settings
from ..media import ffmpeg
from ..media.ffmpeg import MediaInfo
from ..models import DetectedEvent, ImportSettings, ScoreFactor
from . import audio_events as audio_events_mod
from . import detect_cues, detect_ocr
from . import hud as hud_mod

log = logging.getLogger("clipforge.gameplay")

# Share of the video the scoreboard must cover before "the scoreboard is not on
# screen" is trusted as "this is not match time" and other detectors are gated
# by it. Below this the tracker still supplies its goals, but a scoreboard seen
# for a fraction of the footage says little about the rest of it.
HUD_GATE_MIN_COVERAGE = 0.30
# Ceiling for cue / keyword-OCR / audio-event clips while a scoreboard is
# tracked, so unverified moments rank below goals read off the scoreboard.
HUD_UNVERIFIED_CAP = 79
# ... and the floor for those goals, however little one mattered to its match.
HUD_GOAL_FLOOR = 80
# Most a goal can gain from how loud its moment was (a tie-breaker, not proof).
HUD_LOUDNESS_BONUS = 4
# CLIPFORGE_HUD values that switch scoreboard tracking off.
_HUD_OFF_VALUES = {"0", "false", "off", "no"}
# Keyword-OCR labels the scoreboard tracker supersedes.
_HUD_SUPERSEDED_LABELS = {"goal", "scoreboard"}

# Map a game profile to its cue-template folder (aliases included).
_CUE_DIR = {"auto": "generic", "cs": "cs2", "fifa": "eafc", "valorant": "valorant"}

# Per-game tuning of the audio signals. "generic"/"auto" work for ANY game; the
# named profiles bias how peaks are picked and scored:
#   thr_pct        - loudness percentile to count as "action" (higher = pickier)
#   min_gap_factor - min spacing between highlights (× min_len)
#   lead_frac      - how much of the clip is build-up before the peak
#   w_transient    - reward sudden loud-after-quiet (jump scares, gunfire)
#   w_sustain      - reward sustained loudness (goals/crowd roar, team-fights)
_PROFILES: dict[str, dict] = {
    "generic":      dict(thr_pct=82, min_gap_factor=0.6, lead_frac=0.75, w_transient=3.0,  w_sustain=6.0),
    "valorant":     dict(thr_pct=80, min_gap_factor=0.45, lead_frac=0.76, w_transient=10.0, w_sustain=3.0),
    "cs2":          dict(thr_pct=80, min_gap_factor=0.45, lead_frac=0.76, w_transient=10.0, w_sustain=3.0),
    "eafc":         dict(thr_pct=85, min_gap_factor=0.85, lead_frac=0.78, w_transient=2.0,  w_sustain=13.0),
    "rocketleague": dict(thr_pct=83, min_gap_factor=0.60, lead_frac=0.77, w_transient=6.0,  w_sustain=8.0),
    "horror":       dict(thr_pct=88, min_gap_factor=0.90, lead_frac=0.80, w_transient=16.0, w_sustain=2.0),
}
_PROFILES["auto"] = _PROFILES["generic"]
_PROFILES["cs"] = _PROFILES["cs2"]
_PROFILES["fifa"] = _PROFILES["eafc"]

# The accepted values for ImportSettings.game_profile. The profile also names
# a cue directory on disk, so the API clamps input to this set.
KNOWN_PROFILES = frozenset(_PROFILES)

# Visual-on-screen markers for game-profile auto-detection.  When the project's
# game_profile is "auto", we sample a handful of frames, run OCR, and pick the
# profile whose marker set gets the most hits.  Markers are case-insensitive
# substrings — the OCR output is noisy, so we keep them loose.
_AUTO_MARKERS: dict[str, list[str]] = {
    "valorant": [
        "getötet", "kampfbericht", "kauphase", "spike", "runde läuft",
        "runde gewonnen", "agent", "taktiker", "unrated", "competitive",
        "schießstand", "gewerte",
    ],
    "eafc": [
        "ea sports", "ea sports fc", "fc 24", "fc 25", "fc 26", "fc 27",
        "fc24", "fc25", "fc26", "fc27", "eafc", "fifa", "ultimate team",
        "division rivals", "weekend league", "squad battles", "career mode",
        "pro clubs", "clubs", "rush", "kick off", "kick-off", "goal", "tor",
        "champions", "fussball", "abpfiff", "mannschaft", "schuss",
        "karriere", "trikot", "liga", "stadion", "saison", "penalty",
        "half time", "full time", "extra time", "stoppage time",
        "auswechslung", "vorlage", "pässe", "präzision", "dribbling",
        "bewertung", "gesamt", "temp", "sprint",
    ],
    "cs2": [
        "counter-terrorist", "terrorist", "bombe", "defuse",
    ],
    "rocketleague": [
        "rocket league", "punktzahl", "tor",
    ],
}


def auto_detect_profile(src_path: str, n_samples: int = 5) -> str:
    """Sample ~5 frames from the source video, run OCR, and pick a matching game
    profile by counting keyword hits.  Returns "generic" (default gameplay) when
    nothing clearly matches."""
    try:
        from .detect_ocr import _ocr_image_conf as _ocr
        from ..media.ffmpeg import probe, grab_frame
        import tempfile, os

        info = probe(src_path)
        dur = info.duration
        if dur <= 0:
            return "generic"

        # Sample across the whole VOD, with extra density in the first few
        # minutes where game menus (most text-rich screens) usually appear.
        # EA FC menus show "ULTIMATE TEAM", "KARRIERE", "CHAMPIONS" etc.
        # which OCR reads well. In-match scoreboard text is sparser.
        n_spread = max(4, n_samples - 4)
        ts = [int(dur * i / (n_spread + 1)) for i in range(1, n_spread + 1)]
        # Dense sampling in the first 90 seconds (menus, pre-match screens).
        for extra in [3, 8, 15, 25, 40, 55, 75]:
            if extra < dur:
                ts.append(extra)
        text_lower = ""
        with tempfile.TemporaryDirectory() as td:
            for t in ts:
                dst = os.path.join(td, f"f{t}.jpg")
                grab_frame(src_path, dst, t=t, width=1280)
                raw, _ = _ocr(dst, "easyocr", lang="de")
                text_lower += " " + raw.lower()

        hits: dict[str, int] = {}
        for profile, markers in _AUTO_MARKERS.items():
            c = sum(1 for m in markers if m.lower() in text_lower)
            if c:
                hits[profile] = c

        if not hits:
            return "generic"
        best = max(hits, key=hits.get)
        log.info("auto-detected game_profile=%s (hits=%s)", best, hits)
        return best
    except Exception as e:
        log.debug("auto-detect game profile failed: %s", e)
        return "generic"


def get_profile(name: str | None) -> dict:
    return _PROFILES.get((name or "generic").lower().replace(" ", ""), _PROFILES["generic"])


# Base feature weights per game (sum 1.0 over the audio features). These are the
# *defaults* the learning loop personalises (app/feedback.py): e.g. EA FC rewards
# sustained crowd roar, horror/Valorant reward sudden spikes.
GAMEPLAY_WEIGHTS: dict[str, dict[str, float]] = {
    "generic":      {"intensity": 0.50, "sustain": 0.25, "transient": 0.20, "spikes": 0.05},
    "valorant":     {"intensity": 0.45, "sustain": 0.15, "transient": 0.35, "spikes": 0.05},
    "cs2":          {"intensity": 0.45, "sustain": 0.15, "transient": 0.35, "spikes": 0.05},
    "eafc":         {"intensity": 0.40, "sustain": 0.45, "transient": 0.10, "spikes": 0.05},
    "rocketleague": {"intensity": 0.45, "sustain": 0.30, "transient": 0.20, "spikes": 0.05},
    "horror":       {"intensity": 0.35, "sustain": 0.10, "transient": 0.50, "spikes": 0.05},
}
_GW_ALIAS = {"auto": "generic", "cs": "cs2", "fifa": "eafc"}

_GW_FACTOR = {
    "intensity": ("High audio intensity", "Louder than {pct}% of the footage"),
    "transient": ("Sudden spike", "Loud burst after quiet (jump-scare / gunfire)"),
    "sustain": ("Sustained action", "High-energy stretch (goal / team-fight)"),
    "spikes": ("Multiple action spikes", "Several loud events (multi-kill / build-up)"),
    "reaction": ("Streamer reacts hard", "Big facecam reaction during the moment"),
}

# Share of the score the facecam reaction gets when a cam is present (the
# audio weights are scaled down to make room — see orchestrator). Clips with
# a visible reaction earn ~2.3x the engagement of gameplay-only clips
# (creator-platform data, 2026) — the reaction IS the content.
REACTION_WEIGHT = 0.2
CONFIRM_WINDOW = 3.0
MENU_CONTEXT_WINDOW = 5.0
MENU_OCR_LABELS = {"menu", "substitution"}
MENU_OCR_WORDS = {
    "settings", "inventory", "main menu", "menu", "lobby", "store", "shop",
    "collection", "loadout", "agent select", "matchmaking", "einstellungen",
    "inventar", "hauptmenu", "hauptmenue", "hauptmenü", "menue", "menü",
    "laden", "sammlung", "ausruestung", "ausrüstung", "agenten",
    # EA FC / FIFA menu, squad, tactics & hub keywords
    "team-management", "team taktiken", "profi-rollen", "taktik bearbeiten",
    "ansicht wechseln", "spielertausch", "tutorials", "aufteilung", "fc hub",
    "transfermarkt", "zentrale", "modi", "online draft", "einzelspieler",
    "moments", "aufgaben-fortschritt", "dein aufgaben-fortschritt", "ultimate team",
    "chemie", "ges-chemie", "profi-info", "squad battles", "career mode",
    "karriere", "karrieremodus", "rivals", "champions", "weekend league",
    "anstoß", "kick-off", "playstyles", "chemie-stil", "trainer", "verträge",
    "active squad", "player bio", "taktik", "zuweisungen", "auswechslungen",
    "galerie", "galerie-level", "top-scorer", "ausw", "auswechselbank", "substitutes",
    "spiel fortsetzen", "spielinfo", "einstellun", "bedienungshilfe",
    "favoriten", "kader", "zum ueberspringen", "zum überspringen",
    "ueberspringen halten", "überspringen halten", "partie nicht verfuegbar",
    "partie nicht verfügbar", "halbzeit", "half time", "4-1-2-1-2", "4-3-1-2",
    "4-2-1-3", "4-3-3", "4-4-2", "4-2-3-1", "4-2-2-2", "3-5-2", "5-3-2", "5-2-3",
}


def _label_family(label: str | None) -> str:
    """Collapse detector labels into event families for corroboration."""
    s = (label or "").lower().replace("-", "_").replace(" ", "_")
    if s.startswith("auto_"):
        s = s[5:]
    if any(k in s for k in ("kill", "headshot", "eliminiert", "eliminated", "ace")):
        return "kill"
    if any(k in s for k in ("spike", "plant", "defus", "entscharf", "bomb")):
        return "objective"
    if any(k in s for k in ("victory", "defeat", "verloren", "gewonnen", "round")):
        return "round"
    return s


def _same_event_family(a: str | None, b: str | None) -> bool:
    fa, fb = _label_family(a), _label_family(b)
    return bool(fa and fb and fa == fb)


def with_reaction(weights: dict[str, float]) -> dict[str, float]:
    """Scale audio weights down and add the facecam-reaction feature."""
    out = {k: v * (1.0 - REACTION_WEIGHT) for k, v in weights.items()
           if k != "reaction"}
    out["reaction"] = REACTION_WEIGHT
    return out


def audio_weights(profile_name: str | None) -> dict[str, float]:
    name = (profile_name or "generic").lower().replace(" ", "")
    return dict(GAMEPLAY_WEIGHTS.get(_GW_ALIAS.get(name, name), GAMEPLAY_WEIGHTS["generic"]))


def score_audio(features: dict, weights: dict[str, float]) -> tuple[int, list[ScoreFactor]]:
    """Score a gameplay clip from its audio features + (personalised) weights."""
    wsum = sum(float(features.get(k, 0.0)) * w for k, w in weights.items())
    score = int(round(max(1, min(99, 20 + wsum * 80))))
    contrib = sorted(((float(features.get(k, 0.0)) * w, k) for k, w in weights.items()),
                     reverse=True)
    factors: list[ScoreFactor] = []
    for pts, k in contrib:
        v = float(features.get(k, 0.0))
        if v < 0.2 or pts < 0.04 or k not in _GW_FACTOR:
            continue
        label, detail = _GW_FACTOR[k]
        factors.append(ScoreFactor(label=label, weight=round(pts * 80, 1),
                                   detail=detail.format(pct=int(v * 100))))
        if len(factors) >= 3:
            break
    if not factors:
        factors = [ScoreFactor(label="Audio activity", weight=round(wsum * 80, 1), detail="")]
    return score, factors


@dataclass
class GameplayClip:
    start: float
    end: float
    score: int
    factors: list[ScoreFactor] = field(default_factory=list)
    features: dict = field(default_factory=dict)   # signal values (for learning)
    title: str = "Highlight"
    peak_t: float = 0.0
    # Timestamps of the cue events inside this clip (announcer lines etc.) —
    # the caption stage mutes transcript words around them so in-game voices
    # don't end up burned into the captions.
    cue_ts: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _load_rms(wav_path: str, *, hop: float = 0.1, win: float = 0.25):
    """Short-time RMS energy envelope. Returns (times[], rms[]) or (None, None)."""
    try:
        import numpy as np
    except Exception:
        return None, None
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    if not raw:
        return None, None
    import numpy as np

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    data /= 32768.0
    hop_n = max(int(sr * hop), 1)
    win_n = max(int(sr * win), hop_n)
    n = max((len(data) - win_n) // hop_n, 0)
    if n < 3:
        return None, None
    times = np.arange(n) * hop / 1.0
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = data[i * hop_n: i * hop_n + win_n]
        rms[i] = float(np.sqrt((seg * seg).mean() + 1e-9))
    return times, rms


def _pick_peaks(times, rms, *, settings: ImportSettings, profile: dict):
    import numpy as np

    # Loudness threshold: well above the median (action vs. baseline).
    thr = float(np.percentile(rms, profile["thr_pct"]))
    floor = float(np.percentile(rms, 40)) + 1e-6
    min_gap = max(settings.min_len * profile["min_gap_factor"], 3.0)
    order = np.argsort(rms)[::-1]
    chosen: list[int] = []
    for idx in order:
        if rms[idx] < thr:
            break
        t = times[idx]
        if all(abs(t - times[c]) >= min_gap for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= settings.target_clips * 2:  # over-generate; trim later
            break
    return chosen, thr, floor


def _timing_window(settings: ImportSettings, profile: dict, *,
                   sustain_s: float = 0.0) -> tuple[float, float]:
    """Seconds before/after the event peak for gameplay clips.
    Enforces the Golden Climax Interval (72%-85% narrative pacing)."""
    target = (settings.min_len + settings.max_len) / 2.0
    lead_frac = profile.get("lead_frac", 0.75)
    default_lead = target * lead_frac
    default_tail = target - default_lead
    if sustain_s > 0.0:
        default_tail = max(default_tail, min(sustain_s + 1.2, target * 0.32))
        default_lead = max(settings.min_len * 0.68, target - default_tail)
    lead = default_lead if settings.lead_seconds is None else float(settings.lead_seconds)
    tail = default_tail if settings.tail_seconds is None else float(settings.tail_seconds)
    lead = max(0.0, lead)
    tail = max(0.0, tail)
    if lead + tail <= 0.2:
        lead, tail = default_lead, default_tail
    if lead + tail > settings.max_len:
        scale = settings.max_len / max(lead + tail, 1e-6)
        lead, tail = lead * scale, tail * scale
    if lead + tail < settings.min_len:
        tail = max(tail, settings.min_len - lead)
    return lead, tail


def _find_ocr_events(src_path: str, info: MediaInfo, settings: ImportSettings,
                     *, focus_times: list[float] | None = None,
                     warnings_out: list | None = None) -> list:
    if not settings.use_ocr:
        return []
    try:
        return detect_ocr.find_text_events(
            src_path, info, settings, focus_times=focus_times)
    except Exception as e:
        log.warning("ocr detection failed: %s", e)
        _add_warning(warnings_out,
                     "On-screen text detection (OCR) crashed, so no on-screen "
                     "events were found. Check the OCR backend install.")
        return []


def _add_warning(warnings_out: list | None, msg: str) -> None:
    """Record a non-fatal detector failure for the UI (deduped)."""
    if warnings_out is not None and msg not in warnings_out:
        warnings_out.append(msg)


def _track_hud(src_path: str, info: MediaInfo, settings: ImportSettings,
               warnings_out: list | None = None):
    """Track the scoreboard; the ``HudResult`` when it was read reliably, else None.

    ``None`` means "no scoreboard evidence": the caller then detects exactly as
    it would without the tracker. An unhealthy result is treated the same way,
    because goals from a scoreboard that was barely seen are not proof.
    """
    if not settings.use_ocr or not getattr(info, "has_video", False):
        return None
    if os.environ.get("CLIPFORGE_HUD", "").strip().lower() in _HUD_OFF_VALUES:
        return None
    if not hud_mod.layouts_for(settings.game_profile):
        return None
    try:
        # The timing pass is the expensive part; three candidates per wanted
        # clip is plenty to choose from once overlaps are dropped.
        result = hud_mod.track_scoreboard(
            src_path, info, settings,
            refine_limit=max(3 * settings.target_clips, 30))
    except Exception as e:
        log.warning("scoreboard tracking failed: %s", e)
        _add_warning(warnings_out,
                     "Scoreboard tracking crashed, so goals fall back to audio "
                     "and on-screen keywords.")
        return None
    from .hud.layouts import normalize_profile

    # Only worth telling the user when they said which game this is; with
    # "auto" a missing scoreboard just means it is some other game.
    named_game = normalize_profile(settings.game_profile) in hud_mod.LAYOUTS
    if result is None:
        if named_game:
            _add_warning(warnings_out,
                         "Scoreboard tracking could not read this video, so goals "
                         "fall back to audio and on-screen keywords.")
        return None
    if not result.healthy:
        if named_game:
            _add_warning(warnings_out,
                         "No scoreboard was found in this footage, so goals fall "
                         "back to audio and on-screen keywords. Scoreboard "
                         "tracking cannot read a scaled or moved game HUD yet.")
        return None
    if not named_game and (not result.events or result.diagnostics.get("violations", 0)):
        # Guessing the game: only trust a layout that produced clean goals.
        return None
    log.info("scoreboard tracked (%s): %d match(es), %d goal(s), coverage %.0f%%",
             result.layout, len(result.matches), len(result.events),
             result.coverage * 100.0)
    return result


def _legacy_ocr_with_hud(settings: ImportSettings) -> bool:
    cfg = getattr(settings, "game_config", None)
    return bool(getattr(cfg, "legacy_ocr_with_hud", False))


def _ocr_events_beside_hud(events: list, hud, gate: bool) -> list:
    """Keyword-OCR events that may stand next to a tracked scoreboard.

    Goal and scoreline hits are dropped (the tracker reads those properly), and
    with the gate on so is anything read while no scoreboard was on screen.
    """
    if hud is None:
        return events
    out = [e for e in events
           if str(getattr(e, "label", "") or "").lower() not in _HUD_SUPERSEDED_LABELS]
    if gate:
        out = [e for e in out if hud.is_in_play(float(e.t))]
    return out


def _menu_events(ocr_events: list, hud) -> list:
    """OCR events that say the player is in a menu.

    Text read while the scoreboard is on screen is never menu evidence: a
    stadium board reading CHAMPIONS is not the Champions menu.
    """
    out = [e for e in ocr_events if _is_menu_ocr_event(e)]
    if hud is not None:
        out = [e for e in out if not hud.is_in_play(float(e.t))]
    return out


def _hud_ocr_events(hud) -> list:
    """Scoreboard goals as on-screen evidence for :func:`apply_corroboration`."""
    if hud is None:
        return []
    return [detect_ocr.OcrEvent(t=float(ev.t), label="goal", text=ev.detail,
                                confidence=float(ev.confidence))
            for ev in hud.events]


def _cap_unverified(clips: list["GameplayClip"]) -> None:
    for c in clips:
        c.score = min(int(c.score), HUD_UNVERIFIED_CAP)


def _hud_goal_clips(hud, info: MediaInfo, settings: ImportSettings, *,
                    lead: float, tail: float) -> list["GameplayClip"]:
    """One clip per goal read off the scoreboard.

    ``ev.t`` already is the goal moment (the tracker took the scoreboard's
    delay out), so unlike :func:`_ocr_clips` no time offset is applied.
    """
    out: list[GameplayClip] = []
    for ev in hud.events:
        t = float(ev.t)
        start = max(0.0, t - lead)
        end = min(info.duration, t + tail)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        if end - start > settings.max_len:
            start = max(0.0, end - settings.max_len)
        confidence = float(ev.confidence)
        raw = 84.0 + float(ev.importance) - (8.0 if confidence < 0.7 else 0.0)
        score = int(round(max(float(HUD_GOAL_FLOOR), min(99.0, raw))))
        before = f"{ev.score_before[0]}-{ev.score_before[1]}"
        after = f"{ev.score_after[0]}-{ev.score_after[1]}"
        minute = f" ({ev.minute}')" if ev.minute is not None else ""
        detail = "Read off the on-screen scoreboard"
        if ev.clock:
            detail += f" at clock {ev.clock}"
        factors = [ScoreFactor(label=f"Scoreboard {before} → {after}{minute}",
                               weight=round(confidence * 20, 1), detail=detail)]
        if ev.reasons:
            factors.append(ScoreFactor(label="; ".join(ev.reasons[:3]),
                                       weight=float(ev.importance),
                                       detail="What this goal did to the match"))
        m, s = divmod(int(t), 60)
        out.append(GameplayClip(
            start=round(start, 3), end=round(end, 3), score=score, peak_t=t,
            title=f"Goal {after} — {m}:{s:02d}", cue_ts=[],
            features={"ocr": round(confidence, 4), "hud": 1.0,
                      "hud_importance": float(ev.importance), "intensity": 0.0,
                      "sustain": 0.0, "transient": 0.0, "spikes": 0.0, "cue": 0.0,
                      "climax_t": round(t, 3)},
            factors=factors))
    return out


_KILL_TITLES = {1: "Kill", 2: "Double kill", 3: "Triple kill", 4: "Quadra kill", 5: "Ace"}
_KILL_SCORES = {1: 76, 2: 86, 3: 91, 4: 95, 5: 98}


def _track_killfeed(src_path: str, info: MediaInfo, settings: ImportSettings,
                    warnings_out: list | None = None):
    """Kill-banner / kill-feed result for games that have one (Valorant), else None."""
    from .hud import killfeed

    if not settings.use_ocr or not getattr(info, "has_video", False):
        return None
    if os.environ.get("CLIPFORGE_HUD", "").strip().lower() in _HUD_OFF_VALUES:
        return None
    if not killfeed.supports(settings.game_profile):
        return None
    try:
        result = killfeed.track_killfeed(src_path, info)
    except Exception as e:
        log.warning("kill-feed tracking failed: %s", e)
        result = None
    if result is None or not result.healthy:
        _add_warning(warnings_out,
                     "The kill feed could not be read in this footage, so kills fall "
                     "back to game sounds and on-screen keywords.")
        return None
    return result


def _killfeed_clips(kills, info: MediaInfo, settings: ImportSettings, *,
                    lead: float, tail: float) -> list["GameplayClip"]:
    """One clip per round in which the streamer got kills, multi-kills first."""
    out: list[GameplayClip] = []
    for rnd in kills.rounds:
        n = min(rnd.count, 5)
        t_last = float(rnd.t_last)
        end = min(info.duration, t_last + tail)
        # Start before the first kill when the whole streak fits in one clip.
        start = max(0.0, min(float(rnd.t_first) - 3.0, t_last - lead))
        if end - start > settings.max_len:
            start = max(0.0, end - settings.max_len)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        confidence = min(float(k.confidence) for k in rnd.kills)
        quick = rnd.count >= 2 and rnd.span <= 6.0
        score = min(99, _KILL_SCORES[n] + (2 if quick else 0))
        factors = [ScoreFactor(
            label=f"Kill banner: {rnd.count} kill{'s' if rnd.count != 1 else ''} in one round",
            weight=round(confidence * 20, 1),
            detail="Read off the on-screen kill banner and kill feed")]
        if quick:
            factors.append(ScoreFactor(label=f"{rnd.count} kills in {rnd.span:.0f} s",
                                       weight=2.0, detail="Kills in quick succession"))
        m, s = divmod(int(rnd.t_first), 60)
        out.append(GameplayClip(
            start=round(start, 3), end=round(end, 3), score=score, peak_t=t_last,
            title=f"{_KILL_TITLES[n]} — {m}:{s:02d}", cue_ts=[],
            features={"ocr": round(confidence, 4), "hud": 1.0, "kills": float(rnd.count),
                      "intensity": 0.0, "sustain": 0.0, "transient": 0.0, "spikes": 0.0,
                      "cue": 0.0, "climax_t": round(t_last, 3)},
            factors=factors))
    return out


def _add_goal_loudness(clips: list["GameplayClip"], times, rms, *,
                       before: float = 1.0, after: float = 6.0) -> None:
    """Break ties between goals by how loud the moment was.

    With a hundred goals in a session many share a score. The crowd, the
    commentary and above all the streamer are louder for the ones that
    mattered to them, so a goal whose seconds are clearly above the
    footage's typical level gains up to ``HUD_LOUDNESS_BONUS`` points: nothing
    below +1 dB, the full bonus from +8 dB. Loudness only orders goals here;
    it never creates one.

    The level is the mean over the window from ``before`` seconds ahead of the
    goal to ``after`` seconds past it, compared with the median of that same
    mean over the whole footage. (The window's *peak* would not do: some
    sample in any seven seconds is loud, so nearly every goal would qualify.)
    """
    import math

    import numpy as np

    if not clips or times is None or rms is None or len(rms) < 2:
        return
    hop = float(times[1] - times[0]) or 0.1
    n_win = max(int(round((before + after) / hop)), 1)
    csum = np.concatenate(([0.0], np.cumsum(rms, dtype=np.float64)))
    if len(csum) <= n_win:
        return
    roll = (csum[n_win:] - csum[:-n_win]) / float(n_win)   # mean of the window starting at i
    typical = max(float(np.median(roll)), 1e-9)
    ranked = np.sort(roll)
    for c in clips:
        i0 = min(max(int((c.peak_t - before) / hop), 0), len(roll) - 1)
        level = max(float(roll[i0]), 1e-9)
        gain_db = 20.0 * math.log10(level / typical)
        pct = float(np.searchsorted(ranked, level, side="right")) / float(len(ranked))
        c.features["intensity"] = round(pct, 4)
        bonus = int(round(HUD_LOUDNESS_BONUS * min(max((gain_db - 1.0) / 7.0, 0.0), 1.0)))
        if bonus > 0:
            c.score = int(min(99, c.score + bonus))
            c.factors.append(ScoreFactor(
                label="Loud moment", weight=float(bonus),
                detail=f"{gain_db:+.0f} dB against the footage's typical level"))


def detect_gameplay(src_path: str, info: MediaInfo, settings: ImportSettings,
                    *, weights: dict[str, float] | None = None,
                    wav_path: str | None = None,
                    events_out: list | None = None,
                    warnings_out: list | None = None) -> list[GameplayClip]:
    """Return intensity-ranked highlight clips for gameplay footage.

    ``weights`` lets the caller pass personalised audio-feature weights (from the
    learning loop); otherwise the per-game defaults are used. ``wav_path`` is an
    already-extracted 16 kHz mono wav of the source (the pipeline reuses the
    transcription extract); when absent the audio is extracted here. ``events_out``
    (when given) is extended with the :class:`DetectedEvent`s that were matched
    (audio cues + OCR) so the caller can persist them on the project.

    The scoreboard is tracked first (:func:`_track_hud`). With a result, its
    goals lead the returned clips on every path through this function, and
    with enough coverage the other detectors only count during match time.
    """
    import tempfile
    from pathlib import Path

    aw = weights or audio_weights(settings.game_profile)
    profile = get_profile(settings.game_profile)

    hud = _track_hud(src_path, info, settings, warnings_out)
    # gate: the scoreboard covers enough of the footage that its absence means
    # "not in a match" — menus, lobbies, replays between games.
    gate = hud is not None and hud.coverage >= HUD_GATE_MIN_COVERAGE
    goal_clips: list[GameplayClip] = []
    if hud is not None:
        goal_lead, goal_tail = _timing_window(settings, profile)
        goal_clips = _hud_goal_clips(hud, info, settings,
                                     lead=goal_lead, tail=goal_tail)
    # Kills read off the kill banner are on-screen proof just like scoreboard
    # goals: they lead the result the same way.
    kills = _track_killfeed(src_path, info, settings, warnings_out)
    if kills is not None:
        kill_lead, kill_tail = _timing_window(settings, profile)
        goal_clips = goal_clips + _killfeed_clips(kills, info, settings,
                                                  lead=kill_lead, tail=kill_tail)
        if events_out is not None:
            for ev in kills.events:
                events_out.append(DetectedEvent(
                    t=round(float(ev.t), 3), source="ocr", label=ev.kind,
                    detail=ev.detail, confidence=round(float(ev.confidence), 3),
                    data={"detector": "killfeed", "round": ev.round_index,
                          "streak": ev.streak, "sources": list(ev.sources)}))

    # On-screen text markers (VICTORY / ELIMINATED / GOAL / kill-feed).
    # For footage with audio we run OCR after finding likely moments, so it reads
    # the frames that matter instead of doing a slow blind sweep over the VOD.
    # With a tracked scoreboard the keyword sweep is off unless asked for.
    use_ocr = bool(settings.use_ocr
                   and (hud is None or _legacy_ocr_with_hud(settings)))
    ocr_events: list = []
    defer_ocr = bool(use_ocr and info.has_audio)
    if use_ocr and not defer_ocr:
        ocr_events = _ocr_events_beside_hud(
            _find_ocr_events(src_path, info, settings, warnings_out=warnings_out),
            hud, gate)

    if not info.has_audio:
        clips = _uniform_fallback(info, settings)
        ocr_clips = _ocr_clips(ocr_events, info, settings)
        _record_events(events_out, [], ocr_events, [], hud=hud)
        if hud is not None:
            _cap_unverified(ocr_clips)
        priority = goal_clips + ocr_clips
        return _merge(priority, clips, settings.target_clips) if priority else clips

    cue_events: list = []
    audio_window_events: list = []
    if wav_path is not None:
        times, rms = _load_rms(wav_path)
        cue_events = _cue_events(wav_path, settings)
        audio_window_events = _audio_events(wav_path, info.duration, settings,
                                            warnings_out=warnings_out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            try:
                ffmpeg.extract_audio_wav(src_path, wav)
            except Exception as e:
                log.warning("gameplay audio extract failed: %s", e)
                fallback = _uniform_fallback(info, settings)
                if hud is None:
                    return fallback
                # The scoreboard needs no audio: its goals survive this.
                _record_events(events_out, [], [], [], hud=hud)
                return (_merge(goal_clips, fallback, settings.target_clips)
                        if goal_clips else fallback)
            times, rms = _load_rms(str(wav))
            cue_events = _cue_events(str(wav), settings)
            audio_window_events = _audio_events(str(wav), info.duration, settings,
                                                warnings_out=warnings_out)

    if gate:
        cue_events = [e for e in cue_events if hud.is_in_play(float(e.t))]
        audio_window_events = [e for e in audio_window_events
                               if hud.is_in_play(float(e.t))]

    if defer_ocr:
        focus_times = [float(e.t) for e in cue_events]
        focus_times += [float(e.t) for e in audio_window_events]
        if times is not None and rms is not None:
            try:
                peak_idxs, _thr, _floor = _pick_peaks(
                    times, rms, settings=settings, profile=profile)
                focus_times += [float(times[i]) for i in peak_idxs
                                if not gate or hud.is_in_play(float(times[i]))]
            except Exception as e:
                log.warning("ocr focus peaks failed: %s", e)
        ocr_events = _ocr_events_beside_hud(
            _find_ocr_events(src_path, info, settings, focus_times=focus_times,
                             warnings_out=warnings_out),
            hud, gate)

    menu_events = _menu_events(ocr_events, hud)
    audio_window_events = _suppress_audio_events_near_menu(
        audio_window_events, menu_events)
    _record_events(events_out, cue_events, ocr_events, audio_window_events, hud=hud)

    if times is None:
        clips = _uniform_fallback(info, settings)
        ocr_clips = _ocr_clips(ocr_events, info, settings)
        audio_clips = _audio_event_clips(audio_window_events, info, settings)
        if hud is not None:
            _cap_unverified(ocr_clips + audio_clips)
        priority = goal_clips + ocr_clips + audio_clips
        return _merge(priority, clips, settings.target_clips) if priority else clips


    peaks, thr, floor = _pick_peaks(times, rms, settings=settings, profile=profile)
    if not peaks:
        lead, tail = _timing_window(settings, profile)
        unverified = (_cue_clips(cue_events, info, lead, tail, settings)
                      + _ocr_clips(ocr_events, info, settings, lead=lead, tail=tail)
                      + _audio_event_clips(audio_window_events, info, settings,
                                           lead=lead, tail=tail))
        if hud is not None:
            _cap_unverified(unverified)
        priority = goal_clips + unverified
        fallback = _uniform_fallback(info, settings)
        return _merge(priority, fallback, settings.target_clips) if priority else fallback

    lead, tail = _timing_window(settings, profile)
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.1
    pre_n = max(int(2.0 / hop), 1)         # ~2s window before the peak

    menu_times = [float(e.t) for e in menu_events]
    _add_goal_loudness(goal_clips, times, rms)
    # Hold loud moments to match time — unless that would leave nothing at
    # all (no goals, and every loud moment outside a match): an empty result
    # fails the whole project, so then the gate stays open.
    gate_peaks = gate and (bool(goal_clips)
                           or any(hud.is_in_play(float(times[i])) for i in peaks))
    clips: list[GameplayClip] = []
    for idx in peaks:
        pt = float(times[idx])
        if gate_peaks and not hud.is_in_play(pt):
            continue
        if menu_times and any(abs(pt - mt) <= MENU_CONTEXT_WINDOW * 2.0 for mt in menu_times):
            continue
        start = max(0.0, pt - lead)
        end = min(info.duration, pt + tail)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        # intensity 0..1 = how this peak ranks across the whole envelope.
        # Inclusive (<=) so the single loudest frame reaches 1.0 instead of
        # being capped below the top.
        pct = float((rms <= rms[idx]).mean())
        # sustain: share of the window above the action threshold
        w0 = int(start / hop); w1 = int(end / hop)
        window = rms[w0:w1] if w1 > w0 else rms[idx:idx + 1]
        sustain = float((window > thr).mean()) if len(window) else 0.0
        spikes = _count_spikes(window, thr)
        # transient: how much the peak jumps above the ~2s just before it.
        # Guard the slice so an early peak (idx near 0) can't collapse it to an
        # empty/degenerate range and silently fall back to the global floor.
        lo = max(idx - pre_n, 0)
        hi = max(idx - 1, lo + 1)
        pre = rms[lo:hi]
        base = float(pre.mean()) if len(pre) else floor
        transient = float(max(0.0, (rms[idx] - base) / (rms[idx] + 1e-6)))
        # Horror jump-scare surge: reward dramatic silence-to-spike transitions
        if profile.get("w_transient", 0) >= 15.0 and base < floor * 1.5:
            transient = min(1.0, transient * 1.35)

        feats = {"intensity": round(pct, 4), "sustain": round(sustain, 4),
                 "transient": round(transient, 4),
                 "spikes": round(min(spikes / 5.0, 1.0), 4), "cue": 0.0,
                 "reaction": 0.0, "climax_t": round(pt, 3)}
        score, factors = score_audio(feats, aw)
        clips.append(GameplayClip(start=round(start, 3), end=round(end, 3),
                                  score=score, factors=factors, peak_t=pt, features=feats))

    for i, c in enumerate(clips, 1):
        m, s = divmod(int(c.peak_t), 60)
        c.title = f"Highlight — {m}:{s:02d}"

    # Exact-event clips (matched game sounds + OCR banners) take priority over
    # plain loudness peaks: a "VICTORY" banner or a kill ding is a sure thing.
    cue_clips = _cue_clips(cue_events, info, lead, tail, settings)
    ocr_clips = _ocr_clips(ocr_events, info, settings, lead=lead, tail=tail)
    audio_clips = _audio_event_clips(audio_window_events, info, settings,
                                     lead=lead, tail=tail)
    unverified = cue_clips + ocr_clips + audio_clips
    priority = goal_clips + unverified
    # Multimodal corroboration: reward corroborated events BEFORE selection.
    # A goal read off the scoreboard is on-screen evidence like any OCR hit.
    apply_corroboration(priority + clips, cue_events,
                        ocr_events + _hud_ocr_events(hud), audio_window_events)
    if hud is not None:
        # After corroboration, so a bonus cannot lift them past the goals.
        _cap_unverified(unverified)
    merged = _merge(priority, clips, settings.target_clips)
    merged.sort(key=lambda c: c.start)
    return merged


def apply_corroboration(clips: list["GameplayClip"], cue_events: list,
                        ocr_events: list, audio_events: list | None = None) -> None:
    """Bump clips whose window is backed by more than one signal source.

    A kill *ding* + a kill-feed line, a goal *roar* + a rising score — when the
    audio cue and the OCR banner agree on the same instant, it's a guaranteed
    beat. Mutates clips in place: small score bonus + an explainable factor."""
    for c in clips:
        cues = _clustered_events(c, cue_events)
        ocrs = _clustered_events(c, ocr_events)
        audios = _clustered_events(c, audio_events or [])
        matching_ocrs = [
            o for o in ocrs
            if any(_same_event_family(getattr(q, "label", None),
                                      getattr(o, "label", None)) for q in cues)
        ]
        exact_confirm = bool(cues and matching_ocrs)
        soft_support = bool(not cues and ocrs and audios)
        if exact_confirm:
            active = [name for name, evs in (("cue", cues), ("on-screen", matching_ocrs),
                                             ("CLAP", audios)) if evs]
            count = len(cues) + len(matching_ocrs) + len(audios)
            bonus = min(14, 6 + 2 * max(count - 2, 0) + 2 * (len(active) - 2))
            c.score = int(max(1, min(99, c.score + bonus)))
            c.features["corroborated"] = 1.0
            c.features["cue"] = round(max(
                float(c.features.get("cue", 0.0)),
                max(float(getattr(q, "similarity", 0.0)) for q in cues),
            ), 4)
            c.factors.insert(0, ScoreFactor(
                label="Multimodal confirm", weight=float(bonus),
                detail="A matching game sound and matching on-screen text fire here"))
        elif soft_support:
            bonus = 3
            # A bonus must never lower the score: an already-high clip (e.g. cue
            # + reaction at 95) shouldn't be clamped down by the "support" cap.
            if c.score < 89:
                c.score = int(min(89, c.score + bonus))
                c.features["supporting_audio"] = 1.0
                c.factors.insert(0, ScoreFactor(
                    label="Audio supports OCR", weight=float(bonus),
                    detail="Generic audio energy lines up with on-screen text, but no exact cue matched"))


def _anchor_times(c) -> list[float]:
    feats = getattr(c, "features", {}) or {}
    base = feats.get("evidence_t", getattr(c, "peak_t", getattr(c, "start", 0.0)))
    anchors = [float(base)]
    anchors += [float(t) for t in getattr(c, "cue_ts", []) or []]
    return anchors


def _clustered_events(c, events: list, *, window: float = CONFIRM_WINDOW) -> list:
    anchors = _anchor_times(c)
    return [
        e for e in events
        if c.start <= float(e.t) <= c.end
        and any(abs(float(e.t) - a) <= window for a in anchors)
    ]


def accepted_events_for_clips(events: list[DetectedEvent],
                              clips: list["GameplayClip"] | list,
                              *, window: float = CONFIRM_WINDOW
                              ) -> list[DetectedEvent]:
    """Return only detector events that actually supported a final clip."""
    out: list[DetectedEvent] = []
    seen: set[tuple[str, str, int]] = set()
    for c in clips:
        clustered = _clustered_events(c, events, window=window)
        feats = getattr(c, "features", {}) or {}
        has_cue = float(feats.get("cue", 0.0)) > 0.0 or bool(getattr(c, "cue_ts", []))
        has_ocr = float(feats.get("ocr", 0.0)) > 0.0
        has_audio = (
            float(feats.get("audio_event", 0.0)) > 0.0
            or float(feats.get("supporting_audio", 0.0)) > 0.0
        )
        if has_cue and has_ocr:
            clustered = [
                e for e in clustered
                if e.source != "cue" or any(
                    _same_event_family(e.label, other.label)
                    for other in clustered if other.source == "ocr")
            ]
        elif not has_cue:
            clustered = [e for e in clustered if e.source != "cue"]
        if not has_ocr:
            clustered = [e for e in clustered if e.source != "ocr"]
        if not has_audio:
            clustered = [e for e in clustered if e.source != "audio"]
        for e in clustered:
            key = (e.source, e.label, int(round(float(e.t) * 1000)))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    out.sort(key=lambda e: e.t)
    return out


def _record_events(events_out: list | None, cue_events: list,
                   ocr_events: list, audio_events: list | None = None,
                   *, hud=None) -> None:
    """Append matched cues + OCR hits to ``events_out`` as DetectedEvents.

    ``hud`` is the scoreboard result, if any: each of its goals is recorded
    too. Their source stays "ocr" (they were read off the screen, and the
    front-end only knows cue / ocr / audio); ``data["detector"] == "hud"`` is
    what tells them apart from keyword hits.
    """
    if events_out is None:
        return
    for ev in (hud.events if hud is not None else []):
        events_out.append(DetectedEvent(
            t=round(float(ev.t), 3), source="ocr", label="goal",
            detail=ev.detail, confidence=round(float(ev.confidence), 3),
            data={"detector": "hud", "team": ev.team,
                  "score_before": f"{ev.score_before[0]}-{ev.score_before[1]}",
                  "score_after": f"{ev.score_after[0]}-{ev.score_after[1]}",
                  "minute": ev.minute, "clock": ev.clock,
                  "t_change": round(float(ev.t_change), 3),
                  "streamer": ev.streamer,
                  "importance": float(ev.importance)}))
    for e in cue_events:
        events_out.append(DetectedEvent(
            t=round(float(e.t), 3), source="cue", label=e.label,
            detail=f"{e.label} sound", confidence=round(float(e.similarity), 3)))
    for e in ocr_events:
        events_out.append(DetectedEvent(
            t=round(float(e.t), 3), source="ocr", label=e.label,
            detail=e.text, confidence=round(float(e.confidence), 3)))
    for e in audio_events or []:
        events_out.append(DetectedEvent(
            t=round(float(e.t), 3), source="audio", label=e.label,
            detail=e.detail, confidence=round(float(e.confidence), 3)))
    events_out.sort(key=lambda e: e.t)


def _is_menu_ocr_event(event) -> bool:
    label = str(getattr(event, "label", "") or "").lower()
    if label in MENU_OCR_LABELS:
        return True
    detail = str(getattr(event, "text", "") or getattr(event, "detail", "") or "").lower()
    # If labelled scoreboard, drop if it matches menu words or lacks actual score numbers
    if label == "scoreboard":
        if any(word in detail for word in MENU_OCR_WORDS):
            return True
        if not any(c.isdigit() for c in detail):
            return True
    if any(word in detail for word in MENU_OCR_WORDS):
        return True
    import re
    if re.search(r'\b\d-\d-\d(-\d)?\b', detail):
        return True
    card_attrs = sum(1 for attr in ("tem", "sch", "pas", "dri", "def", "phy") if f" {attr} " in f" {detail} ")
    if card_attrs >= 2:
        return True
    return False


def _suppress_audio_events_near_menu(audio_events: list, ocr_events: list,
                                     *, window: float = MENU_CONTEXT_WINDOW) -> list:
    """Drop generic CLAP hits when OCR says the player is in menus/lobbies."""
    if not audio_events or not ocr_events:
        return audio_events
    menu_times = [float(e.t) for e in ocr_events if _is_menu_ocr_event(e)]
    if not menu_times:
        return audio_events
    kept = []
    for e in audio_events:
        t = float(e.t)
        if any(abs(t - mt) <= window for mt in menu_times):
            log.info("suppressed CLAP event %s at %.1fs near menu OCR",
                     getattr(e, "label", "unknown"), t)
            continue
        kept.append(e)
    return kept


# On-screen markers that mean the moment already happened — open a little before
# so the play that earned the banner is in-frame, with less tail.
def _ocr_clips(events: list, info: MediaInfo, settings: ImportSettings, *,
               lead: float | None = None, tail: float | None = None) -> list["GameplayClip"]:
    if not events:
        return []
    target = (settings.min_len + settings.max_len) / 2.0
    if lead is None:
        lead = target * 0.75   # banners are the payoff — 75% setup
    if tail is None:
        tail = max(target - lead, 2.5)
    out: list[GameplayClip] = []
    for e in events:
        if _is_menu_ocr_event(e):
            continue
        lbl = str(getattr(e, "label", "") or "").lower()
        t_event = e.t
        # EA FC / Sports broadcast delay calibration: scoreboard update occurs ~4.5s AFTER the ball crossed the line
        if lbl in {"goal", "scoreboard"} and getattr(settings, "game_profile", "auto") in {"eafc", "fifa", "rocketleague", "auto"}:
            t_event = max(0.0, e.t - 4.5)

        start = max(0.0, t_event - lead)
        end = min(info.duration, t_event + tail)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        if end - start > settings.max_len:
            start = max(0.0, end - settings.max_len)
        # Goals and critical events score top-tier; generic scoreboard changes score moderately
        if lbl in {"goal", "victory", "ace", "clutch"}:
            base_score = 88
        elif lbl in {"scoreboard"}:
            base_score = 74
        else:
            base_score = 78
        score = int(round(max(1, min(99, base_score + e.confidence * 11))))
        m, s = divmod(int(e.t), 60)
        label = e.label.replace("_", " ").title()
        out.append(GameplayClip(
            start=round(start, 3), end=round(end, 3), score=score, peak_t=t_event,
            title=f"{label} — {m}:{s:02d}",
            features={"ocr": round(e.confidence, 4), "intensity": 0.0,
                      "sustain": 0.0, "transient": 0.0, "spikes": 0.0, "cue": 0.0},
            factors=[ScoreFactor(
                label=f"On-screen '{e.text}'", weight=round(e.confidence * 20, 1),
                detail="Detected viral on-screen text (OCR) — a guaranteed beat")]))
    return out


def _audio_event_clips(events: list, info: MediaInfo, settings: ImportSettings, *,
                       lead: float | None = None,
                       tail: float | None = None) -> list["GameplayClip"]:
    if not events:
        return []
    target = (settings.min_len + settings.max_len) / 2.0
    if lead is None:
        lead = target * 0.75   # audio climax — 75% setup
    if tail is None:
        tail = max(target - lead, 2.5)
    out: list[GameplayClip] = []
    for e in events:
        start = max(0.0, e.t - lead)
        end = min(info.duration, e.t + tail)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        if end - start > settings.max_len:
            start = max(0.0, end - settings.max_len)
        score = int(round(max(1, min(99, 62 + e.confidence * 24))))
        m, s = divmod(int(e.t), 60)
        label = e.label.replace("_", " ").title()
        out.append(GameplayClip(
            start=round(start, 3), end=round(end, 3), score=score, peak_t=e.t,
            title=f"{label} - {m}:{s:02d}",
            features={"audio_event": round(e.confidence, 4), "intensity": 0.0,
                      "sustain": 0.0, "transient": 0.0, "spikes": 0.0,
                      "cue": 0.0},
            factors=[ScoreFactor(
                label="Zero-shot audio cue", weight=round(e.confidence * 24, 1),
                detail=e.detail or "CLAP matched a high-energy audio cue")]))
    return out


def _cue_dir(profile_name: str) -> str:
    name = (profile_name or "generic").lower().replace(" ", "")
    return _CUE_DIR.get(name, name)


def _cue_events(wav_path: str, settings: ImportSettings) -> list:
    """Match reference game sounds from <data>/game_cues/<profile>/ (+ /common),
    plus any per-project reference_audio_files named in the game config."""
    if not getattr(settings, "use_cues", True):
        return []
    base = get_settings().data_dir / "game_cues"
    events = []
    prof = getattr(settings, "game_profile", "auto")
    kw = {"threshold": 0.88} if prof in {"eafc", "fifa", "rocketleague", "sports"} else {}
    for sub in {_cue_dir(settings.game_profile), "common"}:
        try:
            try:
                events += detect_cues.find_events(wav_path, base / sub, **kw)
            except TypeError:
                events += detect_cues.find_events(wav_path, base / sub)
        except Exception as e:
            log.warning("cue matching failed for %s: %s", sub, e)
    extra = _reference_cue_files(settings, base)
    if extra:
        try:
            try:
                events += detect_cues.match_templates(wav_path, extra, **kw)
            except TypeError:
                events += detect_cues.match_templates(wav_path, extra)
        except Exception as e:
            log.warning("reference-audio cue matching failed: %s", e)
    return events


def _reference_cue_files(settings: ImportSettings, base) -> list:
    """Resolve game_config.reference_audio_files to on-disk cue templates.

    Entries may be a bare filename, a "<profile>/<file>" path under the cue
    store, or an absolute/relative path. Missing files are skipped (safe no-op).
    """
    from pathlib import Path
    cfg = getattr(settings, "game_config", None)
    names = getattr(cfg, "reference_audio_files", None) if cfg else None
    if not names:
        return []
    prof_dir = _cue_dir(settings.game_profile)
    out: list = []
    seen: set = set()
    for name in names:
        name = (name or "").strip()
        if not name:
            continue
        for cand in (Path(name), base / name, base / prof_dir / name,
                     base / "common" / name):
            try:
                if cand.is_file():
                    key = str(cand.resolve())
                    if key not in seen:
                        seen.add(key)
                        out.append(cand)
                    break
            except OSError:
                continue
    return out


def _audio_events(wav_path: str, duration: float, settings: ImportSettings,
                  *, warnings_out: list | None = None) -> list:
    if not settings.use_audio_events:
        return []
    cfg = getattr(settings, "game_config", None)
    # "manual" detection relies solely on the user's own cues (reference audio
    # + on-screen text), so the zero-shot CLAP sweep is intentionally skipped.
    detection_mode = getattr(cfg, "detection_mode", "zero_shot")
    if detection_mode == "manual":
        return []
    try:
        mode = getattr(settings.power_mode, "value", str(settings.power_mode))
        hop = 4.0 if mode in ("max_gpu", "quality") else 5.0
        window = 5.0 if mode == "max_gpu" else 6.0
        threshold = 0.55 if mode == "quality" else 0.58
        # "hybrid" pairs the user's own cues with the auto sweep, so it only
        # wants the strong auto-hits — raise the bar to cut false positives the
        # manual cues would otherwise have to compete with.
        if detection_mode == "hybrid":
            threshold += 0.04
        return audio_events_mod.find_events(
            wav_path, duration, window=window, hop=hop, threshold=threshold,
            limit=max(settings.target_clips, 6), profile=settings.game_profile,
            language=settings.language,
            positive_prompts=getattr(cfg, "audio_prompts", None),
            negative_prompts=getattr(cfg, "audio_negative_prompts", None))
    except Exception as e:
        log.warning("CLAP audio event search failed: %s", e)
        _add_warning(warnings_out,
                     "Audio-event detection (CLAP) crashed, so no audio "
                     "highlights were found. Check the audio model install.")
        return []


# Cue hits within this many seconds of each other chain into one streak —
# a multi-kill, a brace, a plant-into-retake. Streaks complete better than
# single events, and completion is what the feed algorithms reward.
STREAK_WINDOW = 6.0
MAX_STREAK_EVENTS = 5
_STREAK_WORDS = {2: "Double", 3: "Triple", 4: "Quad", 5: "Penta"}
_KILL_CHAIN_LABELS = {
    "kill", "double_kill", "triple_kill", "quad_kill", "ace", "headshot",
    "one_kill", "two_kills", "three_kills", "four_kills", "five_kills",
}


def _streak_compatible(prev, curr) -> bool:
    if prev.label == curr.label:
        return True
    return prev.label in _KILL_CHAIN_LABELS and curr.label in _KILL_CHAIN_LABELS


def group_streaks(events: list, *, window: float = STREAK_WINDOW) -> list[list]:
    """Chain time-sorted cue events into groups of quick succession."""
    groups: list[list] = []
    for ev in sorted(events, key=lambda e: e.t):
        if (groups and len(groups[-1]) < MAX_STREAK_EVENTS
                and ev.t - groups[-1][-1].t <= window
                and _streak_compatible(groups[-1][-1], ev)):
            groups[-1].append(ev)
        else:
            groups.append([ev])
    return groups


def _cue_clips(events: list, info: MediaInfo, lead: float, tail: float,
               settings: ImportSettings) -> list["GameplayClip"]:
    out: list[GameplayClip] = []
    for grp in group_streaks(events):
        first, last = grp[0], grp[-1]
        n = len(grp)
        sim = max(e.similarity for e in grp)
        start = max(0.0, first.t - lead)
        end = min(info.duration, last.t + tail)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        if end - start > settings.max_len:
            # Keep the kills, sacrifice build-up: action > context for hooks.
            start = max(first.t - 2.0, end - settings.max_len)

        # Cue-only evidence is useful, but it is not proof by itself. Top scores
        # require corroboration from OCR/visual/audio/reaction later in the pipe.
        # Multi-kill exponential streak multipliers (2k: 1.3x, 3k: 1.8x, 4k: 2.4x, 5k: 3.5x ACE)
        streak_mult = 1.0
        if n == 2:
            streak_mult = 1.3
        elif n == 3:
            streak_mult = 1.8
        elif n == 4:
            streak_mult = 2.4
        elif n >= 5:
            streak_mult = 3.5

        cue_strength = max(0.0, sim - 0.84)
        streak_bonus = (n - 1) * 4 * streak_mult
        max_cap = 99 if n >= 3 else 88
        score = int(round(max(1, min(max_cap, 58 + cue_strength * 120 + streak_bonus))))
        m, s = divmod(int(first.t), 60)
        clean_name = first.label.removeprefix("auto_").replace("_", " ").title()
        label = clean_name
        labels = {e.label.removeprefix("auto_") for e in grp}
        if n == 1:
            title = f"{label} — {m}:{s:02d}"
        elif len(labels) == 1:
            word = _STREAK_WORDS.get(n, f"{n}x")
            title = f"{word} {label} — {m}:{s:02d}"
        else:
            title = f"{label} + {n - 1} cues — {m}:{s:02d}"

        factors = [ScoreFactor(label=f"Matched '{clean_name}' sound",
                               weight=round(max(0.0, sim - 0.80) * 50, 1),
                               detail=f"{int(sim*100)}% audio match to your reference cue")]
        if n >= 2:
            span = max(last.t - first.t, 1.0)
            factors.insert(0, ScoreFactor(
                label=f"{n} events in {span:.0f}s — streak",
                weight=round((n - 1) * 4, 1),
                detail="Chained action holds viewers to the end (multi-kill effect)"))

        out.append(GameplayClip(
            start=round(start, 3), end=round(end, 3), score=score, peak_t=first.t,
            title=title, cue_ts=[e.t for e in grp],
            features={"cue": round(sim, 4), "streak": round(min(n / 5.0, 1.0), 4),
                      "intensity": 0.0, "sustain": 0.0, "transient": 0.0,
                      "spikes": 0.0},
            factors=factors))
    return out


def apply_evidence_caps(clips: list["GameplayClip"] | list) -> None:
    """Prevent one weak signal from maxing virality by itself."""
    for c in clips:
        feats = getattr(c, "features", {}) or {}
        cue = float(feats.get("cue", 0.0))
        ocr = float(feats.get("ocr", 0.0))
        reaction = float(feats.get("reaction", 0.0))
        excitement = float(feats.get("excitement", 0.0))
        corroborated = float(feats.get("corroborated", 0.0)) > 0.0
        # A goal read off the scoreboard is proof on its own.
        scoreboard = float(feats.get("hud", 0.0)) > 0.0
        exceptional = (
            corroborated
            or scoreboard
            or (cue >= 0.88 and ocr >= 0.82)
            or (reaction >= 0.55 and (ocr >= 0.82 or excitement >= 0.72 or cue >= 0.88))
        )
        if not exceptional:
            c.score = min(int(c.score), 89)
        if feats.get("cue", 0.0) > 0.0 and not exceptional:
            c.score = min(int(c.score), 88)
        audio_only = (
            feats.get("audio_event", 0.0) > 0.0
            and cue <= 0.0
            and ocr <= 0.0
            and reaction < 0.55
        )
        if audio_only and not exceptional:
            c.score = min(int(c.score), 86)


def apply_title_fallbacks(clips: list["GameplayClip"] | list, profile: str = "generic") -> None:
    """Replace generic peak titles with the strongest visible/audio reason."""
    prof = (profile or "generic").lower().replace(" ", "")
    is_sports = prof in {"eafc", "fifa", "rocketleague", "sports"}
    for c in clips:
        title = getattr(c, "title", "") or ""
        if not title.lower().startswith("highlight"):
            continue
        feats = getattr(c, "features", {}) or {}
        label = "Highlight"
        has_cam = feats.get("facecam") is not None or float(feats.get("reaction", 0.0)) > 0.65
        if float(feats.get("reaction", 0.0)) > 0.65 and has_cam:
            label = "Streamer Reaction" if is_sports else "Big Reaction"
        elif float(feats.get("transient", 0.0)) > 0.65 or float(feats.get("spikes", 0.0)) > 0.7:
            label = "Big Chance" if is_sports else "Sudden Fight"
        elif float(feats.get("intensity", 0.0)) > 0.85:
            label = "Stadium Hype" if is_sports else "Loud Fight"
        elif float(feats.get("audio_event", 0.0)) > 0.25:
            label = _audio_title_from_factors(getattr(c, "factors", []) or [])
        elif float(feats.get("vlm_viral", 0.0)) > 0.55:
            label = "Visual Moment"
        parts = title.rsplit("—", 1)
        if len(parts) == 2 and parts[1].strip():
            c.title = f"{label} — {parts[1].strip()}"
        else:
            m, s = divmod(int(getattr(c, "peak_t", 0.0) or getattr(c, "start", 0.0)), 60)
            c.title = f"{label} — {m}:{s:02d}"


def _audio_title_from_factors(factors: list[ScoreFactor] | list) -> str:
    for f in factors:
        label = (getattr(f, "label", "") or "").lower()
        if "explosive" in label:
            return "Explosive Fight"
        if "cheer" in label or "crowd" in label:
            return "Crowd Pop"
        if "laugh" in label:
            return "Funny Moment"
        if "impact" in label:
            return "Impact Moment"
    return "Audio Moment"


def _overlap(a: "GameplayClip", b: "GameplayClip") -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = a.duration + b.duration - inter
    return inter / union if union > 0 else 0.0


def _merge(priority: list["GameplayClip"], rest: list["GameplayClip"],
           limit: int, *, max_overlap: float = 0.4) -> list["GameplayClip"]:
    kept: list[GameplayClip] = []
    # 1. Take all confirmed gameplay events first (goals, aces, victories, sound cues)
    for c in sorted(priority, key=lambda c: c.score, reverse=True):
        if all(_overlap(c, k) <= max_overlap for k in kept):
            kept.append(c)
        if len(kept) >= limit:
            return kept

    # 2. Fill remaining slots with high-energy audio clips
    # When confirmed in-game events exist, do not flood the user with uncorroborated filler peaks
    max_filler = limit - len(kept)
    if kept:
        max_filler = min(max_filler, 2)

    for c in sorted(rest, key=lambda c: c.score, reverse=True):
        feats = getattr(c, "features", {}) or {}
        has_evidence = (
            feats.get("corroborated")
            or float(feats.get("excitement", 0.0)) >= 0.70
            or float(feats.get("reaction", 0.0)) >= 0.60
            or float(feats.get("audio_event", 0.0)) >= 0.25
        )
        if kept and not has_evidence:
            continue
        if all(_overlap(c, k) <= max_overlap for k in kept):
            kept.append(c)
        if len(kept) >= len(priority) + max_filler or len(kept) >= limit:
            break

    return kept


def _count_spikes(window, thr) -> int:
    import numpy as np

    if len(window) == 0:
        return 0
    above = window > thr
    return int(np.sum((~above[:-1]) & above[1:])) + (1 if above[0] else 0)


def _uniform_fallback(info: MediaInfo, settings: ImportSettings) -> list[GameplayClip]:
    """No usable audio — evenly spaced windows so the user still gets clips."""
    target = (settings.min_len + settings.max_len) / 2.0
    out: list[GameplayClip] = []
    t = 0.0
    i = 1
    while t < info.duration - 2 and len(out) < settings.target_clips:
        end = min(t + target, info.duration)
        m, s = divmod(int(t), 60)
        out.append(GameplayClip(start=round(t, 3), end=round(end, 3), score=50,
                                factors=[ScoreFactor(label="Evenly sampled segment", weight=0.0,
                                                     detail="No audio to rank by")],
                                title=f"Segment {i} — {m}:{s:02d}", peak_t=t))
        t += target
        i += 1
    return out


def compute_climax_curve(times, rms, profile: dict | None = None) -> tuple[list[float], list[float]]:
    """Compute continuous 10 Hz climax intensity curve C(t) in [0, 1]."""
    if times is None or rms is None or len(times) == 0 or len(rms) == 0:
        return [], []
    import numpy as np
    prof = profile or _PROFILES["generic"]
    thr = float(np.percentile(rms, prof.get("thr_pct", 82)))
    rms_max = float(np.max(rms)) if len(rms) else 1.0
    norm_rms = np.clip(rms / max(rms_max, 1e-6), 0.0, 1.0)
    # 1st-derivative surge: rate of loudness increase over 1s (10 hops)
    surge = np.zeros_like(norm_rms)
    hop_1s = 10
    if len(norm_rms) > hop_1s:
        surge[hop_1s:] = np.maximum(0.0, norm_rms[hop_1s:] - norm_rms[:-hop_1s])
    w_t = prof.get("w_transient", 3.0) / 10.0
    w_s = prof.get("w_sustain", 6.0) / 10.0
    total_w = w_t + w_s + 1.0
    curve = (norm_rms + w_t * surge + w_s * (norm_rms > (thr / max(rms_max, 1e-6)))) / total_w
    curve = np.clip(curve, 0.0, 1.0)
    return [round(float(t), 3) for t in times], [round(float(v), 4) for v in curve]

"""Gameplay highlight detection from audio energy.

A game's best moments — a Valorant ace, an EA FC goal, a clutch — don't show up
in the transcript, but they almost always spike the audio: gunfire, ability
sounds, crowd roar, commentary hype, your own reaction. So we find loud,
transient moments and build clips around them, ranked by intensity.

This is game-agnostic and needs no extra models (just ffmpeg + numpy). It's the
foundation; per-game vision detectors (kill-feed / scoreboard OCR) can layer on
top behind this same interface.
"""
from __future__ import annotations

import logging
import wave
from dataclasses import dataclass, field

from ..config import get_settings
from ..media import ffmpeg
from ..media.ffmpeg import MediaInfo
from ..models import DetectedEvent, ImportSettings, ScoreFactor
from . import audio_events as audio_events_mod
from . import detect_cues, detect_ocr

log = logging.getLogger("clipforge.gameplay")

# Map a game profile to its cue-template folder (aliases included).
_CUE_DIR = {"auto": "generic", "cs": "cs2", "fifa": "eafc"}

# Per-game tuning of the audio signals. "generic"/"auto" work for ANY game; the
# named profiles bias how peaks are picked and scored:
#   thr_pct        - loudness percentile to count as "action" (higher = pickier)
#   min_gap_factor - min spacing between highlights (× min_len)
#   lead_frac      - how much of the clip is build-up before the peak
#   w_transient    - reward sudden loud-after-quiet (jump scares, gunfire)
#   w_sustain      - reward sustained loudness (goals/crowd roar, team-fights)
_PROFILES: dict[str, dict] = {
    "generic":      dict(thr_pct=82, min_gap_factor=0.6, lead_frac=0.45, w_transient=3.0,  w_sustain=6.0),
    "valorant":     dict(thr_pct=80, min_gap_factor=0.45, lead_frac=0.50, w_transient=10.0, w_sustain=3.0),
    "cs2":          dict(thr_pct=80, min_gap_factor=0.45, lead_frac=0.50, w_transient=10.0, w_sustain=3.0),
    "eafc":         dict(thr_pct=85, min_gap_factor=0.85, lead_frac=0.55, w_transient=2.0,  w_sustain=13.0),
    "rocketleague": dict(thr_pct=83, min_gap_factor=0.60, lead_frac=0.50, w_transient=6.0,  w_sustain=8.0),
    "horror":       dict(thr_pct=88, min_gap_factor=0.90, lead_frac=0.65, w_transient=16.0, w_sustain=2.0),
}
_PROFILES["auto"] = _PROFILES["generic"]
_PROFILES["cs"] = _PROFILES["cs2"]
_PROFILES["fifa"] = _PROFILES["eafc"]

# The accepted values for ImportSettings.game_profile. The profile also names
# a cue directory on disk, so the API clamps input to this set.
KNOWN_PROFILES = frozenset(_PROFILES)


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


def _timing_window(settings: ImportSettings, profile: dict) -> tuple[float, float]:
    """Seconds before/after the event peak for gameplay clips."""
    target = (settings.min_len + settings.max_len) / 2.0
    default_lead = target * profile["lead_frac"]
    default_tail = target - default_lead
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
                     *, focus_times: list[float] | None = None) -> list:
    if not settings.use_ocr:
        return []
    try:
        return detect_ocr.find_text_events(
            src_path, info, settings, focus_times=focus_times)
    except Exception as e:
        log.warning("ocr detection failed: %s", e)
        return []


def detect_gameplay(src_path: str, info: MediaInfo, settings: ImportSettings,
                    *, weights: dict[str, float] | None = None,
                    wav_path: str | None = None,
                    events_out: list | None = None) -> list[GameplayClip]:
    """Return intensity-ranked highlight clips for gameplay footage.

    ``weights`` lets the caller pass personalised audio-feature weights (from the
    learning loop); otherwise the per-game defaults are used. ``wav_path`` is an
    already-extracted 16 kHz mono wav of the source (the pipeline reuses the
    transcription extract); when absent the audio is extracted here. ``events_out``
    (when given) is extended with the :class:`DetectedEvent`s that were matched
    (audio cues + OCR) so the caller can persist them on the project.
    """
    import tempfile
    from pathlib import Path

    aw = weights or audio_weights(settings.game_profile)

    # On-screen text markers (VICTORY / ELIMINATED / GOAL / kill-feed).
    # For footage with audio we run OCR after finding likely moments, so it reads
    # the frames that matter instead of doing a slow blind sweep over the VOD.
    ocr_events: list = []
    defer_ocr = bool(settings.use_ocr and info.has_audio)
    if settings.use_ocr and not defer_ocr:
        ocr_events = _find_ocr_events(src_path, info, settings)

    if not info.has_audio:
        clips = _uniform_fallback(info, settings)
        ocr_clips = _ocr_clips(ocr_events, info, settings)
        _record_events(events_out, [], ocr_events, [])
        return _merge(ocr_clips, clips, settings.target_clips) if ocr_clips else clips

    cue_events: list = []
    audio_window_events: list = []
    if wav_path is not None:
        times, rms = _load_rms(wav_path)
        cue_events = _cue_events(wav_path, settings)
        audio_window_events = _audio_events(wav_path, info.duration, settings)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            try:
                ffmpeg.extract_audio_wav(src_path, wav)
            except Exception as e:
                log.warning("gameplay audio extract failed: %s", e)
                return _uniform_fallback(info, settings)
            times, rms = _load_rms(str(wav))
            cue_events = _cue_events(str(wav), settings)
            audio_window_events = _audio_events(str(wav), info.duration, settings)

    profile = get_profile(settings.game_profile)
    if defer_ocr:
        focus_times = [float(e.t) for e in cue_events]
        focus_times += [float(e.t) for e in audio_window_events]
        if times is not None and rms is not None:
            try:
                peak_idxs, _thr, _floor = _pick_peaks(
                    times, rms, settings=settings, profile=profile)
                focus_times += [float(times[i]) for i in peak_idxs]
            except Exception as e:
                log.warning("ocr focus peaks failed: %s", e)
        ocr_events = _find_ocr_events(src_path, info, settings,
                                      focus_times=focus_times)

    _record_events(events_out, cue_events, ocr_events, audio_window_events)

    if times is None:
        clips = _uniform_fallback(info, settings)
        ocr_clips = _ocr_clips(ocr_events, info, settings)
        audio_clips = _audio_event_clips(audio_window_events, info, settings)
        priority = ocr_clips + audio_clips
        return _merge(priority, clips, settings.target_clips) if priority else clips


    peaks, thr, floor = _pick_peaks(times, rms, settings=settings, profile=profile)
    if not peaks:
        lead, tail = _timing_window(settings, profile)
        priority = (_cue_clips(cue_events, info, lead, tail, settings)
                    + _ocr_clips(ocr_events, info, settings, lead=lead, tail=tail)
                    + _audio_event_clips(audio_window_events, info, settings,
                                         lead=lead, tail=tail))
        fallback = _uniform_fallback(info, settings)
        return _merge(priority, fallback, settings.target_clips) if priority else fallback

    lead, tail = _timing_window(settings, profile)
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.1
    pre_n = max(int(2.0 / hop), 1)         # ~2s window before the peak

    clips: list[GameplayClip] = []
    for idx in peaks:
        pt = float(times[idx])
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

        feats = {"intensity": round(pct, 4), "sustain": round(sustain, 4),
                 "transient": round(transient, 4),
                 "spikes": round(min(spikes / 5.0, 1.0), 4), "cue": 0.0,
                 "reaction": 0.0}
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
    merged = _merge(cue_clips + ocr_clips + audio_clips, clips, settings.target_clips)
    # Multimodal corroboration: a moment confirmed by BOTH the audio cue and the
    # on-screen text is the surest highlight there is — reward the overlap.
    apply_corroboration(merged, cue_events, ocr_events, audio_window_events)
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
                   ocr_events: list, audio_events: list | None = None) -> None:
    """Append matched cues + OCR hits to ``events_out`` as DetectedEvents."""
    if events_out is None:
        return
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


# On-screen markers that mean the moment already happened — open a little before
# so the play that earned the banner is in-frame, with less tail.
def _ocr_clips(events: list, info: MediaInfo, settings: ImportSettings, *,
               lead: float | None = None, tail: float | None = None) -> list["GameplayClip"]:
    if not events:
        return []
    target = (settings.min_len + settings.max_len) / 2.0
    if lead is None:
        lead = target * 0.7    # banners are the payoff — bias to the build-up
    if tail is None:
        tail = max(target - lead, 2.0)
    out: list[GameplayClip] = []
    for e in events:
        start = max(0.0, e.t - lead)
        end = min(info.duration, e.t + tail)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        if end - start > settings.max_len:
            start = max(0.0, end - settings.max_len)
        score = int(round(max(1, min(99, 68 + e.confidence * 20))))
        m, s = divmod(int(e.t), 60)
        label = e.label.replace("_", " ").title()
        out.append(GameplayClip(
            start=round(start, 3), end=round(end, 3), score=score, peak_t=e.t,
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
        lead = target * 0.55
    if tail is None:
        tail = max(target - lead, 2.0)
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
    """Match reference game sounds from <data>/game_cues/<profile>/ (+ /common)."""
    if not getattr(settings, "use_cues", True):
        return []
    base = get_settings().data_dir / "game_cues"
    events = []
    for sub in {_cue_dir(settings.game_profile), "common"}:
        try:
            events += detect_cues.find_events(wav_path, base / sub)
        except Exception as e:
            log.warning("cue matching failed for %s: %s", sub, e)
    return events


def _audio_events(wav_path: str, duration: float, settings: ImportSettings) -> list:
    if not settings.use_audio_events:
        return []
    try:
        mode = getattr(settings.power_mode, "value", str(settings.power_mode))
        hop = 4.0 if mode in ("max_gpu", "quality") else 5.0
        window = 5.0 if mode == "max_gpu" else 6.0
        threshold = 0.55 if mode == "quality" else 0.58
        return audio_events_mod.find_events(
            wav_path, duration, window=window, hop=hop, threshold=threshold,
            limit=max(settings.target_clips, 6), profile=settings.game_profile,
            language=settings.language)
    except Exception as e:
        log.warning("CLAP audio event search failed: %s", e)
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
        cue_strength = max(0.0, sim - 0.84)
        score = int(round(max(1, min(88, 58 + cue_strength * 120 + (n - 1) * 4))))
        m, s = divmod(int(first.t), 60)
        label = first.label.replace("_", " ").title()
        labels = {e.label for e in grp}
        if n == 1:
            title = f"{label} — {m}:{s:02d}"
        elif len(labels) == 1:
            word = _STREAK_WORDS.get(n, f"{n}x")
            title = f"{word} {label} — {m}:{s:02d}"
        else:
            title = f"{label} + {n - 1} cues — {m}:{s:02d}"

        factors = [ScoreFactor(label=f"Matched '{first.label}' sound",
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
        exceptional = (
            corroborated
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


def apply_title_fallbacks(clips: list["GameplayClip"] | list) -> None:
    """Replace generic peak titles with the strongest visible/audio reason."""
    for c in clips:
        title = getattr(c, "title", "") or ""
        if not title.lower().startswith("highlight"):
            continue
        feats = getattr(c, "features", {}) or {}
        label = "Highlight"
        if float(feats.get("reaction", 0.0)) > 0.55:
            label = "Big Reaction"
        elif float(feats.get("transient", 0.0)) > 0.65 or float(feats.get("spikes", 0.0)) > 0.7:
            label = "Sudden Fight"
        elif float(feats.get("intensity", 0.0)) > 0.85:
            label = "Loud Fight"
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
    for c in sorted(priority, key=lambda c: c.score, reverse=True) + \
             sorted(rest, key=lambda c: c.score, reverse=True):
        if all(_overlap(c, k) <= max_overlap for k in kept):
            kept.append(c)
        if len(kept) >= limit:
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

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
from ..models import ImportSettings, ScoreFactor
from . import detect_cues

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
# audio weights are scaled down to make room — see orchestrator).
REACTION_WEIGHT = 0.15


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


def detect_gameplay(src_path: str, info: MediaInfo, settings: ImportSettings,
                    *, weights: dict[str, float] | None = None,
                    wav_path: str | None = None) -> list[GameplayClip]:
    """Return intensity-ranked highlight clips for gameplay footage.

    ``weights`` lets the caller pass personalised audio-feature weights (from the
    learning loop); otherwise the per-game defaults are used. ``wav_path`` is an
    already-extracted 16 kHz mono wav of the source (the pipeline reuses the
    transcription extract); when absent the audio is extracted here.
    """
    import tempfile
    from pathlib import Path

    aw = weights or audio_weights(settings.game_profile)

    if not info.has_audio:
        return _uniform_fallback(info, settings)

    cue_events: list = []
    if wav_path is not None:
        times, rms = _load_rms(wav_path)
        cue_events = _cue_events(wav_path, settings)
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

    if times is None:
        return _uniform_fallback(info, settings)


    profile = get_profile(settings.game_profile)
    peaks, thr, floor = _pick_peaks(times, rms, settings=settings, profile=profile)
    if not peaks:
        return _uniform_fallback(info, settings)

    target = (settings.min_len + settings.max_len) / 2.0
    lead = target * profile["lead_frac"]   # show the build-up, then the payoff
    tail = target - lead
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.1
    pre_n = max(int(2.0 / hop), 1)         # ~2s window before the peak

    clips: list[GameplayClip] = []
    for idx in peaks:
        pt = float(times[idx])
        start = max(0.0, pt - lead)
        end = min(info.duration, pt + tail)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        # intensity 0..1 = how this peak ranks across the whole envelope
        pct = float((rms < rms[idx]).mean())
        # sustain: share of the window above the action threshold
        w0 = int(start / hop); w1 = int(end / hop)
        window = rms[w0:w1] if w1 > w0 else rms[idx:idx + 1]
        sustain = float((window > thr).mean()) if len(window) else 0.0
        spikes = _count_spikes(window, thr)
        # transient: how much the peak jumps above the ~2s just before it
        pre = rms[max(idx - pre_n, 0):max(idx - 2, 1)]
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

    # Cue-matched clips (exact game sounds) take priority over loudness clips.
    cue_clips = _cue_clips(cue_events, info, lead, tail, settings)
    merged = _merge(cue_clips, clips, settings.target_clips)
    merged.sort(key=lambda c: c.start)
    return merged


def _cue_dir(profile_name: str) -> str:
    name = (profile_name or "generic").lower().replace(" ", "")
    return _CUE_DIR.get(name, name)


def _cue_events(wav_path: str, settings: ImportSettings) -> list:
    """Match reference game sounds from <data>/game_cues/<profile>/ (+ /common)."""
    base = get_settings().data_dir / "game_cues"
    events = []
    for sub in {_cue_dir(settings.game_profile), "common"}:
        try:
            events += detect_cues.find_events(wav_path, base / sub)
        except Exception as e:
            log.warning("cue matching failed for %s: %s", sub, e)
    return events


def _cue_clips(events: list, info: MediaInfo, lead: float, tail: float,
               settings: ImportSettings) -> list["GameplayClip"]:
    out: list[GameplayClip] = []
    for ev in events:
        start = max(0.0, ev.t - lead)
        end = min(info.duration, ev.t + tail)
        if end - start < settings.min_len:
            start = max(0.0, end - settings.min_len)
        score = int(round(max(1, min(99, 75 + ev.similarity * 20))))
        m, s = divmod(int(ev.t), 60)
        out.append(GameplayClip(
            start=round(start, 3), end=round(end, 3), score=score, peak_t=ev.t,
            title=f"{ev.label.replace('_', ' ').title()} — {m}:{s:02d}",
            features={"cue": round(ev.similarity, 4), "intensity": 0.0,
                      "sustain": 0.0, "transient": 0.0, "spikes": 0.0},
            factors=[ScoreFactor(label=f"Matched '{ev.label}' sound",
                                 weight=round(ev.similarity * 20, 1),
                                 detail=f"{int(ev.similarity*100)}% audio match to your reference cue")]))
    return out


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

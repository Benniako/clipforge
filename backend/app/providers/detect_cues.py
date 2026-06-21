"""Audio-cue detection by template matching — find exact game events locally.

Instead of needing your gameplay footage or a trained vision model, this matches
a short *reference sound* (Valorant kill "ding", EA FC goal/whistle, Rocket League
goal explosion, a headshot crunch…) against the video's audio and returns the
timestamps where it occurs. Drop reference ``.wav``/``.mp3`` files into
``<data>/game_cues/<profile>/`` (grab them from soundboards like MyInstants, an
SFX pack, or extract from the game with FModel) and the matcher does the rest.

Method: log-band magnitude spectrograms + normalized cross-correlation (cosine
similarity of the template against every position in the audio). Pure NumPy + the
ffmpeg we already bundle — no API, no extra ML deps.
"""
from __future__ import annotations

import logging
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from ..media import ffmpeg

log = logging.getLogger("clipforge.cues")

SR = 16000
FRAME = 400          # 25 ms
HOP = 160            # 10 ms
BANDS = 40
CUE_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}


@dataclass
class CueEvent:
    t: float          # timestamp in the source (s)
    label: str        # cue name (file stem, e.g. "kill", "goal")
    similarity: float  # 0..1 match confidence


def _load_16k(path: str | Path):
    """Decode any audio to a mono 16 kHz float32 array via ffmpeg."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        ffmpeg.extract_audio_wav(path, wav, sample_rate=SR)
        with wave.open(str(wav), "rb") as wf:
            ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
    if not raw:
        return np.zeros(0, dtype=np.float32)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data / 32768.0


def _band_spectrogram(samples):
    """[n_frames, BANDS] log-band magnitudes, each frame L2-normalized."""
    import numpy as np

    if len(samples) < FRAME:
        return np.zeros((0, BANDS), dtype=np.float32)
    n = 1 + (len(samples) - FRAME) // HOP
    window = np.hanning(FRAME).astype(np.float32)
    # frame the signal (stride view) and FFT
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = samples[idx] * window
    mag = np.abs(np.fft.rfft(frames, axis=1))           # [n, FRAME/2+1]
    # group FFT bins into BANDS, take log
    edges = np.linspace(0, mag.shape[1], BANDS + 1).astype(int)
    bands = np.empty((n, BANDS), dtype=np.float32)
    for b in range(BANDS):
        lo, hi = edges[b], max(edges[b + 1], edges[b] + 1)
        bands[:, b] = mag[:, lo:hi].sum(axis=1)
    bands = np.log1p(bands)
    norm = np.linalg.norm(bands, axis=1, keepdims=True) + 1e-6
    return bands / norm


def _template_quality(samples) -> tuple[bool, str]:
    """Reject cue files that are too weak/flat to be reliable templates."""
    import numpy as np

    duration = len(samples) / SR
    if duration < 0.25:
        return False, "too short"
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
    if peak < 0.025 or rms < 0.004:
        return False, "too quiet"
    # Very flat background loops match everywhere; event cues have transients.
    active = float(np.mean(np.abs(samples) > max(peak * 0.20, 0.015)))
    if active > 0.92:
        return False, "too constant"
    return True, ""


def match_template(sig, tmpl, *, threshold: float, min_gap: float):
    """Return [(t, similarity)] where template ``tmpl`` matches signal ``sig``."""
    import numpy as np

    sig_spec = _band_spectrogram(sig)
    tmpl_spec = _band_spectrogram(tmpl)
    tn = len(tmpl_spec)
    if tn < 2 or len(sig_spec) <= tn:
        return []
    # similarity[k] = mean_i <sig_spec[k+i], tmpl_spec[i]>  (both L2-normalized)
    sim = np.zeros(len(sig_spec) - tn + 1, dtype=np.float32)
    for b in range(BANDS):
        sim += np.correlate(sig_spec[:, b], tmpl_spec[:, b], mode="valid")
    sim /= tn

    gap = max(int(min_gap * SR / HOP), 1)
    out: list[tuple[float, float]] = []
    order = np.argsort(sim)[::-1]
    taken: list[int] = []
    for k in order:
        if sim[k] < threshold:
            break
        if all(abs(int(k) - j) >= gap for j in taken):
            taken.append(int(k))
            out.append((int(k) * HOP / SR, float(sim[k])))
    return out


# Process long audio in ~2-minute chunks (overlapped by the template length) so
# spectrogram memory stays bounded regardless of VOD length.
CHUNK_SAMPLES = 120 * SR


def find_events(audio_path: str, cues_dir: Path, *,
                threshold: float = 0.84, min_gap: float = 6.0,
                max_events_per_template: int = 24) -> list[CueEvent]:
    """Match every reference cue in ``cues_dir`` against the audio."""
    if not cues_dir.is_dir():
        return []
    templates = [p for p in sorted(cues_dir.iterdir()) if p.suffix.lower() in CUE_EXTS]
    return match_templates(audio_path, templates, threshold=threshold,
                           min_gap=min_gap,
                           max_events_per_template=max_events_per_template)


def match_templates(audio_path: str, templates: list[Path], *,
                    threshold: float = 0.84, min_gap: float = 6.0,
                    max_events_per_template: int = 24) -> list[CueEvent]:
    """Match an explicit list of reference cue files against the audio."""
    templates = [p for p in templates if p.suffix.lower() in CUE_EXTS and p.is_file()]
    if not templates:
        return []
    try:
        sig = _load_16k(audio_path)
    except Exception as e:
        log.warning("cue: could not load audio: %s", e)
        return []
    if len(sig) == 0:
        return []

    events: list[CueEvent] = []
    for tpath in templates:
        try:
            tmpl = _load_16k(tpath)
            ok, reason = _template_quality(tmpl)
            if not ok:
                log.warning("cue: skipping %s (%s)", tpath.name, reason)
                continue
            hits: list[tuple[float, float]] = []
            overlap = len(tmpl) + FRAME
            start = 0
            while start < len(sig):
                chunk = sig[start: start + CHUNK_SAMPLES + overlap]
                off = start / SR
                hits += [(t + off, s) for t, s in
                         match_template(chunk, tmpl, threshold=threshold, min_gap=min_gap)]
                start += CHUNK_SAMPLES
            # De-dupe matches that landed in two overlapping chunks.
            hits.sort(key=lambda h: h[1], reverse=True)
            kept: list[tuple[float, float]] = []
            for t, s in hits:
                if all(abs(t - k) >= min_gap for k, _ in kept):
                    kept.append((t, s))
            if max_events_per_template > 0 and len(kept) > max_events_per_template:
                log.warning(
                    "cue: %s produced %d strict matches; keeping strongest %d",
                    tpath.name, len(kept), max_events_per_template)
                kept = kept[:max_events_per_template]
            events += [CueEvent(t=round(t, 3), label=tpath.stem, similarity=round(s, 3))
                       for t, s in kept]
        except Exception as e:
            log.warning("cue: template %s failed: %s", tpath.name, e)
    events.sort(key=lambda e: e.t)
    log.info("cue: matched %d events from %d templates", len(events), len(templates))
    return events

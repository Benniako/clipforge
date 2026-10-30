"""Audio processing utilities — zero-crossing boundary snapping and click-free windowing.

Provides sub-millisecond zero-crossing detection and normalized raised-cosine
(Tukey-equivalent) audio fades to eliminate audible pops and DC bias transients
at clip boundaries.
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

log = logging.getLogger("clipforge.audio")


def find_zero_crossing(samples: np.ndarray, sample_rate: int, target_t: float,
                       *, max_offset_s: float = 0.015) -> float:
    """Find the timestamp of the nearest zero-crossing in a 1D audio waveform.

    Searches within [target_t - max_offset_s, target_t + max_offset_s] for sample
    transitions where samples[i] * samples[i+1] <= 0 with minimal absolute amplitude.
    Returns target_t unchanged if no zero-crossing exists or samples are empty.
    """
    if samples is None or len(samples) < 2 or sample_rate <= 0:
        return float(target_t)

    target_idx = int(round(target_t * sample_rate))
    radius = max(int(round(max_offset_s * sample_rate)), 1)
    start_idx = max(0, target_idx - radius)
    end_idx = min(len(samples) - 1, target_idx + radius)

    if end_idx <= start_idx:
        return float(target_t)

    window = samples[start_idx:end_idx + 1]
    # Check sign changes
    sign_changes = (window[:-1] * window[1:]) <= 0
    indices = np.where(sign_changes)[0]

    if len(indices) == 0:
        return float(target_t)

    # Global sample indices for crossings
    crossing_global_indices = start_idx + indices
    # Pick the crossing closest to target_idx
    best_idx = min(crossing_global_indices, key=lambda idx: abs(idx - target_idx))
    return round(float(best_idx / sample_rate), 4)


def find_zero_crossing_in_file(wav_path: str | Path, target_t: float,
                               *, max_offset_s: float = 0.015) -> float:
    """Read a small audio slice around target_t and locate the nearest zero-crossing.

    Uses soundfile seeking when available for sub-millisecond speed, falling back
    to target_t unchanged on any error or missing file.
    """
    if not wav_path or target_t < 0:
        return float(target_t)

    path = Path(wav_path)
    if not path.is_file():
        return float(target_t)

    try:
        import soundfile as sf
        with sf.SoundFile(str(path)) as sf_in:
            sr = sf_in.samplerate
            total = len(sf_in)
            target_idx = int(round(target_t * sr))
            radius = max(int(round(max_offset_s * sr)), 1)
            start_frame = max(0, target_idx - radius)
            end_frame = min(total, target_idx + radius)
            count = end_frame - start_frame
            if count <= 2:
                return float(target_t)

            sf_in.seek(start_frame)
            data = sf_in.read(count, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            offset = find_zero_crossing(data, sr, (target_idx - start_frame) / sr,
                                        max_offset_s=max_offset_s)
            return round(start_frame / sr + offset, 4)
    except Exception as e:
        log.debug("zero crossing search in %s skipped: %s", path.name, e)
        return float(target_t)


def audio_fade_filter(duration: float, *, fade_s: float = 0.007) -> str:
    """Return an FFmpeg afade filter string with 7ms half-sine (raised-cosine) curves.

    Applies raised-cosine fade-in at t=0 and fade-out at t=duration-fade_d,
    preventing high-frequency step discontinuity pops at render boundaries.
    """
    fade_d = min(max(fade_s, 0.001), max(duration * 0.05, 0.001))
    out_st = max(duration - fade_d, 0.0)
    return (f"afade=t=in:ss=0:d={fade_d:.3f}:curve=hsin,"
            f"afade=t=out:st={out_st:.3f}:d={fade_d:.3f}:curve=hsin")

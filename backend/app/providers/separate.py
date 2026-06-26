"""Vocal isolation via Demucs — make the speech sound studio-clean.

Long-form footage often carries background music, game audio, or room echo that
muddies the speech and bleeds into the burned-in captions' "sound". Demucs (Meta
Research) is the strongest open-source music source-separation model; its
``vocals`` stem is an excellent voice isolator. With Demucs installed and the
project's *Clean voice* toggle on, we separate the source's vocals once, remux
them back onto the video (video stream copied, untouched), and render from that —
so every clip inherits clean speech with no per-clip cost.

Optional and graceful: no Demucs ⇒ :func:`denoise_source` returns None and the
pipeline renders from the original source exactly as before. Heavy: a separation
pass runs a neural net over the whole audio track (GPU strongly recommended),
which is why it's an explicit opt-in rather than always-on.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg

log = logging.getLogger("clipforge.separate")


def available() -> bool:
    """Demucs vocal isolation has been removed — always returns False."""
    return False


def denoise_source(src_path: str, dst_path: str) -> str | None:
    """Write a copy of ``src_path`` with the voice isolated, to ``dst_path``.

    Runs Demucs over the source audio, takes the ``vocals`` stem, and remuxes it
    onto the original video (copied, not re-encoded). Returns ``dst_path`` on
    success or None on any failure / when Demucs is unavailable, so the caller
    falls back to the original source.
    """
    if not available():
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpd = Path(tmp)
            audio = tmpd / "audio.wav"
            # Demucs wants 44.1 kHz stereo; keep it lossless going in.
            ffmpeg.run(["-i", src_path, "-vn", "-ac", "2", "-ar", "44100",
                        "-c:a", "pcm_s16le", str(audio)], timeout=600)
            vocals = _run_demucs(audio, tmpd)
            if vocals is None or not vocals.exists():
                return None
            # Remux: original video (copied) + isolated-vocals audio. Loudness is
            # normalised at render time, so we don't double-normalise here.
            ffmpeg.run(["-i", src_path, "-i", str(vocals),
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", str(dst_path)], timeout=900)
        log.info("denoised source written to %s", dst_path)
        return dst_path
    except Exception as e:
        log.warning("vocal isolation failed (%s); using original audio", e)
        return None


def _run_demucs(audio: Path, out_root: Path) -> Path | None:
    """Run Demucs separation and return the path to the vocals stem, or None."""
    import sys

    from .._util import run_subprocess

    device = "cuda" if get_settings().device == "cuda" else "cpu"
    # --two-stems=vocals only computes vocals vs the rest — ~2x faster than the
    # full 4-stem split and all we need. Invoke via the module so we use the same
    # interpreter/venv Demucs is installed in.
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-d", device, "-o", str(out_root), str(audio)]
    run_subprocess(cmd, timeout=3600, log_label="demucs")
    # Demucs writes <out_root>/<model>/<track>/vocals.wav
    hits = list(out_root.glob("*/*/vocals.wav"))
    return hits[0] if hits else None

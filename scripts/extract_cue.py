#!/usr/bin/env python3
"""Extract a short reference audio cue from a video for ClipForge cue matching.

Usage:
    python scripts/extract_cue.py <video> <timestamp_seconds> <out.wav> [duration]

Example (a kill happens at 1:12.5 in your clip):
    python scripts/extract_cue.py clip.mp4 72.5 backend/data/game_cues/valorant/kill.wav

Saves a mono 16 kHz wav centred slightly before the timestamp — drop it in the
matching game_cues/<profile>/ folder and ClipForge will find every occurrence.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the backend package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.media import ffmpeg  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    video, ts, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
    dur = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(ts - dur * 0.3, 0.0)  # a little lead so the onset is included
    ffmpeg.run(["-ss", f"{start:.3f}", "-i", str(video), "-t", f"{dur:.3f}",
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)])
    print(f"saved cue -> {out}  ({dur:.1f}s @16kHz mono)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

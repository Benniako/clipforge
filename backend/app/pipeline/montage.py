"""Montage building — stitch several clips into one video and score it.

The clips in a project all share the same canvas (aspect/fps/codec), so they
concatenate cleanly. We re-encode the concatenation (robust against tiny per-file
timestamp differences) and give the montage its own virality score derived from
its members — weighted toward the opening, since the first few seconds decide
whether a viewer stays.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg
from ..models import Clip, ScoreFactor


def score_montage(clips: list[Clip]) -> tuple[int, list[ScoreFactor]]:
    """Virality score for an ordered montage of clips."""
    scores = [c.score for c in clips] or [0]
    first = scores[0]
    avg = sum(scores) / len(scores)
    # The hook (first clip) carries extra weight; the rest is the average.
    raw = 0.45 * first + 0.55 * avg
    score = int(round(max(1, min(99, raw))))

    factors: list[ScoreFactor] = []
    if first >= 75:
        factors.append(ScoreFactor(label="Strong opening hook", weight=round(first * 0.45, 1),
                                   detail=f"Opens on a {first}/100 clip"))
    elif first < 55:
        factors.append(ScoreFactor(label="Weak opener — reorder for a stronger start",
                                   weight=0.0, detail=f"First clip is only {first}/100"))
    if min(scores) >= 60:
        factors.append(ScoreFactor(label="Consistently strong clips", weight=round(avg * 0.2, 1),
                                   detail=f"Every clip scores {min(scores)}+"))
    factors.append(ScoreFactor(label=f"{len(clips)} clips combined", weight=0.0,
                               detail="Back-to-back highlights keep momentum"))
    return score, factors


def build_montage_video(clip_paths: list[Path], out_path: Path,
                        thumb_path: Path) -> float:
    """Concatenate clip files into ``out_path``; return the montage duration."""
    paths = [p for p in clip_paths if p.exists()]
    if not paths:
        raise RuntimeError("no rendered clips available for this montage")

    s = get_settings()
    out_abs = str(out_path.resolve())
    with tempfile.TemporaryDirectory() as tmp:
        # concat demuxer list — forward slashes work on every OS and avoid the
        # Windows backslash-escaping pitfalls. A single quote in the path (e.g.
        # C:/Users/O'Brien) must be escaped as '\'' or the list fails to parse.
        listing = "\n".join(
            "file '{}'".format(p.resolve().as_posix().replace("'", "'\\''"))
            for p in paths)
        (Path(tmp) / "list.txt").write_text(listing, encoding="utf-8")

        encoders = [s.video_encoder_args()]
        if s.use_nvenc:
            encoders.append(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-profile:v", "high"])
        last: Exception | None = None
        for enc in encoders:
            try:
                ffmpeg.run(["-f", "concat", "-safe", "0", "-i", "list.txt",
                            *enc, "-c:a", "aac", "-b:a", "128k",
                            "-movflags", "+faststart", out_abs],
                           timeout=1800, cwd=tmp)
                last = None
                break
            except ffmpeg.FFmpegError as e:
                last = e
        if last is not None:
            raise last

    info = ffmpeg.probe(out_path)
    ffmpeg.make_thumbnail(out_path, thumb_path, at=min(info.duration * 0.1, 2.0), width=540)
    return info.duration

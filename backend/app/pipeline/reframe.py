"""Vertical reframing (16:9 → 9:16) with speaker-aware cropping.

We sample frames across the clip, detect faces (OpenCV Haar cascade), and follow
the dominant face to build a horizontal crop path. The path is then *smoothed*
and velocity-limited so the camera glides instead of jittering — the PRD calls
out that jittery or constantly-panning crops read as cheap.

If OpenCV isn't present, or no faces are found (screen-share, graphics, empty
stage), we fall back to a steady centre crop. Either way the output is a list of
``ReframeKeyframe`` (time, centre-x fraction) the renderer turns into an ffmpeg
crop expression.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg
from ..models import LayoutType, Reframe, ReframeKeyframe

log = logging.getLogger("clipforge.reframe")

SAMPLE_FPS = 3.0          # frames/sec sampled for tracking
SAMPLE_WIDTH = 480        # downscale for fast detection (cx is a fraction, so OK)
EMA_ALPHA = 0.22          # smoothing strength
MAX_STEP = 0.05           # max centre move per sample (conservative pan)


def compute_reframe(src: str, start: float, end: float, src_aspect: float,
                    *, speech: list[tuple[float, float]] | None = None) -> Reframe:
    """Return a smoothed 9:16 crop path for the clip [start, end].

    ``speech`` is an optional list of clip-relative (start, end) intervals
    where someone is actually talking. During silence the crop holds its last
    position instead of re-aiming — chasing a nodding listener between
    sentences is the classic multi-cam reframe mistake.
    """
    # A source already at/under 9:16 needs no horizontal tracking.
    if src_aspect and src_aspect <= 9 / 16 + 1e-3:
        return Reframe(layout=LayoutType.center,
                       keyframes=[ReframeKeyframe(t=0.0, cx=0.5)], tracked=False)

    s = get_settings()
    centers = _track_faces(src, start, end, speech) if s.has_opencv else None
    if not centers:
        return Reframe(layout=LayoutType.center,
                       keyframes=[ReframeKeyframe(t=0.0, cx=0.5)], tracked=False)

    smoothed = _smooth(centers)
    keyframes = _to_keyframes(smoothed, start)
    return Reframe(layout=LayoutType.fill, keyframes=keyframes, tracked=True)


def _speech_active(t: float, intervals: list[tuple[float, float]] | None) -> bool:
    """True when ``t`` falls inside a speech interval (or none are known)."""
    if not intervals:
        return True
    return any(a <= t <= b for a, b in intervals)


def _track_faces(src: str, start: float, end: float,
                 speech: list[tuple[float, float]] | None = None
                 ) -> list[tuple[float, float]] | None:
    """Sample frames and return [(t_rel, cx_fraction)], cx=None carried as gaps."""
    try:
        import cv2
        import numpy as np  # noqa: F401
    except Exception:
        return None

    dur = max(end - start, 0.1)
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "f_%05d.jpg")
        try:
            ffmpeg.run([
                "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
                "-vf", f"fps={SAMPLE_FPS},scale={SAMPLE_WIDTH}:-2",
                "-q:v", "4", pattern,
            ])
        except Exception as e:
            log.warning("frame sampling failed: %s", e)
            return None

        frames = sorted(Path(tmp).glob("f_*.jpg"))
        if not frames:
            return None

        from ..media import faces as faces_mod

        centers: list[tuple[float, float]] = []
        last_cx: float | None = None
        prev_gray = None
        hits = 0
        for i, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faces_mod.detect_faces(img, min_size_frac=0.06)
            t_rel = i / SAMPLE_FPS
            if len(faces):
                hits += 1
                # Re-aim only while someone is talking (or when we have no
                # position yet); silence holds the current frame steady.
                if last_cx is None or _speech_active(t_rel, speech):
                    last_cx = _pick_face(faces, gray, prev_gray, w)
                cx = last_cx
            else:
                cx = last_cx  # hold last known position; may be None
            centers.append((t_rel, cx if cx is not None else 0.5))
            prev_gray = gray

        # If faces were essentially never found, this isn't a tracked scene.
        if hits < max(2, len(frames) * 0.15):
            return None
        return centers


def _pick_face(faces, gray, prev_gray, frame_w: int) -> float:
    """Centre-x (fraction) of the face most likely to be SPEAKING.

    With one face it's trivial. With several, the loudest visual cue for "who is
    talking" is mouth-region motion between frames — so each face is scored by
    the mean abs-difference in the lower half of its box (where the mouth is),
    with face size as a soft tiebreaker. Falls back to largest-face when there's
    no previous frame to diff against.
    """
    import cv2  # noqa: F401  (already imported by caller; keeps this testable alone)

    def center(f):
        return (f[0] + f[2] / 2) / frame_w

    if len(faces) == 1 or prev_gray is None or prev_gray.shape != gray.shape:
        fx, fy, fw, fh = max(
            faces, key=lambda f: f[2] * f[3] - abs((f[0] + f[2] / 2) / frame_w - 0.5) * f[2])
        return (fx + fw / 2) / frame_w

    best, best_score = faces[0], -1.0
    area_max = max(int(f[2]) * int(f[3]) for f in faces) or 1
    for f in faces:
        x, y, fw, fh = (int(v) for v in f)
        my0 = y + fh // 2                      # lower half ≈ mouth region
        cur = gray[my0:y + fh, x:x + fw]
        prv = prev_gray[my0:y + fh, x:x + fw]
        if cur.size == 0 or cur.shape != prv.shape:
            motion = 0.0
        else:
            diff = cv2.absdiff(cur, prv)
            motion = float(diff.mean()) / 255.0
        score = motion + 0.02 * (fw * fh / area_max)   # motion dominates
        if score > best_score:
            best, best_score = f, score
    return center(best)


def _smooth(centers: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """EMA + per-step velocity clamp for smooth, conservative panning."""
    out: list[tuple[float, float]] = []
    ema: float | None = None
    for t, cx in centers:
        if ema is None:
            ema = cx
        else:
            target = EMA_ALPHA * cx + (1 - EMA_ALPHA) * ema
            step = max(-MAX_STEP, min(MAX_STEP, target - ema))
            ema = ema + step
        out.append((t, max(0.0, min(1.0, ema))))
    return out


def _to_keyframes(series: list[tuple[float, float]], start: float,
                  *, epsilon: float = 0.012) -> list[ReframeKeyframe]:
    """Douglas-Peucker simplify so the crop expression stays compact."""
    if not series:
        return [ReframeKeyframe(t=0.0, cx=0.5)]
    pts = _rdp(series, epsilon)
    return [ReframeKeyframe(t=round(t, 3), cx=round(cx, 4)) for t, cx in pts]


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    # perpendicular distance of each point to the line (first,last) in cx-space
    (t0, v0), (t1, v1) = points[0], points[-1]
    dmax, idx = 0.0, 0
    dt = (t1 - t0) or 1e-6
    slope = (v1 - v0) / dt
    for i in range(1, len(points) - 1):
        t, v = points[i]
        proj = v0 + slope * (t - t0)
        d = abs(v - proj)
        if d > dmax:
            dmax, idx = d, i
    if dmax <= epsilon:
        return [points[0], points[-1]]
    left = _rdp(points[: idx + 1], epsilon)
    right = _rdp(points[idx:], epsilon)
    return left[:-1] + right

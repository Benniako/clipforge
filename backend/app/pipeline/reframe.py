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
# One-Euro filter tuning. min_cutoff = heavy smoothing when the face is roughly
# still (kills hand-held jitter); beta = how aggressively smoothing relaxes once
# the face genuinely moves (low latency during a pan). The classic 1€ defaults
# (Casiez et al. 2012) — better than a fixed EMA because the speed/smoothing
# tradeoff is adaptive: still → smooth, moving → responsive.
ONE_EURO_MIN_CUTOFF = 1.2
ONE_EURO_BETA = 0.6
ONE_EURO_D_CUTOFF = 1.0
MAX_STEP = 0.05           # max centre move per sample (final safety clamp)


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a 1€ low-pass given cutoff freq and frame dt."""
    te = 1.0 / (2 * 3.141592653589793 * cutoff)
    return 1.0 / (1.0 + te / max(dt, 1e-6))


def one_euro_filter(samples: list[tuple[float, float]], *,
                    min_cutoff: float = ONE_EURO_MIN_CUTOFF,
                    beta: float = ONE_EURO_BETA,
                    d_cutoff: float = ONE_EURO_D_CUTOFF
                    ) -> list[tuple[float, float]]:
    """One-Euro adaptive low-pass over a (t, value) series.

    At low speeds the cutoff collapses → heavy smoothing (no jitter). As the
    derivative grows, beta lifts the cutoff → low lag during real pans. Strictly
    better for face tracking than a fixed-coefficient EMA, which forces one
    static speed/latency compromise. Pure, so unit-tested without OpenCV.
    """
    out: list[tuple[float, float]] = []
    prev_t: float | None = None
    prev_v: float | None = None
    d_prev = 0.0
    for t, v in samples:
        if prev_t is None or prev_v is None:
            out.append((t, v))
            prev_t, prev_v = t, v
            continue
        dt = max(t - prev_t, 1e-6)
        # Estimate the signal speed (derivative), itself low-passed so noise in
        # the derivative doesn't inject noise into the cutoff.
        a_d = _alpha(d_cutoff, dt)
        d = (v - prev_v) / dt
        d_hat = a_d * d + (1 - a_d) * d_prev
        # Speed-adaptive cutoff: barely smoothing when moving fast.
        cutoff = min_cutoff + beta * abs(d_hat)
        a = _alpha(cutoff, dt)
        v_hat = a * v + (1 - a) * prev_v
        out.append((t, v_hat))
        prev_t, prev_v = t, v_hat
        d_prev = d_hat
    return out


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
    centers = None
    if s.has_asd:
        try:
            from ..providers import active_speaker

            centers = active_speaker.track_centers(src, start, end)
        except Exception as e:
            log.warning("active-speaker reframe failed: %s", e)
    if not centers and s.has_opencv:
        centers = _track_faces(src, start, end, speech)
    if not centers:
        return Reframe(layout=LayoutType.center,
                       keyframes=[ReframeKeyframe(t=0.0, cx=0.5)], tracked=False)

    smoothed = _smooth(centers, src_aspect)
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

        # Content-aware fallback (YOLO/MediaPipe): when no face is visible, follow
        # the dominant subject (a turned-away player, a car, a pet) instead of
        # freezing — only when that backend is installed.
        use_subject = get_settings().reframe_engine == "yolo"
        subject_center = None
        if use_subject:
            from ..providers.subject import subject_center  # noqa: F811

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
                sc = subject_center(img) if use_subject else None
                if sc is not None:
                    hits += 1
                    if last_cx is None or _speech_active(t_rel, speech):
                        last_cx = sc
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

    area_max = max(int(f[2]) * int(f[3]) for f in faces) or 1
    # Score every face by mouth-region motion first. Area is only a *tiebreaker*
    # — a large still face must not out-rank a small talking one. So we collect
    # (motion, area_term, face) and let motion dominate, breaking near-ties
    # (within AREA_TIE_EPS) by size.
    AREA_TIE_EPS = 0.02
    scored: list[tuple[float, float, tuple]] = []
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
        area_term = 0.02 * (fw * fh / area_max)   # motion dominates
        scored.append((motion, area_term, f))
    motion_max = max((m for m, _, _ in scored), default=0.0)
    best = max(scored, key=lambda s: s[0])[2]
    # Only when the top faces are essentially still (motion within ε of the
    # winner) do we let size pick — that's the documented "largest face when
    # nobody is clearly talking" fallback, not a default preference for big faces.
    tied = [s for s in scored if motion_max - s[0] <= AREA_TIE_EPS]
    if len(tied) > 1:
        best = max(tied, key=lambda s: s[1])[2]
    return center(best)


def _crop_half(src_aspect: float, out_aspect: float = 9 / 16) -> float:
    """Half the width (as a fraction of the source width) a vertical crop window
    occupies. Used to clamp the centre so a keyframe can't push the window past
    the frame edge (which would render black bars)."""
    if src_aspect <= 0:
        return 0.5
    # Vertical crop width fraction = out_aspect / src_aspect (≤ 1 for landscape → portrait).
    frac = min(1.0, out_aspect / src_aspect)
    return frac / 2.0


def _smooth(centers: list[tuple[float, float]], src_aspect: float = 16 / 9
            ) -> list[tuple[float, float]]:
    """One-Euro adaptive smoothing + per-step velocity clamp + edge-safe clamp.

    The 1€ filter adapts smoothing to face speed (still→smooth, moving→responsive);
    MAX_STEP is kept as a final conservative pan limit; the centre is then clamped
    to the real crop-safe range so an extreme keyframe can't describe a window
    that hangs off the frame edge (the previous [0,1] clamp allowed that).
    """
    if not centers:
        return centers
    lo = _crop_half(src_aspect)
    hi = 1.0 - lo
    out: list[tuple[float, float]] = []
    prev = None
    for t, cx in one_euro_filter(centers):
        # Final velocity clamp — even 1€ can overshoot on a single huge jump.
        if prev is not None:
            step = max(-MAX_STEP, min(MAX_STEP, cx - prev))
            cx = prev + step
        # Edge-safe clamp: centre must keep the whole crop window in-frame.
        cx = max(lo, min(hi, cx))
        out.append((t, cx))
        prev = cx
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

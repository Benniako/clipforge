"""Ball-following crop for football games (EA FC).

A 9:16 crop of a 16:9 frame keeps about a third of its width. EA FC's camera
leads the play, so the ball often sits outside a fixed centre crop. This module
finds the ball and turns its track into the same centre-x keyframes the
speaker tracker produces, so the renderer pans to follow it.

* **Detection**: YOLO's COCO "sports ball" class at a 1920-px input, so an
  8-px ball (1440p, zoomed-out camera) is still seen: over a 16-s goal
  sequence it was found in 65 of 80 samples at 0.3-0.85 confidence. Round HUD
  shapes (the ball icon in the radar, the PlayStation icon in name tags,
  teammates' arrows) are masked out or too big. Painted spots on the grass
  (the penalty spot read as a ball at up to 0.79) are flat from the raised
  camera, about twice as wide as tall, so boxes that wide are skipped.
* **Track**: the path through the detections that earns the most confidence
  for the least implausible movement (a longest path over the samples). A
  short run of a white boot cannot outweigh the real ball, and a camera cut
  costs a fixed price instead of an impossible jump. A target the path holds
  for half a second or less beside the ball is then dropped, unless the
  detector was sure of it (the shot into the net just before the cut to the
  celebration): the crop would pan over to it and straight back.
* **Camera**: a dead zone keeps the crop still while the ball moves around
  its middle; then the One-Euro filter and a speed limit smooth the pan.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterator

from ..models import ReframeKeyframe

log = logging.getLogger("clipforge.ballcam")

BALL_FPS = 4.0
BALL_IMGSZ = 1920
BALL_CLASS = 32            # COCO "sports ball"
MIN_CONF = 0.10
MAX_BALL_H = 0.02          # box width as a share of the frame height (the ball reads ~0.01)
FLAT_MIN = 1.5             # box width / height of a painted spot (reads ~2; the ball ~1)
BATCH = 8
# HUD areas (frame fractions x0, y0, x1, y1) where a round shape is never the match ball.
HUD_MASK = (
    (0.42, 0.72, 0.58, 0.88),    # radar, bottom centre
    (0.00, 0.00, 1.00, 0.07),    # stands and ad boards above the pitch
    (0.00, 0.885, 1.00, 1.00),   # player name bars
    (0.00, 0.00, 0.17, 0.14),    # scoreboard
)
# Track search.
V_MAX = 0.9                # plausible ball speed across the screen, frame widths per second
SPEED_COST = 4.0           # per frame width moved beyond that
SKIP_COST = 0.05           # per sample without the ball (it is often hidden by a player)
RESTART_COST = 0.8         # a camera cut: the ball reappears anywhere
MAX_SKIP = 12              # samples a track may jump over (3 s at 4 fps)
MIN_COVERAGE = 0.3         # share of samples on the track needed to follow it
# A run the track holds this long or less (first to last sample) that lies DETOUR_X or
# more beyond the track on both sides is a detour: the crop would pan over and back.
DETOUR_S = 0.5
DETOUR_X = 0.03            # frame widths, about a tenth of the 9:16 crop
SURE_CONF = 0.6            # a detection this sure is the ball: a short visit to it is followed
# Camera.
DEAD_ZONE = 0.05           # the ball may drift this far (frame widths) before the crop pans
MAX_SPEED = 0.6            # crop pan limit, frame widths per second
ONE_EURO_MIN_CUTOFF = 0.8
ONE_EURO_BETA = 1.5

FOOTBALL_PROFILES = frozenset({"eafc"})
_OFF = {"0", "false", "off", "no"}

FrameSource = Callable[..., Iterator[tuple[float, object]]]
Detector = Callable[[list], list[list[tuple[float, float, float, float, float]]]]


@dataclass(frozen=True)
class Candidate:
    x: float                   # centre, share of the frame width
    y: float                   # centre, share of the frame height
    conf: float


def follows(profile: str | None) -> bool:
    """True for games where the crop should follow the ball."""
    if os.environ.get("CLIPFORGE_BALLCAM", "").strip().lower() in _OFF:
        return False
    from ..providers.hud.layouts import normalize_profile

    return normalize_profile(profile) in FOOTBALL_PROFILES


def candidates_from_boxes(boxes, width: int, height: int) -> list[Candidate]:
    """Plausible balls among ``(x0, y0, x1, y1, conf)`` pixel boxes."""
    out: list[Candidate] = []
    for x0, y0, x1, y1, conf in boxes:
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        if conf < MIN_CONF or (x1 - x0) / height > MAX_BALL_H:
            continue
        if x1 - x0 >= FLAT_MIN * (y1 - y0):      # a painted spot on the grass
            continue
        if any(a <= cx <= c and b <= cy <= d for a, b, c, d in HUD_MASK):
            continue
        out.append(Candidate(round(cx, 4), round(cy, 4), round(float(conf), 3)))
    return out


def best_track(samples: list[tuple[float, list[Candidate]]], aspect: float = 16 / 9
               ) -> list[tuple[float, float]]:
    """``[(t, x)]`` of the ball along the best path through the candidates.

    A longest path over the samples: a node per candidate and an edge to every
    candidate up to MAX_SKIP samples later. Taking a candidate earns its
    confidence; skipped samples, movement faster than V_MAX and camera cuts
    cost (a cut never costs more than RESTART_COST). Short detours are then
    dropped (drop_detours).
    """
    nodes = [(i, float(t), c) for i, (t, cs) in enumerate(samples) for c in cs]
    if not nodes:
        return []
    score = [n[2].conf for n in nodes]
    prev = [-1] * len(nodes)
    for k, (i, t, c) in enumerate(nodes):
        for j in range(k - 1, -1, -1):
            i0, t0, c0 = nodes[j]
            if i0 == i:
                continue
            if i - i0 > MAX_SKIP + 1:
                break
            dist = math.hypot(c.x - c0.x, (c.y - c0.y) / aspect)
            move = max(0.0, dist - V_MAX * (t - t0))
            cost = min(SPEED_COST * move, RESTART_COST) + SKIP_COST * (i - i0 - 1)
            if score[j] + c.conf - cost > score[k]:
                score[k], prev[k] = score[j] + c.conf - cost, j
    k = max(range(len(nodes)), key=lambda n: score[n])
    path: list[tuple[float, Candidate]] = []
    while k >= 0:
        path.append((nodes[k][1], nodes[k][2]))
        k = prev[k]
    return [(t, c.x) for t, c in drop_detours(path[::-1])]


def drop_detours(path: list[tuple[float, Candidate]]) -> list[tuple[float, Candidate]]:
    """``path`` without detours: runs held for DETOUR_S or less, with no sure ball
    (SURE_CONF) in them, that lie DETOUR_X or more beyond the points on both sides.

    The longest path takes any candidate the ball could have reached, so a white
    boot beside the ball, or a guess during a camera cut, costs it nothing, and the
    crop pans over and straight back. Without the run, the ball's path is the line
    between its neighbours. Kept: a cut (the ball stays at the new place), a ball
    that stays out longer, and a sure ball (the shot into the net just before the
    game cuts to the celebration).
    """
    pts = list(path)
    changed = True
    while changed:
        changed = False
        k = 1
        while k < len(pts) - 1:
            n = _detour_len(pts, k)
            if n:
                del pts[k:k + n]
                changed = True
            else:
                k += 1
    return pts


def _detour_len(pts: list[tuple[float, Candidate]], k: int) -> int:
    """Points in the shortest detour that starts at ``pts[k]``, or 0."""
    for n in range(1, len(pts) - k):
        t, last = pts[k + n - 1]
        if t - pts[k][0] > DETOUR_S + 1e-9 or last.conf >= SURE_CONF:
            break
        lo, hi = sorted((pts[k - 1][1].x, pts[k + n][1].x))
        run = [p.x for _t, p in pts[k:k + n]]
        if all(x >= hi + DETOUR_X for x in run) or all(x <= lo - DETOUR_X for x in run):
            return n
    return 0


def _interpolate(track: list[tuple[float, float]], times: list[float]) -> list[float]:
    """Ball x at each of ``times``: linear between track points, held at the ends."""
    out: list[float] = []
    j = 0
    for t in times:
        while j + 1 < len(track) and track[j + 1][0] <= t:
            j += 1
        t0, x0 = track[j]
        if t <= t0 or j + 1 >= len(track):
            out.append(x0)
            continue
        t1, x1 = track[j + 1]
        out.append(x0 + (x1 - x0) * (t - t0) / max(t1 - t0, 1e-6))
    return out


def camera_path(track: list[tuple[float, float]], times: list[float],
                crop_half: float) -> list[tuple[float, float]]:
    """Crop centre at each of ``times`` following the ball ``track``."""
    from .reframe import one_euro_filter

    if not track or not times:
        return []
    xs = _interpolate(track, times)
    c = xs[0]
    held: list[tuple[float, float]] = []
    for t, x in zip(times, xs):
        if x - c > DEAD_ZONE:
            c = x - DEAD_ZONE
        elif c - x > DEAD_ZONE:
            c = x + DEAD_ZONE
        held.append((t, c))
    lo, hi = crop_half, 1.0 - crop_half
    out: list[tuple[float, float]] = []
    for t, c in one_euro_filter(held, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA):
        if out:
            t0, c0 = out[-1]
            step = MAX_SPEED * (t - t0)
            c = max(c0 - step, min(c0 + step, c))
        out.append((t, max(lo, min(hi, c))))
    return out


def _yolo_detector() -> Detector | None:
    from ..providers.subject import _load_yolo

    model = _load_yolo()
    if model is None:
        return None

    def detect(frames: list) -> list[list[tuple[float, float, float, float, float]]]:
        results = model.predict(frames, imgsz=BALL_IMGSZ, classes=[BALL_CLASS],
                                conf=MIN_CONF, verbose=False)
        return [[(*(float(v) for v in b.xyxy[0]), float(b.conf[0])) for b in r.boxes]
                for r in results]

    return detect


def _has_nvidia() -> bool:
    try:
        from ..config import get_settings

        return bool(get_settings().has_nvidia)
    except Exception:
        return False


def track_ball(src_path: str, start: float, end: float, info, *,
               frame_source: FrameSource | None = None, detector: Detector | None = None
               ) -> tuple[list[tuple[float, list[Candidate]]], list[tuple[float, float]]]:
    """Candidates per sample and the chosen ball track, times relative to ``start``."""
    width, height = int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
    if width <= 0 or height <= 0 or end - start < 1.0:
        return [], []
    detector = detector or _yolo_detector()
    if detector is None:
        return [], []
    if frame_source is None:
        from ..providers.hud import scan

        def frame_source(region, **kw):
            return scan.iter_frames(src_path, region, **kw)

    samples: list[tuple[float, list[Candidate]]] = []
    times: list[float] = []
    frames: list = []

    def flush() -> None:
        for t, boxes in zip(times, detector(frames)):
            samples.append((round(t - start, 3), candidates_from_boxes(boxes, width, height)))
        times.clear()
        frames.clear()

    for t, frame in frame_source((0, 0, width, height), keyframes=False, fps=BALL_FPS,
                                 start=start, duration=end - start, hwaccel=_has_nvidia(),
                                 pix_fmt="bgr24"):
        times.append(float(t))
        frames.append(frame)
        if len(frames) >= BATCH:
            flush()
    if frames:
        flush()
    return samples, best_track(samples, width / height)


def ball_keyframes(src_path: str, start: float, end: float, info, **kw
                   ) -> list[ReframeKeyframe] | None:
    """Crop keyframes (clip-relative) that follow the ball; None when it was not
    seen often enough to follow."""
    from .reframe import _crop_half, _to_keyframes

    samples, track = track_ball(src_path, start, end, info, **kw)
    if not samples:
        return None
    coverage = len(track) / len(samples)
    if coverage < MIN_COVERAGE:
        log.info("ball cam: ball on %d of %d samples, too few to follow", len(track), len(samples))
        return None
    aspect = int(info.width) / int(info.height)
    path = camera_path(track, [t for t, _ in samples], _crop_half(aspect))
    log.info("ball cam: ball on %d of %d samples, crop pans %.2f..%.2f", len(track), len(samples),
             min(c for _, c in path), max(c for _, c in path))
    return _to_keyframes(path, 0.0)

"""Visual-cue detection by template matching — find exact game events on screen.

The audio matcher (detect_cues.py) hears events; this one *sees* them. Many game
moments come with a distinctive on-screen graphic — the Valorant kill banner, an
"ACE" splash, EA FC's GOAL overlay, Rocket League's goal explosion text — that is
pixel-stable across matches. Drop a cropped reference image of that element into
``<data>/game_cues/<profile>/visual/`` (crop it straight out of a screenshot) and
this matcher finds every frame where it appears.

Method: decode the video at a low fps to small grayscale frames (ffmpeg pipe, so
memory stays constant for multi-hour VODs), then normalized template matching
(OpenCV ``matchTemplate``) at several scales to absorb resolution differences
between the reference screenshot and the footage. OpenCV is an optional
dependency project-wide; without it visual cues are skipped (audio cues and the
energy detector still run).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..media import ffmpeg
from .detect_cues import CueEvent

log = logging.getLogger("clipforge.cues.visual")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

FRAME_W = 480        # frames are downscaled to this width before matching
SAMPLE_FPS = 1.0     # game banners persist >= ~1 s, so 1 fps is enough
# Reference crops usually come from a screenshot of the user's own footage, so
# the template's natural scale is frame-width / footage-width; the sweep (x2
# each way) absorbs a reference captured at a different resolution (a 4K
# screenshot vs 1080p footage, UI-scale settings, web-sourced images).
_SCALES = (0.5, 0.7, 1.0, 1.4, 2.0)


def _load_templates(cues_dir: Path, base_scale: float) -> list[tuple[str, list, object]]:
    """[(label, scaled_variants, orb_descriptor)] for every reference image.

    ``scaled_variants`` feed the matchTemplate pass.
    ``orb_descriptor`` (keypoints+descriptors pre-computed on the full-size
    reference) feeds the ORB fallback that handles rotated/warped banners.
    """
    import cv2
    import numpy as np

    orb = cv2.ORB_create(nfeatures=500)
    out: list[tuple[str, list, object]] = []
    for p in sorted(cues_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            log.warning("visual cue: could not decode %s", p.name)
            continue
        variants = []
        for s in _SCALES:
            f = base_scale * s
            w, h = max(int(img.shape[1] * f), 8), max(int(img.shape[0] * f), 8)
            if w >= FRAME_W or h >= FRAME_W:
                continue
            variants.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
        # Pre-compute ORB descriptors on the full-res reference (before scaling).
        kp, des = orb.detectAndCompute(img, None)
        orb_data = (kp, des)
        if variants:
            out.append((p.stem, variants, orb_data))
    return out


# Minimum good ORB matches to count as a hit, and the Lowe ratio for filtering.
_ORB_MIN_MATCHES = 8
_ORB_LOWE = 0.75
# ORB score reported when keypoint matching succeeds (it's binary pass/fail,
# not a continuous similarity — report a fixed conservative confidence).
_ORB_HIT_SIM = 0.80


def _orb_match(frame, orb_data) -> float:
    """ORB keypoint match score (0.0 or _ORB_HIT_SIM) for a single frame.

    Handles rotated, slightly warped, and differently-scaled banners that
    matchTemplate misses. Uses Lowe ratio test + minimum inlier count.
    Falls back to 0.0 on any failure (no descriptors, tiny frame, etc.).
    """
    import cv2

    kp_ref, des_ref = orb_data
    if des_ref is None or len(des_ref) < _ORB_MIN_MATCHES:
        return 0.0
    orb = cv2.ORB_create(nfeatures=500)
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    if des_frame is None or len(des_frame) < _ORB_MIN_MATCHES:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(des_ref, des_frame, k=2)
    good = [m for m, n in pairs if m.distance < _ORB_LOWE * n.distance]
    return _ORB_HIT_SIM if len(good) >= _ORB_MIN_MATCHES else 0.0


# matchTemplate threshold below which we also try ORB (may be rotated/warped).
_TM_ORB_FALLBACK = 0.50


def _best_match(frame, variants, orb_data) -> float:
    """Best similarity across matchTemplate (multi-scale) + ORB fallback."""
    import cv2

    best = 0.0
    fh, fw = frame.shape
    for t in variants:
        th, tw = t.shape
        if th >= fh or tw >= fw:
            continue
        s = float(cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED).max())
        if s > best:
            best = s
    # ORB handles rotation/warp that TM_CCOEFF_NORMED can't; only run it when
    # TM didn't already give a confident hit (avoid double-counting).
    if best < _TM_ORB_FALLBACK:
        best = max(best, _orb_match(frame, orb_data))
    return best


def nms_hits(hits: list[tuple[float, float]], *, min_gap: float) -> list[tuple[float, float]]:
    """Greedy non-max suppression: keep the strongest hit per ``min_gap`` window."""
    kept: list[tuple[float, float]] = []
    for t, s in sorted(hits, key=lambda h: h[1], reverse=True):
        if all(abs(t - k) >= min_gap for k, _ in kept):
            kept.append((t, s))
    return kept


def _iter_frames(video_path: str, src_w: int, src_h: int, *, fps: float):
    """Yield (t, gray uint8 [h, w]) frames decoded at ``fps`` via an ffmpeg pipe."""
    import numpy as np

    h = max(int(round(FRAME_W * src_h / max(src_w, 1))), 8)
    cmd = [ffmpeg._ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", video_path,
           "-vf", f"fps={fps},scale={FRAME_W}:{h}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    frame_bytes = FRAME_W * h
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        i = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(h, FRAME_W)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def find_visual_events(video_path: str, cues_dir: Path, *, width: int, height: int,
                       threshold: float = 0.72, min_gap: float = 4.0,
                       fps: float = SAMPLE_FPS) -> list[CueEvent]:
    """Match every reference image in ``cues_dir`` against the video's frames."""
    if not cues_dir.is_dir():
        return []
    if not any(p.suffix.lower() in IMG_EXTS for p in cues_dir.iterdir()):
        return []
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except Exception:
        log.info("visual cues present in %s but OpenCV is missing — skipping", cues_dir)
        return []

    templates = _load_templates(cues_dir, FRAME_W / max(width, 1))
    if not templates:
        return []

    hits: dict[str, list[tuple[float, float]]] = {label: [] for label, _, _orb in templates}
    try:
        for t, frame in _iter_frames(video_path, width, height, fps=fps):
            for label, variants, orb_data in templates:
                s = _best_match(frame, variants, orb_data)
                if s >= threshold:
                    hits[label].append((t, s))
    except Exception as e:
        log.warning("visual cue matching failed: %s", e)
        return []

    events = [CueEvent(t=round(t, 3), label=label, similarity=round(s, 3), kind="visual")
              for label, hh in hits.items() for t, s in nms_hits(hh, min_gap=min_gap)]
    events.sort(key=lambda e: e.t)
    log.info("visual cue: matched %d events from %d templates", len(events), len(templates))
    return events

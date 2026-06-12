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


def _load_templates(cues_dir: Path, base_scale: float) -> list[tuple[str, list]]:
    """[(label, [gray template at each usable scale])] for every reference image."""
    import cv2
    import numpy as np

    out: list[tuple[str, list]] = []
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
            if w >= FRAME_W or h >= FRAME_W:   # bigger than any frame — skip scale
                continue
            variants.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
        if variants:
            out.append((p.stem, variants))
    return out


def _best_match(frame, variants) -> float:
    """Best TM_CCOEFF_NORMED score of any scale of one template in one frame."""
    import cv2

    best = 0.0
    fh, fw = frame.shape
    for t in variants:
        th, tw = t.shape
        if th >= fh or tw >= fw:
            continue
        best = max(best, float(cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED).max()))
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

    hits: dict[str, list[tuple[float, float]]] = {label: [] for label, _ in templates}
    try:
        for t, frame in _iter_frames(video_path, width, height, fps=fps):
            for label, variants in templates:
                s = _best_match(frame, variants)
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

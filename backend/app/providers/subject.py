"""Content-aware subject centre for 9:16 reframe — optional upgrade over Haar.

The built-in reframe follows the dominant *face*. That misses the subject when
the face is turned, tiny, or absent (a player's back, a car, a pet). With
ultralytics (YOLO) or MediaPipe installed we locate the most salient subject —
person first, else the largest tracked object — and return its horizontal centre,
so the crop follows the action, not just a face.

Graceful: no backend ⇒ :func:`subject_center` returns None and the reframe falls
back to its face track / centre crop exactly as before.
"""
from __future__ import annotations

import logging

from ..config import get_settings

log = logging.getLogger("clipforge.subject")

_yolo = None  # cached ultralytics model, or False


def _load_yolo():
    global _yolo
    if _yolo is not None:
        return _yolo or None
    try:
        from ultralytics import YOLO

        _yolo = YOLO("yolo11n.pt")  # nano: fast, auto-downloaded once
        log.info("YOLO subject model loaded")
    except Exception as e:
        log.info("YOLO unavailable (%s)", e)
        _yolo = False
    return _yolo or None


def _center_from_boxes(boxes, frame_w: int) -> float | None:
    """Centre-x fraction of the most salient box: people first, then largest.

    Pure helper. ``boxes`` is a list of (cls_is_person, x0, x1, area)."""
    if not boxes or frame_w <= 0:
        return None
    people = [b for b in boxes if b[0]]
    pool = people or boxes
    x0, x1 = max(pool, key=lambda b: b[3])[1:3]
    return max(0.0, min(1.0, ((x0 + x1) / 2) / frame_w))


def subject_center(img) -> float | None:
    """Horizontal centre (0..1) of the dominant subject in a BGR frame, or None."""
    if get_settings().reframe_engine != "yolo":
        return None
    model = _load_yolo()
    if model is None:
        return None
    try:
        h, w = img.shape[:2]
        res = model.predict(img, verbose=False, conf=0.35)[0]
        boxes = []
        for b in res.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
            cls = int(b.cls[0])
            boxes.append((cls == 0, x0, x1, (x1 - x0) * (y1 - y0)))  # 0 = person
        return _center_from_boxes(boxes, w)
    except Exception as e:
        log.warning("YOLO subject detect failed (%s)", e)
        return None

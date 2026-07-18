"""Content-aware subject centre for 9:16 reframe — optional upgrade over face-only tracking.

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
import os
from pathlib import Path

from ..config import get_settings

log = logging.getLogger("clipforge.subject")

def _default_yolo_model() -> str:
    """yolo11s on GPU (better small-subject recall than nano, still ~100 fps),
    nano on CPU. CLIPFORGE_YOLO_MODEL overrides."""
    env = os.environ.get("CLIPFORGE_YOLO_MODEL")
    if env:
        return env
    return "yolo11s.pt" if get_settings().device == "cuda" else "yolo11n.pt"


_YOLO_MODEL = _default_yolo_model()
_yolo = None  # cached ultralytics model, or False

_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
_SAM2_URL = ("https://dl.fbaipublicfiles.com/segment_anything_2/"
             "092824/sam2.1_hiera_tiny.pt")
_SAM2_MIN_BYTES = 100_000_000       # real checkpoint is ~149 MB
_sam2 = None                        # cached SAM2 video predictor, or False


def _sam2_ckpt_path() -> Path:
    env = os.environ.get("CLIPFORGE_SAM2_CKPT")
    if env:
        return Path(env)
    return get_settings().data_dir / "models" / "sam2.1_hiera_tiny.pt"


def _load_sam2():
    """Best-effort SAM2 video predictor (tiny hiera, ~300 MB VRAM on CUDA).

    SAM2 tracks a subject mask through the whole clip with temporal memory —
    much more stable than per-frame YOLO boxes when the camera cuts or the
    subject is briefly occluded. Downloads the checkpoint once on first use.
    """
    global _sam2
    if _sam2 is not None:
        return _sam2 or None
    try:
        from .._util import http_download
        ckpt = _sam2_ckpt_path()
        if not ckpt.exists():
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt.with_suffix(".part")
            http_download(_SAM2_URL, tmp, timeout=300)
            if tmp.stat().st_size < _SAM2_MIN_BYTES:
                tmp.unlink(missing_ok=True)
                raise RuntimeError("SAM2 checkpoint download truncated")
            tmp.replace(ckpt)
            log.info("downloaded SAM2 checkpoint -> %s", ckpt)
        from sam2.build_sam import build_sam2_video_predictor
        device = get_settings().device
        _sam2 = build_sam2_video_predictor(_SAM2_CFG, str(ckpt), device=device)
        log.info("SAM2 video predictor loaded (device=%s)", device)
    except Exception as e:
        log.info("SAM2 unavailable (%s); using per-frame YOLO subject detect", e)
        _sam2 = False
    return _sam2 or None


def _load_yolo():
    global _yolo
    if _yolo is not None:
        return _yolo or None
    try:
        from ultralytics import YOLO

        _yolo = YOLO(_YOLO_MODEL)  # nano default: fast, auto-downloaded once
        if _yolo:
            _yolo.to(get_settings().device)
        log.info("YOLO subject model loaded (%s)", _YOLO_MODEL)
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


def _yolo_seed_box(img) -> tuple | None:
    """Best subject box (x0, y0, x1, y1) from YOLO on a BGR frame, or None."""
    model = _load_yolo()
    if model is None:
        return None
    try:
        res = model.predict(img, verbose=False, conf=0.35)[0]
        best, best_area = None, 0.0
        for b in res.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
            area = (x1 - x0) * (y1 - y0)
            # Prefer people; otherwise take the largest object.
            score = area * (2.0 if int(b.cls[0]) == 0 else 1.0)
            if score > best_area:
                best, best_area = (x0, y0, x1, y1), score
        return best
    except Exception as e:
        log.debug("YOLO seed box failed (%s)", e)
        return None


def subject_centers_sam2(frame_paths: list[Path],
                         sample_width: int) -> dict[int, float] | None:
    """Track the dominant subject across sampled frames with SAM2.

    Seeds the tracker with the YOLO subject box from the first frame that has a
    detection, then propagates the mask forward and backward through the whole
    sampled sequence. Returns ``{frame_index: cx_fraction}`` for every frame
    where the mask is visible, or None when SAM2 is unavailable — the caller
    then falls back to per-frame :func:`subject_center`.

    ``sample_width`` is the width the frames were scaled to (masks are emitted
    at the SAM2 input resolution, so centres are normalised, not pixel-based).
    """
    predictor = _load_sam2()
    if predictor is None or not frame_paths:
        return None
    try:
        import cv2
        import numpy as np

        # Find the seed frame: the first one with a confident YOLO subject box.
        seed_idx, seed_box = None, None
        for i, fp in enumerate(frame_paths[:10]):  # don't scan far for a seed
            img = cv2.imread(str(fp))
            if img is None:
                continue
            box = _yolo_seed_box(img)
            if box is not None:
                seed_idx, seed_box = i, box
                break
        if seed_idx is None or seed_box is None:
            return None

        state = predictor.init_state(str(frame_paths[0].parent))
        predictor.add_new_points_or_box(
            state, frame_idx=seed_idx, obj_id=1,
            box=np.array(seed_box, dtype=np.float32))

        import torch

        def _mask_cx(mask_logits) -> float | None:
            mask = (mask_logits[0] > 0.0)
            if not bool(mask.any()):
                return None
            _ys, xs = torch.nonzero(mask, as_tuple=True)
            if len(xs) == 0:
                return None
            cx = float(xs.float().mean()) / mask.shape[-1]
            return max(0.0, min(1.0, cx))

        out: dict[int, float] = {}
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            cx = _mask_cx(mask_logits)
            if cx is not None:
                out[frame_idx] = cx

        # Track backward from the seed too, so frames before it are covered.
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
                state, reverse=True):
            if frame_idx in out:
                continue
            cx = _mask_cx(mask_logits)
            if cx is not None:
                out[frame_idx] = cx

        predictor.reset_state(state)
        return out or None
    except Exception as e:
        log.info("SAM2 subject tracking failed (%s)", e)
        return None

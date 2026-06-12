"""Unified face detection — three-tier cascade.

Tier 1 — **MediaPipe** (opt-in, ``pip install mediapipe``):
  BlazeFace CNN, 30–70 FPS CPU with built-in frame-to-frame tracking.
  Handles side-profiles better than Haar and needs no model download.
  Apache 2.0 — fully commercial-safe. Install to enable, auto-detected.

Tier 2 — **YuNet** (default when OpenCV is present):
  337 KB ONNX model shipped via ``cv2.FaceDetectorYN``; excellent at
  small boxes and partial occlusion (streamer facecam corner). One
  best-effort model download on first use; degrades to Haar if offline.

Tier 3 — **Haar cascade** (always available with OpenCV):
  Fast, CPU-only, frontal faces only. Final fallback.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from ..config import get_settings

log = logging.getLogger("clipforge.faces")

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
_YUNET_MIN_BYTES = 100_000

_lock = threading.Lock()
_yunet = None
_haar = None
_mediapipe = None   # MediaPipe FaceDetector (opt-in, Apache 2.0)

_MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                 "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
_MP_MIN_BYTES = 100_000


def _yunet_path() -> Path:
    env = os.environ.get("CLIPFORGE_YUNET_PATH")
    if env:
        return Path(env)
    return get_settings().data_dir / "models" / "face_detection_yunet_2023mar.onnx"


def _fetch_yunet(dst: Path) -> bool:
    """One best-effort model download (offline installs just use Haar)."""
    import urllib.request

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(YUNET_URL, timeout=15) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        if tmp.stat().st_size < _YUNET_MIN_BYTES:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        log.info("downloaded YuNet face model -> %s", dst)
        return True
    except Exception as e:
        log.info("YuNet model unavailable (%s); using Haar cascade", e)
        return False


def _mp_model_path() -> Path:
    env = os.environ.get("CLIPFORGE_MP_FACE_MODEL")
    if env:
        return Path(env)
    return get_settings().data_dir / "models" / "blaze_face_short_range.tflite"


def _fetch_mp_model(dst: Path) -> bool:
    """One best-effort download of the BlazeFace TFLite model (~230 KB)."""
    import urllib.request

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        req = urllib.request.Request(_MP_MODEL_URL,
                                     headers={"User-Agent": "ClipForge/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        if tmp.stat().st_size < _MP_MIN_BYTES:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        log.info("downloaded MediaPipe BlazeFace model -> %s", dst)
        return True
    except Exception as e:
        log.info("MediaPipe model unavailable (%s); using YuNet/Haar", e)
        return False


def _get_mediapipe():
    """MediaPipe BlazeFace detector (tier 1, opt-in).

    30–70 FPS on CPU, handles side-profiles, Apache 2.0.
    Requires a one-time ~230 KB model download (same pattern as YuNet).
    Returns the detector or None when mediapipe isn't installed or offline.
    """
    global _mediapipe
    if _mediapipe is not None:
        return None if _mediapipe == "unavailable" else _mediapipe
    try:
        import mediapipe as mp

        path = _mp_model_path()
        if not path.exists() and not _fetch_mp_model(path):
            _mediapipe = "unavailable"
            return None
        opts = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        _mediapipe = mp.tasks.vision.FaceDetector.create_from_options(opts)
        log.info("face detection: MediaPipe BlazeFace loaded (tier 1)")
        return _mediapipe
    except Exception as e:
        log.info("MediaPipe unavailable (%s); using YuNet/Haar", e)
        _mediapipe = "unavailable"
        return None


def _get_yunet():
    global _yunet
    if _yunet is not None:
        return None if _yunet == "unavailable" else _yunet
    import cv2

    if not hasattr(cv2, "FaceDetectorYN"):
        _yunet = "unavailable"
        return None
    path = _yunet_path()
    if not path.exists() and not _fetch_yunet(path):
        _yunet = "unavailable"
        return None
    try:
        _yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320),
                                           score_threshold=0.6)
        return _yunet
    except Exception as e:
        log.warning("YuNet load failed (%s); using Haar cascade", e)
        _yunet = "unavailable"
        return None


def _get_haar():
    global _haar
    if _haar is None:
        import cv2

        _haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _haar


def detect_faces(img_bgr, *, min_size_frac: float = 0.03) -> list[tuple[int, int, int, int]]:
    """Face boxes [(x, y, w, h)] in pixels for one BGR frame.

    ``min_size_frac`` is the minimum face width as a fraction of frame width —
    keeps tiny in-game character faces from registering.

    Detection tier used: InsightFace buffalo_s (if installed) → YuNet → Haar.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    min_px = max(int(w * min_size_frac), 10)

    # --- Tier 1: MediaPipe BlazeFace (opt-in) --------------------------------
    mp_det = _get_mediapipe()
    if mp_det is not None:
        try:
            import mediapipe as mp

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=img_bgr[..., ::-1].copy())  # BGR→RGB
            with _lock:
                result = mp_det.detect(mp_img)
            out = []
            for det in result.detections:
                bb = det.bounding_box
                fx, fy, fw, fh = bb.origin_x, bb.origin_y, bb.width, bb.height
                if fw >= min_px and fh >= min_px:
                    out.append((max(int(fx), 0), max(int(fy), 0),
                                min(int(fw), w - int(fx)),
                                min(int(fh), h - int(fy))))
            return out
        except Exception as e:
            log.debug("MediaPipe inference failed (%s); falling back", e)

    # --- Tier 2: YuNet -------------------------------------------------------
    with _lock:
        det = _get_yunet()
        if det is not None:
            det.setInputSize((w, h))
            _, faces = det.detect(img_bgr)
            out = []
            for f in (faces if faces is not None else []):
                fx, fy, fw, fh = (int(round(v)) for v in f[:4])
                if fw >= min_px and fh >= min_px:
                    fx, fy = max(fx, 0), max(fy, 0)
                    out.append((fx, fy, min(fw, w - fx), min(fh, h - fy)))
            return out

    # --- Tier 3: Haar cascade ------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    with _lock:
        faces = _get_haar().detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                             minSize=(max(min_px, 30), max(min_px, 30)))
    return [tuple(int(v) for v in f) for f in faces]


def active_tier() -> str:
    """Which detection tier is currently loaded ('mediapipe'/'yunet'/'haar')."""
    if _mediapipe and _mediapipe != "unavailable":
        return "mediapipe"
    if _yunet and _yunet != "unavailable":
        return "yunet"
    return "haar"

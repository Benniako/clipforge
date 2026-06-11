"""Unified face detection — YuNet (CNN) when available, Haar cascade otherwise.

YuNet is a 337KB ONNX model that detects faces down to ~10x10px, including side
faces and partial occlusion — exactly the regime where a streamer's small corner
facecam lives and where the Haar cascade fails. opencv-python ships the
``cv2.FaceDetectorYN`` runtime but not the model file, so we look for it in the
data dir (or ``CLIPFORGE_YUNET_PATH``) and make one best-effort download attempt;
everything degrades to the Haar cascade if neither works.
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
_YUNET_MIN_BYTES = 100_000          # sanity floor for a complete download

_lock = threading.Lock()            # FaceDetectorYN instances aren't thread-safe
_yunet = None                       # cached detector ("unavailable" = gave up)
_haar = None


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
    """
    import cv2

    h, w = img_bgr.shape[:2]
    min_px = max(int(w * min_size_frac), 10)
    with _lock:
        det = _get_yunet()
        if det is not None:
            det.setInputSize((w, h))
            _, faces = det.detect(img_bgr)
            out = []
            for f in (faces if faces is not None else []):
                fx, fy, fw, fh = (int(round(v)) for v in f[:4])
                if fw >= min_px and fh >= min_px:
                    # clamp — YuNet can return boxes slightly outside the frame
                    fx, fy = max(fx, 0), max(fy, 0)
                    out.append((fx, fy, min(fw, w - fx), min(fh, h - fy)))
            return out
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # CascadeClassifier isn't thread-safe either — the pipeline worker and an
    # editor-triggered reframe can call this concurrently.
    with _lock:
        faces = _get_haar().detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                             minSize=(max(min_px, 30), max(min_px, 30)))
    return [tuple(int(v) for v in f) for f in faces]

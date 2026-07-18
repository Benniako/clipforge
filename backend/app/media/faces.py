"""Unified face detection — YOLO/YuNet first, optional OpenCV fallback last.

YuNet is a 337KB ONNX model that detects faces down to ~10x10px, including side
faces and partial occlusion — exactly the regime where a streamer's small corner
facecam lives and where the old Haar cascade fails. opencv-python ships the
``cv2.FaceDetectorYN`` runtime but not the model file, so we look for it in the
data dir (or ``CLIPFORGE_YUNET_PATH``) and make one best-effort download attempt;
everything degrades to no face boxes if the legacy cascade is not present.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from .._util import http_download
from ..config import get_settings

log = logging.getLogger("clipforge.faces")

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
_YUNET_MIN_BYTES = 100_000          # sanity floor for a complete download

_lock = threading.Lock()            # FaceDetectorYN instances aren't thread-safe
_yunet = None                       # cached detector ("unavailable" = gave up)
_haar = None                       # cached CascadeClassifier, or False if absent
_yolo_face = None                   # cached ultralytics YOLO model or False
_scrfd = None                       # cached SCRFD session or False

_SCRFD_URL = ("https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/"
              "resolve/main/scrfd_2.5g_bnkps.onnx")
_SCRFD_MIN_BYTES = 1_000_000        # real model is ~3.3 MB


def _scrfd_path() -> Path:
    env = os.environ.get("CLIPFORGE_SCRFD_PATH")
    if env:
        return Path(env)
    return get_settings().data_dir / "models" / "scrfd_2.5g_bnkps.onnx"


def _get_scrfd():
    """SCRFD 2.5G face detector (ONNX, GPU via onnxruntime when available).

    Better accuracy than YuNet on small/profile faces (WIDER FACE hard set:
    77.9 vs ~70 for YuNet) and runs on CUDA when the onnxruntime-gpu build is
    installed. Falls back to CPU execution transparently.
    """
    global _scrfd
    if _scrfd is not None:
        return None if _scrfd is False else _scrfd
    try:
        path = _scrfd_path()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            http_download(_SCRFD_URL, tmp, timeout=30)
            if tmp.stat().st_size < _SCRFD_MIN_BYTES:
                tmp.unlink(missing_ok=True)
                raise RuntimeError("SCRFD download truncated")
            tmp.replace(path)
            log.info("downloaded SCRFD face model -> %s", path)
        from scrfd import SCRFD

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if get_settings().device != "cuda":
            providers = ["CPUExecutionProvider"]
        _scrfd = SCRFD.from_path(str(path), providers=providers)
        used = _scrfd._inner.session.get_providers()
        log.info("SCRFD face detector loaded (%s)", used[0])
        return _scrfd
    except Exception as e:
        log.info("SCRFD unavailable (%s); using YuNet/OpenCV fallback", e)
        _scrfd = False
        return None


def _get_yolo_face():
    global _yolo_face
    if _yolo_face is not None:
        return _yolo_face
    try:
        from ultralytics import YOLO
        # YOLOv8n-face: lightweight face-specific model (~3MB, runs at 200+ fps
        # on RTX. Better accuracy than YuNet, especially for profile/occluded
        # faces and the small corner facecam that video reframing needs.
        _yolo_face = YOLO("yolov8n-face.pt")
        # Ultralytics auto-detects CUDA, but explicitly passing the config
        # device ensures CLIPFORGE_DEVICE=cpu is honoured.
        if _yolo_face:
            _yolo_face.to(get_settings().device)
        return _yolo_face
    except Exception as e:
        log.info("YOLOv8-face unavailable (%s); using YuNet/OpenCV fallback", e)
        _yolo_face = False
        return None


def _yunet_path() -> Path:
    env = os.environ.get("CLIPFORGE_YUNET_PATH")
    if env:
        return Path(env)
    return get_settings().data_dir / "models" / "face_detection_yunet_2023mar.onnx"


def _fetch_yunet(dst: Path) -> bool:
    """One best-effort model download (offline installs skip this detector)."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        http_download(YUNET_URL, tmp, timeout=15)
        if tmp.stat().st_size < _YUNET_MIN_BYTES:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        log.info("downloaded YuNet face model -> %s", dst)
        return True
    except Exception as e:
        log.info("YuNet model unavailable (%s); using OpenCV fallback", e)
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
        log.warning("YuNet load failed (%s); using OpenCV fallback", e)
        _yunet = "unavailable"
        return None


def _get_haar():
    global _haar
    if _haar is False:
        return None
    if _haar is None:
        try:
            import cv2

            cascade_cls = getattr(cv2, "CascadeClassifier", None)
            data = getattr(cv2, "data", None)
            cascade_dir = getattr(data, "haarcascades", "")
            if cascade_cls is None or not cascade_dir:
                log.info("OpenCV Haar cascade runtime is unavailable")
                _haar = False
                return None

            cascade_path = Path(cascade_dir) / "haarcascade_frontalface_default.xml"
            if not cascade_path.is_file():
                log.info("OpenCV Haar cascade file is unavailable: %s", cascade_path)
                _haar = False
                return None

            cascade = cascade_cls(str(cascade_path))
            if hasattr(cascade, "empty") and cascade.empty():
                log.info("OpenCV Haar cascade failed to load: %s", cascade_path)
                _haar = False
                return None
            _haar = cascade
        except Exception as exc:
            log.info("OpenCV Haar cascade unavailable (%s)", exc)
            _haar = False
            return None
    return _haar


def detect_faces(img_bgr, *, min_size_frac: float = 0.03) -> list[tuple[int, int, int, int]]:
    """Face boxes [(x, y, w, h)] in pixels for one BGR frame.

    ``min_size_frac`` is the minimum face width as a fraction of frame width —
    keeps tiny in-game character faces from registering.

    Detection priority: YOLOv8-face (best, GPU) → SCRFD (ONNX, GPU/CPU) →
    YuNet (ONNX) → Haar if the installed OpenCV build still ships it. OpenCV 5
    moved Haar/HOG detectors out of the main package, so the final fallback may
    be absent.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    min_px = max(int(w * min_size_frac), 10)

    # 1. YOLOv8-face (GPU via ultralytics) — best accuracy, handles profile faces.
    yolo = _get_yolo_face()
    if yolo is not None:
        try:
            results = yolo(img_bgr, conf=0.4, iou=0.5, verbose=False)
            out = []
            for r in results:
                for box in (r.boxes or []):
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    fw, fh = x2 - x1, y2 - y1
                    if fw >= min_px and fh >= min_px:
                        out.append((x1, y1, fw, fh))
            if out:
                return out
        except Exception:
            log.debug("YOLO face detection failed; falling through to SCRFD")

    # 2. SCRFD (ONNX; CUDA when onnxruntime-gpu + cuFFT are present).
    scrfd = _get_scrfd()
    if scrfd is not None:
        try:
            from scrfd import Threshold

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image

            det = scrfd.detect(Image.fromarray(rgb),
                               threshold=Threshold(probability=0.4))
            out = []
            for f in det or []:
                fx, fy = int(f.bbox.upper_left.x), int(f.bbox.upper_left.y)
                fw, fh = int(f.bbox.width()), int(f.bbox.height())
                if fw >= min_px and fh >= min_px:
                    fx, fy = max(fx, 0), max(fy, 0)
                    out.append((fx, fy, min(fw, w - fx), min(fh, h - fy)))
            if out:
                return out
        except Exception:
            log.debug("SCRFD detection failed; falling through to YuNet")

    # 3. YuNet (ONNX, OpenCV DNN).
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
            if out:
                return out

    # 4. Haar cascade (CPU, legacy fallback; absent in some OpenCV 5 builds).
    with _lock:
        haar = _get_haar()
        if haar is None:
            return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                      minSize=(max(min_px, 30), max(min_px, 30)))
    return [tuple(int(v) for v in f) for f in faces]

"""Unified face detection — SAM2/YOLO/MediaPipe/YuNet/Haar cascade.

Detection runs in priority order, returning the first tier that produces boxes:

1. **SAM2** (best, GPU via segment-anything-2) — Meta's Segment Anything Model 2
   for subject segmentation; produces accurate bounding boxes around any detected
   subject. Opt-in (``pip install segment-anything-2``).
2. **YOLOv8-face** (GPU via ultralytics) — best face accuracy, handles profile
   faces. Opt-in (``pip install ultralytics``).
3. **MediaPipe BlazeFace** (opt-in, ``pip install mediapipe``) — 30-70 FPS CPU,
   handles side-profiles better than YuNet/Haar, Apache 2.0 / commercial-safe.
   One ~230 KB model download on first use (same pattern as YuNet).
4. **YuNet** (default with OpenCV) — 337KB ONNX model via
   ``cv2.FaceDetectorYN``; excellent at small boxes and partial occlusion
   (streamer facecam corner). One best-effort download; degrades to Haar.
5. **Haar cascade** (final fallback) — fast, CPU-only, frontal faces only.
   Absent in some OpenCV 5 builds (handled gracefully).

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

# MediaPipe BlazeFace short-range TFLite model (~230 KB). Tasks API requires a
# real model_asset_path (None fails at runtime); download once on first use.
_MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                 "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
_MP_MIN_BYTES = 100_000

_lock = threading.Lock()            # FaceDetectorYN instances aren't thread-safe
_yunet = None                       # cached detector ("unavailable" = gave up)
_haar = None                       # cached CascadeClassifier, or False if absent
_yolo_face = None                   # cached ultralytics YOLO model or False
_mediapipe = None                   # cached MediaPipe FaceDetector or "unavailable"
_sam2 = None                        # cached SAM2 predictor or "unavailable"

# SAM2 — Segment Anything Model 2 (highest priority, opt-in).
# ~2.4 GB for the large model; downloaded once on first use.
_SAM2_MODEL_URL = ("https://dl.fbaipublicfiles.com/segment_anything_2/"
                   "092824/sam2.1_hiera_large.pt")
_SAM2_MIN_BYTES = 1_000_000


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


def _mp_model_path() -> Path:
    env = os.environ.get("CLIPFORGE_MP_FACE_MODEL")
    if env:
        return Path(env)
    return get_settings().data_dir / "models" / "blaze_face_short_range.tflite"


def _fetch_mp_model(dst: Path) -> bool:
    """One best-effort download of the BlazeFace TFLite model (~230 KB)."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        http_download(_MP_MODEL_URL, tmp, timeout=15)
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
    """MediaPipe BlazeFace detector (tier 3, opt-in).

    30-70 FPS on CPU, handles side-profiles, Apache 2.0. Requires a one-time
    ~230 KB model download (same pattern as YuNet). Returns the detector or
    None when mediapipe isn't installed or the model can't be fetched.
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
        log.info("face detection: MediaPipe BlazeFace loaded (tier 3)")
        return _mediapipe
    except Exception as e:
        log.info("MediaPipe unavailable (%s); using YuNet/Haar", e)
        _mediapipe = "unavailable"
        return None


# ---------------------------------------------------------------------------
# SAM2 — Segment Anything Model 2 (tier 1, highest priority, opt-in)
# ---------------------------------------------------------------------------

def _sam2_model_path() -> Path:
    """Resolve the SAM2 checkpoint path.

    Checks ``CLIPFORGE_SAM2_MODEL`` env var first, then the data dir.
    """
    env = os.environ.get("CLIPFORGE_SAM2_MODEL")
    if env:
        return Path(env)
    return get_settings().data_dir / "models" / "sam2.1_hiera_large.pt"


def _fetch_sam2(dst: Path) -> bool:
    """One best-effort download of the SAM2 checkpoint (~2.4 GB)."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        http_download(_SAM2_MODEL_URL, tmp, timeout=30)
        if tmp.stat().st_size < _SAM2_MIN_BYTES:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        log.info("downloaded SAM2 model -> %s", dst)
        return True
    except Exception as e:
        log.info("SAM2 model download failed (%s); falling back to YOLO", e)
        return False


def _get_sam2():
    """SAM2 predictor (tier 1, highest priority, opt-in).

    Returns a ``Sam2ImagePredictor`` or ``None`` when ``segment-anything-2``
    (or ``sam2``) isn't installed or the model can't be loaded.
    """
    global _sam2
    if _sam2 is not None:
        return None if _sam2 == "unavailable" else _sam2
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        try:
            from segment_anything_2.build_sam import build_sam2
            from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            _sam2 = "unavailable"
            return None

    path = _sam2_model_path()
    if not path.exists() and not _fetch_sam2(path):
        _sam2 = "unavailable"
        return None
    try:
        device = get_settings().device
        sam2_model = build_sam2(
            model_cfg="sam2.1_hiera_l.yaml",
            ckpt=str(path),
            device=device,
        )
        _sam2 = SAM2ImagePredictor(sam2_model)
        log.info("face detection: SAM2 loaded (tier 1)")
        return _sam2
    except Exception as e:
        log.info("SAM2 unavailable (%s); falling back to YOLO", e)
        _sam2 = "unavailable"
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

    Detection priority: SAM2 (best, GPU subject segmentation) → YOLOv8-face
    (GPU) → MediaPipe BlazeFace (CPU, side-profiles, opt-in) → YuNet (ONNX) →
    Haar if the installed OpenCV build still ships it. OpenCV 5 moved Haar/HOG
    detectors out of the main package, so the final fallback may be absent.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    min_px = max(int(w * min_size_frac), 10)

    # 1. SAM2 (Segment Anything 2) — best subject segmentation, GPU via
    #    segment-anything-2. Produces accurate bounding boxes around any
    #    detected subject (person/face).
    sam2 = _get_sam2()
    if sam2 is not None:
        try:
            import numpy as np

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            sam2.set_image(rgb)
            # Use a grid of foreground points to prompt automatic detection.
            n_pts = 16
            grid_x = np.linspace(0, w - 1, n_pts, dtype=np.float32)
            grid_y = np.linspace(0, h - 1, n_pts, dtype=np.float32)
            gx, gy = np.meshgrid(grid_x, grid_y)
            point_coords = np.stack([gx.ravel(), gy.ravel()], axis=-1)
            point_labels = np.ones(len(point_coords), dtype=np.int32)
            masks, scores, _ = sam2.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            out = []
            for mask, score in zip(masks, scores):
                if score < 0.5:
                    continue
                ys, xs = np.where(mask)
                if len(xs) < 10:
                    continue
                fx, fy = int(xs.min()), int(ys.min())
                fw, fh = int(xs.max() - fx), int(ys.max() - fy)
                if fw >= min_px and fh >= min_px:
                    out.append((max(fx, 0), max(fy, 0),
                                min(fw, w - fx), min(fh, h - fy)))
            if out:
                return out
        except Exception as e:
            log.debug("SAM2 inference failed (%s); falling through to YOLO", e)

    # 2. YOLOv8-face (GPU via ultralytics) — best accuracy, handles profile faces.
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
            log.debug("YOLO face detection failed; falling through to MediaPipe")

    # 3. MediaPipe BlazeFace (opt-in, CPU, handles side-profiles).
    mp_det = _get_mediapipe()
    if mp_det is not None:
        try:
            import mediapipe as mp

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=img_bgr[..., ::-1].copy())  # BGR -> RGB
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
            if out:
                return out
        except Exception as e:
            log.debug("MediaPipe inference failed (%s); falling back", e)

    # 4. YuNet (ONNX, OpenCV DNN).
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

    # 5. Haar cascade (CPU, legacy fallback; absent in some OpenCV 5 builds).
    with _lock:
        haar = _get_haar()
        if haar is None:
            return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                      minSize=(max(min_px, 30), max(min_px, 30)))
    return [tuple(int(v) for v in f) for f in faces]


def active_tier() -> str:
    """Which face-detection tier is currently loaded.

    Returns one of ``"sam2"``, ``"yolo"``, ``"mediapipe"``, ``"yunet"``,
    ``"haar"`` — the first tier (in priority order) that is actually
    available. Surfaced in /api/health so the UI can show which engine is
    driving face tracking.
    """
    if _sam2 and _sam2 is not False and _sam2 != "unavailable":
        return "sam2"
    if _yolo_face and _yolo_face is not False:
        return "yolo"
    if _mediapipe and _mediapipe != "unavailable":
        return "mediapipe"
    if _yunet and _yunet != "unavailable":
        return "yunet"
    return "haar"

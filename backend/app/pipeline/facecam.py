"""Facecam handling for gameplay footage.

A streamer's webcam overlay is the one face that stays in the same small region
for the whole VOD. We sample frames across the source, cluster face detections
over time, and accept a cluster that is present in most samples with almost no
positional drift — then expand the median face box to an estimate of the camera
overlay rectangle (a face fills roughly a third of a typical cam frame).
When NVIDIA background removal hides the rectangular webcam background and makes
faces harder to detect, an optional YOLO fallback looks for a stable small
person cutout in the same way.

The detected ``Rect`` enables:
  • split / framed layouts (cam stacked above or overlaid on the gameplay),
  • a "reaction" scoring signal — motion energy inside the cam during a
    highlight (the streamer jumping, laughing, leaning in),
  • excluding the cam from the gameplay crop so it never appears twice.

Everything degrades to None when OpenCV/faces are unavailable.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg
from ..models import Rect

log = logging.getLogger("clipforge.facecam")

SAMPLE_FRAMES = 20         # independent seeks across the whole source
SAMPLE_WIDTH = 640         # facecam faces are small; detect at a higher res
MIN_PRESENCE = 0.55        # cluster must appear in >= this share of samples
MAX_DRIFT = 0.025          # max std-dev of the cluster centre (fractions)
MAX_FACE_FRAC = 0.22       # a "facecam" face wider than this is a talking head
# Face box -> overlay rectangle expansion (face sits upper-middle of the cam).
EXPAND_W, EXPAND_H = 2.1, 2.5
MIN_PERSON_PRESENCE = 0.35
MAX_PERSON_DRIFT = 0.045
MAX_PERSON_W = 0.38
MAX_PERSON_H = 0.72


def detect_facecam(src_path: str, duration: float) -> Rect | None:
    """Detect a static webcam overlay; returns fractions of the source frame."""
    if not get_settings().has_opencv or duration <= 0:
        return None
    try:
        import cv2
    except Exception:
        return None
    from ..media import faces as faces_mod

    samples: list[list[tuple[float, float, float, float]]] = []
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(SAMPLE_FRAMES):
            t = duration * (i + 0.5) / SAMPLE_FRAMES
            fp = Path(tmp) / f"f_{i:03d}.jpg"
            try:
                ffmpeg.grab_frame(src_path, fp, t=t, width=SAMPLE_WIDTH, quality=4)
            except Exception:
                continue
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes = faces_mod.detect_faces(img, min_size_frac=0.025)
            samples.append([(x / w, y / h, fw / w, fh / h)
                            for x, y, fw, fh in boxes])
            frames.append(img)
    rect = stable_face_cluster(samples)
    if rect is not None:
        cam = expand_to_overlay(rect)
    else:
        person = stable_person_cluster(_person_samples(frames))
        if person is None:
            return None
        cam = expand_person_to_overlay(person)
    log.info("facecam detected at x=%.2f y=%.2f w=%.2f h=%.2f",
             cam.x, cam.y, cam.w, cam.h)
    return cam


def stable_face_cluster(
        samples: list[list[tuple[float, float, float, float]]]) -> Rect | None:
    """The median face box of a temporally stable cluster, or None.

    Pure function (no cv2) so the clustering rules are unit-testable. Boxes are
    (x, y, w, h) fractions. Greedy clustering by centre distance; a cluster
    qualifies as a facecam when it shows up in most samples, barely moves, and
    is small enough that it isn't just a talking head filling the frame.
    """
    valid = [s for s in samples if s is not None]
    if len(valid) < 6:
        return None
    clusters: list[list[tuple[float, float, float, float]]] = []
    for boxes in valid:
        for b in boxes:
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            for cl in clusters:
                m = cl[len(cl) // 2]
                mx, my = m[0] + m[2] / 2, m[1] + m[3] / 2
                if abs(cx - mx) < 0.06 and abs(cy - my) < 0.06:
                    cl.append(b)
                    break
            else:
                clusters.append([b])

    best: Rect | None = None
    best_n = 0
    for cl in clusters:
        if len(cl) < max(int(len(valid) * MIN_PRESENCE), 4) or len(cl) <= best_n:
            continue
        xs = sorted(b[0] + b[2] / 2 for b in cl)
        ys = sorted(b[1] + b[3] / 2 for b in cl)
        if _std(xs) > MAX_DRIFT or _std(ys) > MAX_DRIFT:
            continue
        ws = sorted(b[2] for b in cl)
        hs = sorted(b[3] for b in cl)
        w, h = ws[len(ws) // 2], hs[len(hs) // 2]
        if w > MAX_FACE_FRAC:        # that's a talking head, not an overlay
            continue
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
        best = Rect(x=cx - w / 2, y=cy - h / 2, w=w, h=h).clamped()
        best_n = len(cl)
    return best


def expand_to_overlay(face: Rect) -> Rect:
    """Grow a face box to the (estimated) webcam overlay rectangle."""
    w = face.w * EXPAND_W
    h = face.h * EXPAND_H
    cx = face.x + face.w / 2
    # faces sit in the upper-middle of a cam frame: bias the box downward
    y = face.y - h * 0.18
    return Rect(x=cx - w / 2, y=y, w=w, h=h).clamped()


def stable_person_cluster(
        samples: list[list[tuple[float, float, float, float]]]) -> Rect | None:
    """Stable small person cutout fallback for background-removed facecams."""
    valid = [s for s in samples if s is not None]
    if len(valid) < 6:
        return None
    clusters: list[list[tuple[float, float, float, float]]] = []
    for boxes in valid:
        for b in boxes:
            if b[2] > MAX_PERSON_W or b[3] > MAX_PERSON_H:
                continue
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            for cl in clusters:
                m = cl[len(cl) // 2]
                mx, my = m[0] + m[2] / 2, m[1] + m[3] / 2
                if abs(cx - mx) < 0.10 and abs(cy - my) < 0.10:
                    cl.append(b)
                    break
            else:
                clusters.append([b])

    best: Rect | None = None
    best_n = 0
    for cl in clusters:
        if len(cl) < max(int(len(valid) * MIN_PERSON_PRESENCE), 4) or len(cl) <= best_n:
            continue
        xs = sorted(b[0] + b[2] / 2 for b in cl)
        ys = sorted(b[1] + b[3] / 2 for b in cl)
        if _std(xs) > MAX_PERSON_DRIFT or _std(ys) > MAX_PERSON_DRIFT:
            continue
        ws = sorted(b[2] for b in cl)
        hs = sorted(b[3] for b in cl)
        w, h = ws[len(ws) // 2], hs[len(hs) // 2]
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
        best = Rect(x=cx - w / 2, y=cy - h / 2, w=w, h=h).clamped()
        best_n = len(cl)
    return best


def expand_person_to_overlay(person: Rect) -> Rect:
    """Pad a background-removed person cutout into a usable PiP/stack crop."""
    pad_x = person.w * 0.20
    pad_y = person.h * 0.08
    return Rect(x=person.x - pad_x, y=person.y - pad_y,
                w=person.w + pad_x * 2, h=person.h + pad_y * 2).clamped()


def _person_samples(frames) -> list[list[tuple[float, float, float, float]]]:
    if not frames:
        return []
    try:
        from ..providers import subject as subject_mod

        model = subject_mod._load_yolo()
    except Exception:
        model = None
    if model is None:
        return []
    out = []
    for img in frames:
        try:
            h, w = img.shape[:2]
            res = model.predict(img, verbose=False, conf=0.30)[0]
            boxes = []
            for b in res.boxes:
                if int(b.cls[0]) != 0:  # COCO person
                    continue
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
                boxes.append((x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h))
            out.append(boxes)
        except Exception as e:
            log.warning("YOLO facecam fallback failed: %s", e)
            out.append([])
    return out


def _std(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    return (sum((v - mean) ** 2 for v in vals) / n) ** 0.5


# --------------------------------------------------------------------------- #
# Reaction energy — does the streamer react inside the cam during a window?
# --------------------------------------------------------------------------- #
def reaction_energy(src_path: str, cam: Rect, t0: float, t1: float,
                    *, samples: int = 4) -> float | None:
    """Mean motion energy [0..1] inside the facecam over [t0, t1].

    Samples a handful of frames, crops to the cam region, and averages the
    absolute frame difference — cheap, and a strong proxy for "the streamer is
    going off". Returns None when frames can't be read.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    span = max(t1 - t0, 0.5)
    crops = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(samples):
            t = t0 + span * (i + 0.5) / samples
            fp = Path(tmp) / f"r_{i:02d}.jpg"
            try:
                ffmpeg.run(["-ss", f"{t:.2f}", "-i", src_path, "-frames:v", "1",
                            "-vf", f"crop=iw*{cam.w:.4f}:ih*{cam.h:.4f}"
                                   f":iw*{cam.x:.4f}:ih*{cam.y:.4f},scale=96:96",
                            "-q:v", "5", str(fp)], timeout=60)
            except Exception:
                continue
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                crops.append(img.astype("float32"))
    if len(crops) < 2:
        return None
    diffs = [float(np.abs(a - b).mean()) / 255.0
             for a, b in zip(crops, crops[1:])]
    # ~0.04 mean-abs-diff is already a lively cam; saturate around there.
    return min(sum(diffs) / len(diffs) / 0.04, 1.0)


# --------------------------------------------------------------------------- #
# Action centroid — where in the frame is the gameplay actually happening?
# --------------------------------------------------------------------------- #
def action_center(src_path: str, t0: float, t1: float, cam: Rect | None = None,
                  *, samples: int = 5) -> float | None:
    """Horizontal centre [0..1] of motion energy over the window, or None.

    Frame-differences a few downscaled frames and takes the motion-weighted
    column centroid, ignoring the facecam region. Clamped to the middle band so
    a HUD flicker can't fling the crop to an edge.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    span = max(t1 - t0, 0.5)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(samples):
            t = t0 + span * (i + 0.5) / samples
            fp = Path(tmp) / f"a_{i:02d}.jpg"
            try:
                ffmpeg.run(["-ss", f"{t:.2f}", "-i", src_path, "-frames:v", "1",
                            "-vf", "scale=160:90", "-q:v", "5", str(fp)],
                           timeout=60)
            except Exception:
                continue
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img.astype("float32"))
    if len(frames) < 2:
        return None
    h, w = frames[0].shape
    energy = np.zeros((h, w), dtype="float32")
    for a, b in zip(frames, frames[1:]):
        if a.shape == b.shape:
            energy += np.abs(a - b)
    if cam is not None:   # the streamer moving shouldn't drag the game crop
        x0, x1 = int(cam.x * w), min(int((cam.x + cam.w) * w) + 1, w)
        y0, y1 = int(cam.y * h), min(int((cam.y + cam.h) * h) + 1, h)
        energy[y0:y1, x0:x1] = 0.0
    total = float(energy.sum())
    if total <= 1e-6:
        return None
    cols = energy.sum(axis=0)
    centroid = float((cols * np.arange(w)).sum() / total) / max(w - 1, 1)
    return min(max(centroid, 0.32), 0.68)

"""Lightweight IoU-based face tracker for stable reframe tracking.

ByteTrack is the 2026 standard for multi-object tracking, but its
implementation pulls in several heavy dependencies (boxmot, supervision) that
are overkill for ClipForge's use case — we track at ~3 fps with at most 2-3
faces visible. A simple centroid-based tracker with IoU matching achieves the
same result (stable face IDs across frames) without any extra deps.

The tracker assigns each detection a stable ID across frames. When the same
face moves between frames, its box overlaps with the previous frame's box →
same ID. A new face entering the frame gets a new ID. A face leaving the frame
is removed after 5 frames of absence.

This is used by reframe._track_faces to replace per-frame face selection
with per-track selection — the camera stays on the same speaker ID instead
of jumping between faces every time the motion score flips.
"""
from __future__ import annotations


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union for two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    yi = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = xi * yi
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTrack:
    """One tracked face with stable ID and motion history."""
    __slots__ = ("id", "box", "cx", "age", "missed")

    def __init__(self, track_id: int, box: tuple[float, float, float, float]):
        self.id = track_id
        self.box = box
        self.cx = box[0] + box[2] / 2.0  # centre-x fraction
        self.age = 0
        self.missed = 0


class FaceTracker:
    """Tracks faces across frames using IoU matching.

    Usage:
        tracker = FaceTracker()
        for frame in frames:
            faces = detect_faces(frame)
            tracker.update(faces)  # assigns stable IDs
            # faces now have .id attributes set by tracker
    """
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 5):
        self._next_id = 1
        self._tracks: dict[int, FaceTrack] = {}
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed

    def update(self, detections: list[tuple[float, float, float, float]]
               ) -> dict[tuple[float, float, float, float], int]:
        """Match ``detections`` against existing tracks and return ``{box: id}``.

        Each detection is ``(x, y, w, h)`` — the raw face box format from
        ``detect_faces``. Returns a mapping from box tuple to stable track ID.
        This ID is then used by the reframe to keep the camera on the same
        speaker instead of switching every frame.
        """
        result: dict[tuple[float, float, float, float], int] = {}

        for box in detections:
            best_iou, best_tid = 0.0, None
            for tid, track in self._tracks.items():
                iou = _iou(box, track.box)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_tid is not None and best_iou >= self._iou_threshold:
                # Update existing track
                track = self._tracks[best_tid]
                track.box = box
                track.cx = box[0] + box[2] / 2.0
                track.age += 1
                track.missed = 0
                result[box] = track.id
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = FaceTrack(tid, box)
                result[box] = tid

        # Mark unmatched tracks as missed; remove stale ones.
        tracked_boxes = {tuple(round(v, 4) for v in t.box): t
                         for t in self._tracks.values()}
        for box, t in list(tracked_boxes.items()):
            if box not in result:
                t.missed += 1
                if t.missed > self._max_missed:
                    del self._tracks[t.id]

        return result

    def tracks_for(self, cx: float, *, radius: float = 0.06) -> int | None:
        """Return the ID of the track closest to ``cx`` (centre-x fraction).

        Used by the reframe to check: "is the face I was following still
        here?" before re-aiming.
        """
        best_id, best_d = None, radius
        for t in self._tracks.values():
            d = abs(t.cx - cx)
            if d < best_d:
                best_d = d
                best_id = t.id
        return best_id

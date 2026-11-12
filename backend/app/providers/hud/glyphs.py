"""Turn scoreboard pixels into comparable glyph bitmaps.

The score digits of a game HUD are a fixed font on a flat background, so the
same digit always produces (almost) the same pixels. Instead of running OCR on
every frame we binarise each score cell into a small normalised bitmap, group
identical bitmaps into clusters, and read each *cluster* once. On a four-hour
recording that is a few dozen OCR calls instead of several thousand, and every
frame showing the same digit is guaranteed the same answer.

Everything here is NumPy + OpenCV and free of OCR dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CANVAS_H = 32          # normalised glyph bitmap height
CANVAS_W = 48          # ... and width (room for two digits, e.g. "10")
WORK_H = 66            # cells are resampled to this height before thresholding
MIN_GLYPH_CONTRAST = 40.0
# Two bitmaps of the same digit overlap >= 0.87 (5th percentile, measured);
# different digits never exceeded 0.82 (6 vs 8). 0.88 keeps them apart; the few
# same-digit outliers below it just open a second cluster for that digit.
CLUSTER_MATCH = 0.88
# Looser bar used only to classify a glyph against already-labelled clusters.
CLASSIFY_MATCH = 0.80
CLASSIFY_MARGIN = 0.04


def crop(gray, box: tuple[int, int, int, int]):
    """Cut ``(x, y, w, h)`` out of a 2-D image (clipped to its bounds)."""
    x, y, w, h = box
    x0, y0 = max(x, 0), max(y, 0)
    return gray[y0:max(y + h, y0), x0:max(x + w, x0)]


def structure_levels(gray, rows_box: tuple[int, int, int, int],
                     clock_box: tuple[int, int, int, int]) -> tuple[float, float, float]:
    """Robust luma levels of the dark rows block and the light clock strip.

    Returns ``(rows_background, strip_background, strip_text)``. A real
    scoreboard has a dark rows block sitting on a clearly lighter clock strip
    that itself carries dark text; menus, replays and cutscenes do not.

    The strip background is its *median*: the clock's ink covers well under
    half of the strip, so the median is the light background. A menu caption
    in the same place is the opposite polarity (light text on a dark page) and
    its median stays dark, however bold the caption.
    """
    import numpy as np

    rows = crop(gray, rows_box)
    strip = crop(gray, clock_box)
    if rows.size == 0 or strip.size == 0:
        return 0.0, 0.0, 0.0
    return (float(np.percentile(rows, 30)), float(np.percentile(strip, 50)),
            float(np.percentile(strip, 4)))


def structure_ok(levels: tuple[float, float, float], min_contrast: float = 45.0) -> bool:
    rows_bg, strip_bg, strip_text = levels
    return (strip_bg - rows_bg) >= min_contrast and (strip_bg - strip_text) >= min_contrast


def extract_glyph(cell):
    """Binarise one score cell (light digit on a darker background).

    Returns a ``CANVAS_H x CANVAS_W`` uint8 bitmap (0/1) with the glyph scaled
    to full height and centred, or ``None`` when the cell does not hold a clean
    glyph (wrong contrast, too much or too little ink, ink touching two
    borders — i.e. scene content rather than a HUD digit).
    """
    import cv2
    import numpy as np

    if cell is None or cell.size == 0 or cell.shape[0] < 4 or cell.shape[1] < 4:
        return None
    h, w = cell.shape[:2]
    scale = WORK_H / float(h)
    g = cv2.resize(cell, (max(int(round(w * scale)), 8), WORK_H), interpolation=cv2.INTER_CUBIC)
    _thr, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fg = mask > 0
    frac = float(fg.mean())
    if frac < 0.03 or frac > 0.60:
        return None
    if float(g[fg].mean()) - float(g[~fg].mean()) < MIN_GLYPH_CONTRAST:
        return None
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big_h, big_w = g.shape
    keep = [i for i in range(1, n)
            if stats[i, cv2.CC_STAT_HEIGHT] >= 0.40 * big_h
            and stats[i, cv2.CC_STAT_AREA] >= 0.015 * big_h * big_w]
    if not keep or len(keep) > 2:
        return None
    x0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in keep)
    y0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in keep)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in keep)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in keep)
    touches = int(x0 <= 0) + int(y0 <= 0) + int(x1 >= big_w) + int(y1 >= big_h)
    if touches >= 2:
        return None
    sub = (np.isin(labels[y0:y1, x0:x1], keep)).astype(np.uint8) * 255
    gh, gw = sub.shape
    nw = max(4, min(CANVAS_W, int(round(gw * CANVAS_H / float(gh)))))
    scaled = cv2.resize(sub, (nw, CANVAS_H), interpolation=cv2.INTER_AREA) > 127
    canvas = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    off = (CANVAS_W - nw) // 2
    canvas[:, off:off + nw] = scaled
    return canvas


def glyph_iou(a, b) -> float:
    """Intersection-over-union of two glyph bitmaps."""
    import numpy as np

    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def ocr_image(bitmap, *, height: int = 56, pad: int = 12):
    """A glyph bitmap as an OCR-friendly image: black on white, tightly
    cropped, smooth edges, a modest border.

    The tight crop matters: with the glyph floating in a wide canvas EasyOCR
    read every bold "1" as "71"; cropped, it read all digits correctly.
    """
    import cv2
    import numpy as np

    cols = np.flatnonzero(bitmap.any(axis=0))
    if cols.size:
        bitmap = bitmap[:, int(cols.min()):int(cols.max()) + 1]
    scale = height / float(bitmap.shape[0])
    big = cv2.resize(bitmap.astype(np.uint8) * 255, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (3, 3), 0)
    return cv2.copyMakeBorder(255 - big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def line_image(strip, *, target_h: int = 84, pad: int = 12):
    """A text strip (e.g. the match clock) upscaled for line OCR."""
    import cv2

    if strip is None or strip.size == 0:
        return None
    scale = max(target_h / float(strip.shape[0]), 1.0)
    big = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def signature(cell, *, size: tuple[int, int] = (36, 12)):
    """Coarse binary fingerprint of a text cell (used to tell team names apart
    without reading them). ``size`` is ``(width, height)``. ``None`` when the
    cell has no usable contrast."""
    import cv2

    if cell is None or cell.size == 0:
        return None
    if float(cell.max()) - float(cell.min()) < MIN_GLYPH_CONTRAST:
        return None
    _thr, mask = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return (cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 127).astype("uint8")


@dataclass
class Cluster:
    """A group of (near-)identical glyph bitmaps — one visual form of a digit."""
    acc: object                       # float32 running sum of member bitmaps
    n: int = 0
    members: list[tuple[int, str]] = field(default_factory=list)   # (sample index, side)
    label: int | None = None
    anchor: bool = False              # label fixed by confident OCR
    source: str = ""                  # "ocr" | "rule" | "mixed"
    votes: dict[int, float] = field(default_factory=dict)
    empty_votes: int = 0

    def proto(self):
        """Per-pixel majority bitmap of the members."""
        return (self.acc * 2.0 >= float(self.n)).astype("uint8")


class GlyphClusters:
    """Greedy clustering of glyph bitmaps by IoU against cluster prototypes."""

    def __init__(self, match: float = CLUSTER_MATCH):
        self.match = match
        self.clusters: list[Cluster] = []

    def add(self, bitmap, ref: tuple[int, str]) -> int:
        """Assign a bitmap to a cluster (creating one if needed); returns its id."""
        import numpy as np

        idx, sim = -1, 0.0
        for i, c in enumerate(self.clusters):
            s = glyph_iou(bitmap, c.proto())
            if s > sim:
                idx, sim = i, s
        if idx < 0 or sim < self.match:
            self.clusters.append(Cluster(acc=np.zeros(bitmap.shape, np.float32)))
            idx = len(self.clusters) - 1
        c = self.clusters[idx]
        c.acc += bitmap
        c.n += 1
        c.members.append(ref)
        return idx

    def classify(self, bitmap) -> int | None:
        """Digit value for a bitmap using the labelled clusters, else ``None``.

        The best-matching labelled cluster must clear ``CLASSIFY_MATCH`` and
        beat every cluster carrying a *different* digit by ``CLASSIFY_MARGIN``.
        """
        sims = [(glyph_iou(bitmap, c.proto()), c.label)
                for c in self.clusters if c.label is not None]
        if not sims:
            return None
        best_s, best_label = max(sims, key=lambda p: p[0])
        other_s = max((s for s, lab in sims if lab != best_label), default=0.0)
        if best_s < CLASSIFY_MATCH or (best_s - other_s) < CLASSIFY_MARGIN:
            return None
        return best_label

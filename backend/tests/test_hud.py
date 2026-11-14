"""Tests for the scoreboard tracker (``app.providers.hud``).

The package reads a game's score off the screen, so the tests need pixels. A
small synthetic renderer below draws just the scanned region the way the real
HUD looks (dark rows block, light digits, a lighter clock strip with dark
text, team names) and a scripted timeline turns it into a fake frame source,
so ``track_scoreboard`` runs end to end with no ffmpeg and no OCR engine.

    cd backend && python -m pytest tests/test_hud.py -q
"""
from __future__ import annotations

import itertools
import os
import subprocess
import tempfile
import types
from dataclasses import dataclass, field, replace
from pathlib import Path

# Keep the real data directory out of reach before anything imports settings.
os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.providers import hud  # noqa: E402
from app.providers.hud import detect as D  # noqa: E402
from app.providers.hud import glyphs as G  # noqa: E402
from app.providers.hud import labeling as L  # noqa: E402
from app.providers.hud import readers as R  # noqa: E402
from app.providers.hud import scan as S  # noqa: E402
from app.providers.hud import tracker as T  # noqa: E402
from app.providers.hud.layouts import EAFC_STACKED, LAYOUTS, Cell, HudLayout, layouts_for  # noqa: E402

LAY = EAFC_STACKED
RESOLUTIONS = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]


# --------------------------------------------------------------------------- #
# Synthetic scoreboard renderer
# --------------------------------------------------------------------------- #
INK = 238.0            # score digits and team names
ROWS_BG = 26.0         # the dark block behind them
STRIP_BG = 222.0       # the clock strip ...
STRIP_INK = 22.0       # ... and its text
SCENE = 96.0           # the game behind a (faded) scoreboard
MENU_BG = 20.0
# Digits fill three quarters of the score cell. At two thirds a 720p digit is
# ten pixels tall and its bitmap drifts just below CLUSTER_MATCH against the
# 1440p rendering of the same digit (see the cross-resolution test).
DIGIT_FILL = 0.76
NAME_FILL = 0.50
CLOCK_FILL = 0.72

_FONT = cv2.FONT_HERSHEY_DUPLEX
_CAP = 160             # cap height of the master glyphs, in pixels
_THICK = 12            # stroke width at that size: bold, like a real scoreboard font
_GLYPHS: dict[str, np.ndarray] = {}
_MASTERS: dict[str, np.ndarray] = {}
_METRICS: dict[str, float] = {}


def _metrics() -> dict[str, float]:
    """Font scale and ink rows for a ``_CAP`` pixel tall line of text."""
    if not _METRICS:
        (_tw, th), _base = cv2.getTextSize("0123456789", _FONT, 1.0, _THICK)
        scale = _CAP / float(th)
        (tw, th), base = cv2.getTextSize("0123456789", _FONT, scale, _THICK)
        org_y = th + base + 40
        probe = np.zeros((org_y + 2 * base + 60, tw + 120), np.uint8)
        cv2.putText(probe, "0123456789", (60, org_y), _FONT, scale, 255, _THICK, cv2.LINE_AA)
        rows = np.flatnonzero((probe > 127).any(axis=1))
        _METRICS.update(scale=scale, org_y=org_y, top=int(rows.min()), bottom=int(rows.max()) + 1,
                        canvas_h=probe.shape[0])
    return _METRICS


def _glyph(ch: str) -> np.ndarray:
    """One character on the common baseline, cropped to its ink columns.

    A zero is drawn with the letter O: OpenCV's zero carries a slash that
    fills its counter at HUD sizes and makes it look like an 8 (IoU 0.86,
    against 0.81 for the closest pair of real HUD digits). Scoreboard fonts
    have an open zero.
    """
    if ch not in _GLYPHS:
        m = _metrics()
        drawn = "O" if ch == "0" else ch
        (tw, _th), _base = cv2.getTextSize(drawn, _FONT, m["scale"], _THICK)
        img = np.zeros((int(m["canvas_h"]), tw + 160), np.uint8)
        cv2.putText(img, drawn, (80, int(m["org_y"])), _FONT, m["scale"], 255, _THICK, cv2.LINE_AA)
        img = img[int(m["top"]):int(m["bottom"])]
        cols = np.flatnonzero((img > 127).any(axis=0))
        if cols.size:
            img = img[:, int(cols.min()):int(cols.max()) + 1]
        else:                                   # a space
            img = np.zeros((img.shape[0], _CAP // 3), np.uint8)
        _GLYPHS[ch] = img
    return _GLYPHS[ch]


def _master(text: str) -> np.ndarray:
    """Anti-aliased ``text``, ``_CAP`` pixels tall, cropped to its ink."""
    if text not in _MASTERS:
        gap = np.zeros((_glyph("0").shape[0], _CAP // 9), np.uint8)
        parts: list[np.ndarray] = []
        for ch in text:
            if parts:
                parts.append(gap)
            parts.append(_glyph(ch))
        _MASTERS[text] = np.hstack(parts) if parts else gap[:, :0]
    return _MASTERS[text]


def text_mask(text: str, w: int, h: int, *, fill: float, max_w: float = 0.9) -> np.ndarray:
    """Coverage (0..1) of ``text`` centred in a ``w`` x ``h`` cell.

    The text is ``fill`` of the cell height (less when it would not fit the
    width) and is resampled from a large master the way a video scaler would
    do it, so the same text looks the same at every resolution.
    """
    out = np.zeros((h, w), np.float32)
    m = _master(text)
    if m.shape[1] == 0:
        return out
    gh = max(int(round(fill * h)), 3)
    gw = max(int(round(gh * m.shape[1] / float(m.shape[0]))), 2)
    if gw > max_w * w:
        gw = max(int(max_w * w), 2)
        gh = max(int(round(gw * m.shape[0] / float(m.shape[1]))), 3)
    small = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x0, y0 = (w - gw) // 2, (h - gh) // 2
    out[y0:y0 + gh, x0:x0 + gw] = small
    return out


def digit_cell(value, w: int, h: int, *, fg: float = INK, bg: float = ROWS_BG,
               fill: float = DIGIT_FILL) -> np.ndarray:
    """A score cell on its own: light digit(s) on the dark rows block."""
    m = text_mask(str(value), w, h, fill=fill)
    return np.clip(bg + (fg - bg) * m, 0, 255).astype(np.uint8)


def score_cell_size(height: int, lay: HudLayout = LAY) -> tuple[int, int]:
    _x, _y, w, h = lay.score_a.px(height)
    return w, h


def digit_bitmap(value, height: int = 1080, **kw) -> np.ndarray:
    """The glyph bitmap the tracker extracts for a rendered score."""
    bm = G.extract_glyph(digit_cell(value, *score_cell_size(height), **kw))
    assert bm is not None, f"renderer produced an unreadable {value!r} at {height}p"
    return bm


@dataclass(frozen=True)
class HudState:
    """What the scanned region shows at one moment."""
    a: int = 0
    b: int = 0
    clock: str = "12:29"
    name_a: str = "RMA"
    name_b: str = "FCB"
    visible: bool = True
    fade: float = 1.0          # 1 = solid scoreboard, lower = blended into the game
    off: str = "menu"          # what is on screen while the scoreboard is not: menu | noise
    seed: int = 0              # varies the noise frame


def _paint(img: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray, ink: float) -> None:
    x, y, w, h = box
    part = img[y:y + h, x:x + w]
    part[:] = part * (1.0 - mask) + ink * mask


def render_region(lay: HudLayout, width: int, height: int, st: HudState) -> np.ndarray:
    """The gray pixels of ``lay.region_px(width, height)`` for a HUD state."""
    region = lay.region_px(width, height)
    _rx, _ry, rw, rh = region
    if not st.visible:
        return _render_off(rw, rh, st)
    box = lambda cell: lay.cell_in_region(cell, height, region)  # noqa: E731
    img = np.full((rh, rw), SCENE, np.float32)
    x, y, w, h = box(lay.rows)
    sx, _sy, sw, _sh = box(lay.score_b)
    img[y:y + h, x:sx + sw] = ROWS_BG                 # names and scores share the dark block
    div = (x + w + sx) // 2                           # the divider between them
    img[y:y + h, div:div + max(height // 720, 1)] = 120.0
    cx, cy, cw, ch = box(lay.clock)
    img[cy:cy + ch, cx:cx + cw] = STRIP_BG
    _paint(img, (cx, cy, cw, ch), text_mask(st.clock, cw, ch, fill=CLOCK_FILL), STRIP_INK)
    for cell, value in ((lay.score_a, st.a), (lay.score_b, st.b)):
        bx = box(cell)
        _paint(img, bx, text_mask(str(value), bx[2], bx[3], fill=DIGIT_FILL), INK)
    for cell, name in ((lay.name_a, st.name_a), (lay.name_b, st.name_b)):
        if cell is not None and name:
            bx = box(cell)
            _paint(img, bx, text_mask(name, bx[2], bx[3], fill=NAME_FILL), INK)
    if st.fade < 1.0:
        img = SCENE * (1.0 - st.fade) + img * st.fade
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _render_off(rw: int, rh: int, st: HudState) -> np.ndarray:
    if st.off == "noise":
        rng = np.random.default_rng(1000 + st.seed)
        return np.clip(rng.normal(SCENE, 14.0, (rh, rw)), 0, 255).astype(np.uint8)
    # A menu: dark page, light captions that happen to sit where the HUD would be.
    img = np.full((rh, rw), MENU_BG, np.float32)
    half = rh // 2
    _paint(img, (0, 0, rw, half), text_mask("SQUAD", rw, half, fill=0.5), 230.0)
    _paint(img, (0, half, rw, rh - half), text_mask("PLAY 12:29", rw, rh - half, fill=0.4), 230.0)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Scripted timeline and fake frame source
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScriptGoal:
    t: float                                   # when the new digit is (or would be) on screen
    team: str
    hide: tuple[float, float] | None = None    # celebration / replay: scoreboard off screen


@dataclass(frozen=True)
class ScriptMatch:
    start: float
    end: float
    name_a: str
    name_b: str
    goals: tuple[ScriptGoal, ...] = ()
    clock_rate: float = 28.0                   # match seconds per second of video
    hides: tuple[tuple[float, float], ...] = ()    # replays without a goal

    def score_at(self, t: float) -> tuple[int, int]:
        return (sum(1 for g in self.goals if g.team == "a" and g.t <= t),
                sum(1 for g in self.goals if g.team == "b" and g.t <= t))

    def clock_at(self, t: float) -> str:
        s = int(max(t - self.start, 0.0) * self.clock_rate)
        return "%02d:%02d" % (s // 60, s % 60)

    def minute_at(self, t: float) -> int:
        return int(max(t - self.start, 0.0) * self.clock_rate) // 60 + 1

    def hidden_at(self, t: float) -> bool:
        spans = [g.hide for g in self.goals if g.hide] + list(self.hides)
        return any(a <= t < b for a, b in spans)

    def goal_moment(self, g: ScriptGoal, lay: HudLayout = LAY) -> float:
        """When the ball crossed the line, per the layout's measured delays."""
        if g.hide:
            return g.hide[0] - lay.cut_delay_s
        return g.t - lay.score_delay_s


@dataclass
class Timeline:
    duration: float
    matches: list[ScriptMatch]
    off: str = "menu"
    # (t0, t1, (a, b)): the scoreboard shows this score instead for t0 <= t < t1.
    glitches: list[tuple[float, float, tuple[int, int]]] = field(default_factory=list)
    fades: list[tuple[float, float, float]] = field(default_factory=list)   # (t0, t1, fade)

    def match_at(self, t: float) -> ScriptMatch | None:
        for m in self.matches:
            if m.start <= t < m.end:
                return m
        return None

    def state(self, t: float) -> HudState:
        m = self.match_at(t)
        if m is None or m.hidden_at(t):
            return HudState(visible=False, off=self.off,
                            seed=int(round(t * 10)) if self.off == "noise" else 0)
        a, b = m.score_at(t)
        for t0, t1, score in self.glitches:
            if t0 <= t < t1:
                a, b = score
        fade = next((f for t0, t1, f in self.fades if t0 <= t < t1), 1.0)
        return HudState(a=a, b=b, clock=m.clock_at(t), name_a=m.name_a, name_b=m.name_b, fade=fade)


class FakeSource:
    """Stands in for ``scan.iter_frames``: frames rendered from a timeline.

    The coarse pass gets one frame per ``keyint`` seconds, a dense window the
    requested fps between ``start`` and ``start + duration``.
    """

    def __init__(self, timeline: Timeline, *, width: int = 1920, height: int = 1080,
                 keyint: float = 2.0, lay: HudLayout = LAY, fail_dense: bool = False):
        self.timeline, self.width, self.height = timeline, width, height
        self.keyint, self.lay, self.fail_dense = keyint, lay, fail_dense
        self.region = lay.region_px(width, height)
        self.calls: list[dict] = []
        self.clock_text: dict[bytes, str] = {}       # upscaled clock strip -> its text
        self._frames: dict[HudState, np.ndarray] = {}
        self._clock_box = lay.cell_in_region(lay.clock, height, self.region)

    @property
    def info(self):
        return types.SimpleNamespace(width=self.width, height=self.height,
                                     duration=self.timeline.duration)

    def frame(self, t: float) -> np.ndarray:
        st = self.timeline.state(t)
        img = self._frames.get(st)
        if img is None:
            img = render_region(self.lay, self.width, self.height, st)
            img.flags.writeable = False              # like np.frombuffer in the real scanner
            self._frames[st] = img
            if st.visible:
                strip = G.line_image(G.crop(img, self._clock_box))
                self.clock_text[strip.tobytes()] = st.clock
        return img

    def times(self, *, keyframes: bool, fps: float | None, start: float | None,
              duration: float | None) -> list[float]:
        total = self.timeline.duration
        if keyframes:
            return [k * self.keyint for k in range(int(total / self.keyint) + 1)
                    if k * self.keyint < total]
        t0 = float(start or 0.0)
        t1 = min(t0 + duration, total) if duration is not None else total
        out, k = [], 0
        while t0 + k / float(fps) < t1 - 1e-9:
            out.append(t0 + k / float(fps))
            k += 1
        return out

    def __call__(self, region, *, keyframes: bool = True, fps: float | None = None,
                 start: float | None = None, duration: float | None = None,
                 hwaccel: bool = False):
        assert tuple(region) == self.region
        self.calls.append({"keyframes": keyframes, "fps": fps, "start": start,
                           "duration": duration, "hwaccel": hwaccel})
        if self.fail_dense and not keyframes and start is not None:
            raise S.HudScanError("scripted dense-read failure")
        for t in self.times(keyframes=keyframes, fps=fps, start=start, duration=duration):
            yield t, self.frame(t)


class TemplateDigitReader:
    """A fake OCR engine for score digits: nearest rendered template wins.

    ``misread`` maps a true value to what the engine claims instead, ``blind``
    lists values it returns nothing for, ``conf`` is the confidence it reports
    (a number, or a dict per true value).
    """
    name = "fake-digits"

    def __init__(self, height: int = 1080, *, values=range(0, 13), misread: dict | None = None,
                 blind=(), conf=0.93):
        self.templates = [(v, G.ocr_image(digit_bitmap(v, height))) for v in values]
        self.misread, self.blind, self.conf = dict(misread or {}), set(blind), conf
        self.calls = 0

    def recognise(self, img) -> int | None:
        best, best_d = None, 60.0
        for value, tpl in self.templates:
            if tpl.shape != img.shape:
                tpl = cv2.resize(tpl, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            d = float(np.abs(tpl.astype(np.float32) - img.astype(np.float32)).mean())
            if d < best_d:
                best, best_d = value, d
        return best

    def read_digits(self, img) -> tuple[str, float]:
        self.calls += 1
        value = self.recognise(img)
        if value is None or value in self.blind:
            return "", 0.0
        conf = self.conf.get(value, 0.93) if isinstance(self.conf, dict) else self.conf
        return str(self.misread.get(value, value)), float(conf)

    def read_line(self, img, charset: str) -> tuple[str, float]:
        return "", 0.0


class ClockReader:
    """A fake line reader that knows the clock text of every frame a
    ``FakeSource`` rendered. ``garble`` mimics an engine's habits."""
    name = "fake-clock"

    def __init__(self, source: FakeSource, *, garble=lambda text: text):
        self.source, self.garble = source, garble
        self.calls = 0

    def read_line(self, img, charset: str) -> tuple[str, float]:
        self.calls += 1
        assert charset == R.CLOCK_CHARS
        text = self.source.clock_text.get(img.tobytes())
        return (self.garble(text), 0.9) if text is not None else ("", 0.0)

    def read_digits(self, img) -> tuple[str, float]:
        return "", 0.0


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #
def test_region_is_even_inside_the_frame_and_holds_every_cell():
    for w, h in RESOLUTIONS:
        region = LAY.region_px(w, h)
        x, y, rw, rh = region
        assert x % 2 == 0 and y % 2 == 0 and rw % 2 == 0 and rh % 2 == 0
        assert x >= 0 and y >= 0 and x + rw <= w and y + rh <= h
        for cell in LAY.cells():
            cx, cy, cw, ch = LAY.cell_in_region(cell, h, region)
            assert cx >= 0 and cy >= 0 and cx + cw <= rw and cy + ch <= rh


def test_cell_pixels_scale_with_frame_height():
    assert Cell(0.5, 0.25, 0.1, 0.05).px(1000) == (500, 250, 100, 50)
    x, y, w, h = LAY.score_a.px(1440)
    assert (x, y, w, h) == (340, 79, 48, 33)          # the numbers the preset was measured from


def test_layouts_for_profiles():
    assert layouts_for("eafc") == [EAFC_STACKED]
    assert layouts_for("FIFA") == [EAFC_STACKED]
    assert layouts_for("EA FC") == [EAFC_STACKED]
    assert layouts_for("auto") == list(LAYOUTS.values()) == layouts_for(None)
    assert layouts_for("valorant") == list(LAYOUTS.values())
    assert hud.layouts_for("generic") == list(LAYOUTS.values())


# --------------------------------------------------------------------------- #
# Glyphs
# --------------------------------------------------------------------------- #
def test_extract_glyph_reads_a_digit_and_rejects_everything_else():
    w, h = score_cell_size(1080)
    bm = G.extract_glyph(digit_cell(7, w, h))
    assert bm is not None and bm.shape == (G.CANVAS_H, G.CANVAS_W)
    assert set(np.unique(bm).tolist()) <= {0, 1}
    assert bm.any(axis=1).all()                        # scaled to the full canvas height
    assert G.extract_glyph(None) is None
    assert G.extract_glyph(np.full((h, w), 40, np.uint8)) is None             # flat
    assert G.extract_glyph(digit_cell(7, w, h, fg=58, bg=40)) is None          # too little contrast
    inked = np.full((h, w), 230, np.uint8)
    inked[:3] = 20
    assert G.extract_glyph(inked) is None                                     # mostly ink
    bar = np.full((h, w), 30, np.uint8)
    bar[:, w // 2 - 3:w // 2 + 3] = 230
    assert G.extract_glyph(bar) is None                                       # touches two borders


def test_same_digit_one_cluster_different_digits_apart():
    book = G.GlyphClusters()
    ids: dict[int, set[int]] = {}
    for v in range(10):
        for rep in range(3):
            ids.setdefault(v, set()).add(book.add(digit_bitmap(v, 1440), (rep, "a")))
    assert all(len(s) == 1 for s in ids.values())
    assert len({next(iter(s)) for s in ids.values()}) == 10
    worst = max(G.glyph_iou(digit_bitmap(a, 1440), digit_bitmap(b, 1440))
                for a, b in itertools.combinations(range(10), 2))
    assert worst < G.CLUSTER_MATCH
    assert G.glyph_iou(digit_bitmap(4, 1440), digit_bitmap(4, 1440)) == 1.0
    assert book.add(digit_bitmap(10, 1440), (0, "a")) == 10                    # two digits: its own cluster


def test_classify_needs_a_label_a_close_match_and_a_margin():
    book = G.GlyphClusters()
    three = digit_bitmap(3, 1440)
    assert book.classify(three) is None                                       # nothing labelled yet
    for v in (1, 3):
        book.clusters[book.add(digit_bitmap(v, 1440), (0, "a"))].label = v
    assert book.classify(three) == 3
    assert book.classify(digit_bitmap(7, 1440)) is None                        # far from every cluster
    book.clusters.append(G.Cluster(acc=three.astype(np.float32), n=1, label=8))
    assert book.classify(three) is None                                       # two digits fit equally


def test_ocr_image_is_tight_dark_on_light():
    img = G.ocr_image(digit_bitmap(1, 1440))
    assert img.dtype == np.uint8 and img[0, 0] == 255 and img.min() < 60
    cols = np.flatnonzero((img < 128).any(axis=0))
    assert cols.min() <= 14 and img.shape[1] - 1 - cols.max() <= 14            # only the 12 px border
    assert G.line_image(None) is None and G.signature(np.full((8, 20), 90, np.uint8)) is None


def test_structure_test_separates_a_scoreboard_from_menus():
    for w, h in RESOLUTIONS:
        region = LAY.region_px(w, h)
        rows = LAY.cell_in_region(LAY.rows, h, region)
        clock = LAY.cell_in_region(LAY.clock, h, region)

        def ok(st: HudState) -> bool:
            levels = G.structure_levels(render_region(LAY, w, h, st), rows, clock)
            return G.structure_ok(levels, LAY.min_structure_contrast)

        assert ok(HudState(a=2, b=1))
        assert ok(HudState(a=2, b=1, fade=0.6))
        assert not ok(HudState(visible=False))
        assert not ok(HudState(visible=False, off="noise"))
    assert not G.structure_ok((200.0, 210.0, 190.0))


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def test_clean_digits():
    assert R.clean_digits("7") == 7 and R.clean_digits(" 10 ") == 10
    for bad in ("", "O", "123", "1a", None):
        assert R.clean_digits(bad) is None


def test_parse_clock_variants():
    assert R.parse_clock("12:29") == (12, 29, 0)
    assert R.parse_clock("90:56+2") == (90, 56, 2)
    assert R.parse_clock("1229") == (12, 29, 0)            # colon dropped
    assert R.parse_clock("12829") == (12, 29, 0)           # colon read as a digit
    assert R.parse_clock("90856++2") == (90, 56, 2)
    assert R.parse_clock("105:30") == (105, 30, 0)
    assert R.parse_clock("105830") == (105, 30, 0)
    for bad in ("", "abc", "12:75", "999:10", "12", "1:2:3", None):
        assert R.parse_clock(bad) is None
    assert R.match_minute((12, 29, 0)) == 13 and R.match_minute(None) is None


def test_parse_tsv_joins_words_and_scales_confidence():
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
           "1\t1\t0\t0\t0\t0\t0\t0\t100\t50\t-1\t\n"
           "5\t1\t1\t1\t1\t1\t10\t10\t30\t30\t96.5\t12:29\n"
           "5\t1\t1\t1\t1\t2\t50\t10\t20\t30\t80\t+2\n")
    assert R.parse_tsv(tsv) == ("12:29+2", 0.965)
    assert R.parse_tsv("") == ("", 0.0)


def test_tessdata_falls_back_when_the_prefix_is_empty(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "tesseract.exe"
    (exe.parent / "tessdata").mkdir(parents=True)
    exe.write_bytes(b"")
    (exe.parent / "tessdata" / "eng.traineddata").write_bytes(b"x")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("TESSDATA_PREFIX", str(empty))
    assert Path(R._tessdata_dir(str(exe))) == exe.parent / "tessdata"
    monkeypatch.setenv("TESSDATA_PREFIX", str(exe.parent / "tessdata"))
    assert Path(R._tessdata_dir(str(exe))) == exe.parent / "tessdata"


def test_pick_readers_survives_missing_engines(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("not installed")

    monkeypatch.setattr(R, "RapidOcrReader", boom)
    monkeypatch.setattr(R, "EasyOcrReader", boom)
    monkeypatch.setattr(R, "TesseractCliReader", boom)
    assert R.pick_readers() == (None, None)


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #
class _ScriptedReader:
    """Answers per cluster id (``L.ocr_image`` is patched to pass the id through)."""
    name = "scripted"

    def __init__(self, answers: dict[int, tuple[str, float]]):
        self.answers = answers

    def read_digits(self, cid):
        return self.answers.get(cid, ("", 0.0))


def _label(monkeypatch, seq: list[int], answers: dict | None, *, dt: float = 4.0):
    """Run label_clusters on one cell's cluster sequence; returns (clusters, diag)."""
    monkeypatch.setattr(L, "ocr_image", lambda bm: bm)
    times = [i * dt for i in range(len(seq))]
    clusters = [G.Cluster(acc=None) for _ in range(max(seq) + 1)]
    for i, cid in enumerate(seq):
        if cid >= 0:
            clusters[cid].n += 1
            clusters[cid].members.append((i, "a"))
    reader = _ScriptedReader(answers) if answers is not None else None
    diag = L.label_clusters(clusters, {"a": seq, "b": [-1] * len(seq)}, times,
                            lambda ref: seq[ref[0]], reader)
    return clusters, diag


def test_cluster_runs_drop_blips_and_merge_neighbours():
    seq = [0, 0, -1, 0, 1, 0, 0, 2, 2]
    assert L.cluster_runs(seq, [float(i) for i in range(9)]) == [(0, 0.0, 6.0, 5), (2, 7.0, 8.0, 2)]


def test_transition_edges_are_one_way_and_respect_the_gap():
    t6 = [float(i) for i in range(6)]
    assert L.transition_edges({"a": [0, 0, 1, 1, 2, 2]}, t6) == {(0, 1): 1, (1, 2): 1}
    flicker = [0, 0, 1, 1, 0, 0, 1, 1]
    assert L.transition_edges({"a": flicker}, [float(i) for i in range(8)]) == {}
    assert L.transition_edges({"a": [0, 0, 1, 1]}, [0.0, 1.0, 200.0, 201.0]) == {}


def test_confident_ocr_anchors_every_cluster(monkeypatch):
    seq = [0] * 4 + [1] * 4 + [2] * 4
    clusters, diag = _label(monkeypatch, seq, {0: ("0", 0.9), 1: ("1", 0.9), 2: ("2", 0.9)})
    assert [c.label for c in clusters] == [0, 1, 2]
    assert all(c.anchor and c.source == "ocr" for c in clusters)
    assert diag["violations"] == 0 and diag["ocr_calls"] == 12 and not diag["anchored_by_rule"]


def test_the_rule_fills_digits_the_engine_cannot_read(monkeypatch):
    seq = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    clusters, _diag = _label(monkeypatch, seq, {1: ("1", 0.95)})
    assert [c.label for c in clusters] == [0, 1, 2, 3]
    assert [c.source for c in clusters] == ["rule", "ocr", "rule", "rule"]


def test_the_rule_overrides_a_weak_misread_but_not_an_anchor(monkeypatch):
    seq = ([0] * 3 + [1] * 3) * 3 + [2] * 3
    weak, _ = _label(monkeypatch, seq, {0: ("0", 0.9), 1: ("7", 0.2), 2: ("2", 0.9)})
    assert [c.label for c in weak] == [0, 1, 2] and weak[1].source == "mixed"
    strong, diag = _label(monkeypatch, seq, {0: ("0", 0.9), 1: ("7", 0.9), 2: ("2", 0.9)})
    assert [c.label for c in strong] == [0, 7, 2]          # a confident engine is believed ...
    assert diag["violations"] > 0                          # ... and the contradiction is reported


def test_without_ocr_the_zero_anchors_the_labels(monkeypatch):
    seq = [0] * 3 + [1] * 3 + [2] * 3 + [0] * 3 + [1] * 3
    times_gap = None  # the second match starts right away: a reset edge 2->0 exists
    clusters, diag = _label(monkeypatch, seq, None)
    assert times_gap is None and [c.label for c in clusters] == [0, 1, 2]
    assert diag["anchored_by_rule"] and diag["ocr_calls"] == 0 and diag["violations"] == 0


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
def _samples(spec: list, dt: float = 4.0) -> list[T.Sample]:
    """``(a, b)`` = readable, ``None`` = scoreboard absent, ``"x"`` = present but unreadable."""
    out = []
    for i, s in enumerate(spec):
        if s is None:
            out.append(T.Sample(t=i * dt, present=False))
        elif s == "x":
            out.append(T.Sample(t=i * dt, present=True))
        else:
            out.append(T.Sample(t=i * dt, present=True, a=s[0], b=s[1]))
    return out


def _goals(segments: list[T.Segment]) -> list[T.Goal]:
    return [g for seg in segments for g in seg.goals]


def test_track_a_plain_goal():
    segs = T.track(_samples([(0, 0)] * 3 + [(1, 0)] * 3))
    assert len(segs) == 1 and segs[0].start_score == (0, 0) and segs[0].final_score == (1, 0)
    (g,) = _goals(segs)
    assert (g.team, g.before, g.after, g.lo, g.hi, g.i_hi) == ("a", (0, 0), (1, 0), 8.0, 12.0, 3)
    assert not g.hidden and not g.tentative and not g.inferred


def test_track_goal_behind_a_celebration_and_the_limit_of_that():
    (g,) = _goals(T.track(_samples([(0, 0)] * 3 + [None] * 5 + [(0, 1)] * 3)))
    assert g.team == "b" and g.hidden and (g.lo, g.hi) == (8.0, 32.0)
    far = T.track(_samples([(0, 0)] * 3 + [None] * 30 + [(0, 1)] * 3))    # 124 s later: another match
    assert len(far) == 2 and _goals(far) == []


def test_track_reset_and_long_break_start_a_new_match():
    reset = T.track(_samples([(2, 1)] * 3 + [(0, 0)] * 3))
    assert [s.final_score for s in reset] == [(2, 1), (0, 0)] and _goals(reset) == []
    same = T.track(_samples([(1, 1)] * 3 + [None] * 70 + [(1, 1)] * 3))    # 284 s apart
    assert len(same) == 2 and _goals(same) == []


def test_track_ignores_one_sample_misreads():
    assert _goals(T.track(_samples([(1, 0)] * 3 + [(1, 5)] + [(1, 0)] * 3))) == []
    assert _goals(T.track(_samples([(1, 0)] * 3 + [(2, 0)] + [(1, 0)] * 3))) == []
    segs = T.track(_samples([(1, 0)] * 2 + ["x"] + [(1, 0)] * 2 + [(1, 1)] * 2))
    (g,) = _goals(segs)
    assert len(segs) == 1 and g.team == "b" and not g.hidden


def test_track_keeps_a_last_second_goal_as_tentative():
    (g,) = _goals(T.track(_samples([(1, 0)] * 3 + [(2, 0)] + [None] * 40)))
    assert g.tentative and g.after == (2, 0) and g.hi == 12.0
    (g,) = _goals(T.track(_samples([(1, 0)] * 3 + [(2, 0)] + [None] * 2 + [(2, 0)] * 3)))
    assert not g.tentative and g.hi == 12.0                 # confirmed once the score returns


def test_track_two_goals_between_two_readings():
    goals = _goals(T.track(_samples([(0, 0)] * 3 + [None] * 4 + [(1, 1)] * 3)))
    assert [(g.team, g.before, g.after, g.inferred) for g in goals] == [
        ("a", (0, 0), (1, 0), True), ("b", (1, 0), (1, 1), True)]
    assert all(g.hidden for g in goals)
    adjacent = T.track(_samples([(0, 0)] * 3 + [(2, 0)] * 3))               # no time for two goals
    assert len(adjacent) == 2 and _goals(adjacent) == []


def test_in_play_intervals_bridge_short_absences_only():
    spec = [(0, 0)] * 3 + [None] * 5 + [(0, 0)] * 2 + [None] * 20 + [(0, 0)] * 2
    spans = T.in_play_intervals(_samples(spec), bridge_s=45.0)
    assert spans == [(0.0, 36.0), (120.0, 124.0)]
    assert T.covered(spans, 20.0) and not T.covered(spans, 60.0) and T.covered(spans, 37.5, pad=2.0)
    assert T.step_team((1, 1), (2, 1)) == "a" and T.step_team((1, 1), (1, 2)) == "b"
    assert T.step_team((1, 1), (2, 2)) is None


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
def test_build_command_keyframe_and_fps_modes():
    kf = S.build_command("ffmpeg", "v.mp4", (206, 74, 190, 110), keyframes=True, fps=None, start=None,
                         duration=None, hwaccel=True, passthrough=("-fps_mode", "passthrough"))
    assert kf[kf.index("-discard:v") + 1] == "nokey" and kf[kf.index("-skip_frame") + 1] == "nokey"
    assert "-hwaccel" not in kf and "-ss" not in kf and "-t" not in kf
    assert kf[kf.index("-vf") + 1] == "crop=190:110:206:74:exact=1,showinfo"
    assert kf[kf.index("-pix_fmt") + 1] == "gray" and kf[-1] == "pipe:1"
    assert kf.index("-fps_mode") > kf.index("-i")
    dense = S.build_command("ffmpeg", "v.mp4", (0, 0, 64, 48), keyframes=False, fps=5.0, start=12.5,
                            duration=4.0, hwaccel=False, passthrough=("-vsync", "0"))
    assert "-discard:v" not in dense and "-skip_frame" not in dense and "-hwaccel" not in dense
    assert dense[dense.index("-ss") + 1] == "12.500" and dense.index("-ss") < dense.index("-i")
    assert dense[dense.index("-t") + 1] == "4.000"
    assert dense[dense.index("-vf") + 1] == "crop=64:48:0:0:exact=1,fps=5,showinfo"
    assert dense[dense.index("-vsync") + 1] == "0"


def test_iter_frames_on_a_real_video(tmp_path):
    from app.config import get_settings

    exe = get_settings().ffmpeg
    if not exe:
        pytest.skip("no ffmpeg binary in this environment")
    video = tmp_path / "v.mp4"
    made = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                           "testsrc=size=320x240:rate=30:duration=4", "-g", "30", "-pix_fmt", "yuv420p",
                           str(video)], capture_output=True, timeout=120)
    if made.returncode != 0 or not video.exists():
        pytest.skip("this ffmpeg build cannot encode the test video")
    region = (32, 16, 64, 48)
    key = list(S.iter_frames(str(video), region, keyframes=True))
    assert len(key) == 4 and all(f.shape == (48, 64) and f.dtype == np.uint8 for _t, f in key)
    # Keyframes at 0, 1, 2, 3 s. Keyframe mode is accurate to a few frame durations only.
    assert all(abs(t - k) < 0.15 for k, (t, _f) in enumerate(key))
    dense = [t for t, _f in S.iter_frames(str(video), region, keyframes=False, fps=5.0,
                                          start=1.0, duration=2.0)]
    assert 9 <= len(dense) <= 11 and dense == sorted(dense)
    assert dense[0] >= 0.99 and dense[-1] <= 3.21
    with pytest.raises(S.HudScanError):
        list(S.iter_frames(str(tmp_path / "missing.mp4"), region, keyframes=True))


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def _session() -> Timeline:
    """Two matches from 0-0, a menu in between, FCB (side b) in both."""
    first = ScriptMatch(start=10.0, end=130.0, name_a="RMA", name_b="FCB", goals=(
        ScriptGoal(31.3, "a"),
        ScriptGoal(55.1, "b"),
        ScriptGoal(80.0, "b", hide=(77.0, 84.0)),          # celebration hides the scoreboard
        ScriptGoal(129.1, "a"),                            # a second before the final whistle
    ))
    second = ScriptMatch(start=200.0, end=290.0, name_a="JUV", name_b="FCB", goals=(
        ScriptGoal(221.7, "b"), ScriptGoal(250.3, "a"), ScriptGoal(270.9, "b")))
    return Timeline(duration=300.0, matches=[first, second])


def _run(timeline: Timeline, readers=(None, None), **kw):
    opts = {k: kw.pop(k) for k in ("width", "height", "keyint", "fail_dense") if k in kw}
    src = FakeSource(timeline, **opts)
    kw.setdefault("refine_limit", None)
    return D.track_scoreboard("unused.mp4", src.info, None, layout=LAY, readers=readers,
                              frame_source=src, **kw), src


def test_session_end_to_end_with_the_rule_alone():
    timeline = _session()
    res, src = _run(timeline)
    assert res is not None and res.healthy and res.layout == LAY.name
    assert res.diagnostics["anchored_by_rule"] and res.diagnostics["violations"] == 0
    got = [(e.team, e.score_before, e.score_after) for e in res.events]
    assert got == [("a", (0, 0), (1, 0)), ("b", (1, 0), (1, 1)), ("b", (1, 1), (1, 2)),
                   ("a", (1, 2), (2, 2)),
                   ("b", (0, 0), (0, 1)), ("a", (0, 1), (1, 1)), ("b", (1, 1), (1, 2))]
    assert [(m.start_score, m.final_score, m.goals) for m in res.matches] == [
        ((0, 0), (2, 2), 4), ((0, 0), (1, 2), 3)]
    first, second = timeline.matches
    scripted = [first.goal_moment(g) for g in first.goals] + [second.goal_moment(g) for g in second.goals]
    for ev, want in zip(res.events, scripted):
        assert abs(ev.t - want) <= 0.3, (ev.t, want)
        assert ev.refined
    assert [e.hidden for e in res.events] == [False, False, True, False, False, False, False]
    assert res.streamer_side == "b"
    assert [e.streamer for e in res.events] == [False, True, True, False, True, False, True]
    assert len(res.in_play) == 2
    assert res.is_in_play(60.0) and res.is_in_play(80.0) and not res.is_in_play(165.0)
    assert 0.65 <= res.coverage <= 0.75
    assert any(c["keyframes"] for c in src.calls) and any(c["fps"] == D.DENSE_FPS for c in src.calls)


def test_session_at_other_resolutions_and_keyframe_spacing():
    for (w, h), keyint in (((1280, 720), 2.0), ((2560, 1440), 4.0)):
        res, _src = _run(_session(), width=w, height=h, keyint=keyint)
        assert res is not None and res.healthy, (w, h)
        assert [e.score_after for e in res.events] == [(1, 0), (1, 1), (1, 2), (2, 2),
                                                       (0, 1), (1, 1), (1, 2)], (w, h)


def test_ocr_readers_give_minutes_and_importance():
    timeline = _session()
    src = FakeSource(timeline)
    res = D.track_scoreboard("unused.mp4", src.info, None, layout=LAY, frame_source=src,
                             readers=(TemplateDigitReader(), ClockReader(src)), refine_limit=None)
    assert res is not None and res.healthy and not res.diagnostics["anchored_by_rule"]
    assert res.diagnostics["digit_reader"] == "fake-digits" and res.diagnostics["ocr_calls"] > 0
    first = timeline.matches[0]
    for ev, g in zip(res.events[:2], first.goals[:2]):
        assert ev.minute is not None and abs(ev.minute - first.minute_at(first.goal_moment(g))) <= 1
        assert ev.clock and ev.confidence == 0.95
    equaliser = res.events[1]                              # FCB make it 1-1
    assert equaliser.importance == 9 and equaliser.reasons == ["Equaliser", "Streamer's team scores"]
    garbled = D.track_scoreboard("unused.mp4", src.info, None, layout=LAY, frame_source=src,
                                 readers=(TemplateDigitReader(), ClockReader(src, garble=lambda t: t.replace(":", "8"))),
                                 refine_limit=None)
    assert [e.minute for e in garbled.events] == [e.minute for e in res.events]


def test_unrefined_goals_are_still_found():
    res, src = _run(_session(), refine_limit=0)
    assert len(res.events) == 7
    assert [e.refined for e in res.events] == [False, False, False, True, False, False, False]
    first = _session().matches[0]
    assert abs(res.events[0].t - first.goal_moment(first.goals[0])) <= 2.0    # coarse: half an interval
    # Only the two tail sweeps and the two kick-off reads; no goal timing.
    assert sum(1 for c in src.calls if not c["keyframes"]) == 4


def test_kickoff_is_the_first_frame_with_the_scoreboard():
    timeline = _session()
    res, src = _run(timeline, refine_limit=0)
    kickoffs = [m.kickoff for m in res.matches]
    for got, match in zip(kickoffs, timeline.matches):
        assert got is not None and 0.0 <= got - match.start <= 1.0 / D.DENSE_FPS + 1e-6
    assert all(m.play_start == m.kickoff <= m.t_start for m in res.matches)
    # The first coarse sample of a match can come up to a keyframe gap later.
    late, src = _run(timeline, refine_limit=0, keyint=4.0)
    assert [round(m.kickoff, 1) for m in late.matches] == [round(k, 1) for k in kickoffs]
    assert late.matches[0].t_start - late.matches[0].kickoff > 1.0


def test_a_glitch_before_the_whistle_is_not_a_goal():
    glitchy = Timeline(duration=120.0, matches=[
        ScriptMatch(start=10.0, end=60.0, name_a="RMA", name_b="FCB", goals=(ScriptGoal(30.3, "a"),))],
        glitches=[(58.0, 58.3, (1, 1))])                   # one coarse sample, two dense frames
    res, _src = _run(glitchy)
    assert [e.score_after for e in res.events] == [(1, 0)]
    assert res.diagnostics["tentative_dropped"] == 1
    kept, _src = _run(glitchy, fail_dense=True)            # cannot check: kept, but marked doubtful
    assert [(e.score_after, e.confidence) for e in kept.events] == [((1, 0), 0.95), ((1, 1), 0.6)]


def test_footage_without_a_scoreboard_is_unhealthy_not_wrong():
    for off in ("menu", "noise"):
        res, _src = _run(Timeline(duration=120.0, matches=[], off=off))
        assert res is not None and not res.healthy and res.events == [] and res.coverage == 0.0


def test_inapplicable_inputs_return_none():
    src = FakeSource(_session())
    # Another game's profile still tries the scoreboard: one recording can
    # hold several games.
    settings = types.SimpleNamespace(game_profile="valorant")
    assert D.track_scoreboard("x.mp4", src.info, settings, frame_source=src, readers=(None, None)) is not None
    empty = types.SimpleNamespace(width=0, height=0, duration=0.0)
    assert D.track_scoreboard("x.mp4", empty, None, layout=LAY, frame_source=src) is None
    eafc = types.SimpleNamespace(game_profile="fifa")
    res = D.track_scoreboard("x.mp4", src.info, eafc, frame_source=src, readers=(None, None), refine_limit=0)
    assert res is not None and res.healthy and len(res.events) == 7


def test_goal_importance_cases():
    def ev(before, after, team="a", minute=None, streamer=None) -> D.HudEvent:
        return D.HudEvent(t=0.0, team=team, score_before=before, score_after=after, t_change=0.0,
                          window=(0.0, 0.0), minute=minute, streamer=streamer)

    assert D.goal_importance(ev((0, 1), (1, 1))) == (5, ["Equaliser"])
    assert D.goal_importance(ev((1, 1), (1, 2), team="b")) == (5, ["Go-ahead goal"])
    assert D.goal_importance(ev((2, 0), (3, 0))) == (-4, ["Extends a big lead"])
    assert D.goal_importance(ev((0, 2), (1, 2))) == (0, [])
    late = D.goal_importance(ev((1, 1), (2, 1), minute=89))
    assert late == (12, ["Go-ahead goal", "Late goal (89')"])
    assert D.goal_importance(ev((3, 0), (4, 0), minute=90))[0] == -1
    comeback = D.goal_importance(ev((1, 2), (2, 2)), max_deficit=2)
    assert comeback == (9, ["Equaliser", "Comeback from 2 down"])
    assert D.goal_importance(ev((0, 0), (1, 0), streamer=True))[0] == 9
    assert D.goal_importance(ev((0, 0), (1, 0), streamer=False))[0] == 2
    assert replace(ev((0, 0), (1, 0), minute=13), clock="12:29").detail == "Scoreboard 0-0 -> 1-0 (13')"

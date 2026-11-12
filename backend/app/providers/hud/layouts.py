"""Scoreboard (HUD) layouts — where a game prints its score and match clock.

Every box is expressed in units of the frame HEIGHT, measured from the frame's
top-left corner (both x and y are divided by the height). Game HUDs are
anchored to a screen corner and scale with the vertical resolution, so one
preset fits 720p, 1080p, 1440p and 4K recordings of a 16:9 game.

Presets are measured on real footage; the pixel numbers they were measured
from are kept in the source (see ``_c``) so they can be re-checked by eye.
"""
from __future__ import annotations

from dataclasses import dataclass

# Game-profile aliases, same convention as detect_gameplay / detect_ocr.
_ALIAS = {"fifa": "eafc", "fc": "eafc", "easportsfc": "eafc"}


@dataclass(frozen=True)
class Cell:
    """A rectangle in frame-height units (x, y from the frame's top-left)."""
    x: float
    y: float
    w: float
    h: float

    def px(self, height: int) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` in pixels for a frame of the given height."""
        x0 = int(round(self.x * height))
        y0 = int(round(self.y * height))
        x1 = int(round((self.x + self.w) * height))
        y1 = int(round((self.y + self.h) * height))
        return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)


@dataclass(frozen=True)
class HudLayout:
    """Where the two score digits, the clock and the team names sit.

    ``score_a`` / ``name_a`` belong to the first team on the scoreboard (top
    row or left side), ``score_b`` / ``name_b`` to the second. ``rows`` is the
    dark block holding the team rows and ``clock`` the lighter strip with the
    match clock — together they form the structure test that tells a real
    scoreboard apart from a menu that happens to have text in the same place.
    """
    name: str
    score_a: Cell
    score_b: Cell
    clock: Cell
    rows: Cell
    name_a: Cell | None = None
    name_b: Cell | None = None
    # Seconds between the ball crossing the line and the digit changing while
    # the scoreboard stays on screen.
    score_delay_s: float = 1.8
    # Seconds between the goal and the scoreboard disappearing when a
    # celebration / replay hides it.
    cut_delay_s: float = 1.5
    # Minimum luma gap between the light clock strip and the dark rows block
    # (and between the strip and its own dark text) for the HUD to count as
    # present. Measured: solid HUD ~130-160, faded HUD 45-130, menus < 20.
    min_structure_contrast: float = 45.0

    def cells(self) -> list[Cell]:
        return [c for c in (self.score_a, self.score_b, self.clock, self.rows,
                            self.name_a, self.name_b) if c is not None]

    def region_px(self, width: int, height: int, *, pad: int = 4) -> tuple[int, int, int, int]:
        """Even-aligned pixel box ``(x, y, w, h)`` covering every cell.

        This is the only part of the frame the scanner decodes. Even alignment
        keeps ffmpeg's crop exact on 4:2:0 video, so cell offsets computed from
        this origin are never off by one.
        """
        boxes = [c.px(height) for c in self.cells()]
        x0 = max(min(b[0] for b in boxes) - pad, 0)
        y0 = max(min(b[1] for b in boxes) - pad, 0)
        x1 = min(max(b[0] + b[2] for b in boxes) + pad, width)
        y1 = min(max(b[1] + b[3] for b in boxes) + pad, height)
        x0 -= x0 % 2
        y0 -= y0 % 2
        w = x1 - x0
        h = y1 - y0
        w += w % 2
        h += h % 2
        w = min(w, width - x0)
        h = min(h, height - y0)
        return x0, y0, max(w - w % 2, 2), max(h - h % 2, 2)

    def cell_in_region(self, cell: Cell, height: int,
                       region: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """A cell's pixel box relative to the decoded ``region``'s origin."""
        x, y, w, h = cell.px(height)
        return x - region[0], y - region[1], w, h


def _c(x: float, y: float, w: float, h: float, *, measured_at: float = 1440.0) -> Cell:
    """Cell from pixel numbers measured on a frame ``measured_at`` pixels tall."""
    return Cell(x / measured_at, y / measured_at, w / measured_at, h / measured_at)


# EA SPORTS FC (FC 27, Ultimate Team: Champions and Division Rivals share this
# geometry; only the colours differ). Stacked scoreboard, top-left corner:
#
#   [logo] | AAA (crest) 2 |
#   [mode] | BBB (crest) 1 |
#          | 12:29         |
#
# Measured on a 2560x1440 recording: rows block x 207..388, y 77..149; clock
# strip y 151..181; a divider at x~335 separates names/crests from the scores.
EAFC_STACKED = HudLayout(
    name="eafc_stacked",
    score_a=_c(340, 79, 48, 33),
    score_b=_c(340, 115, 48, 33),
    clock=_c(210, 152, 182, 28),
    rows=_c(210, 79, 122, 69),
    name_a=_c(212, 79, 84, 33),
    name_b=_c(212, 115, 84, 33),
)

LAYOUTS: dict[str, HudLayout] = {
    "eafc": EAFC_STACKED,
}


def normalize_profile(profile: str | None) -> str:
    name = (profile or "generic").lower().replace(" ", "")
    return _ALIAS.get(name, name)


def layouts_for(profile: str | None) -> list[HudLayout]:
    """Layouts worth trying for a game profile.

    A named profile maps to its own layout. ``auto`` / ``generic`` try every
    known layout: scanning is cheap, and a layout that does not match the
    footage simply never passes the structure test.
    """
    name = normalize_profile(profile)
    if name in LAYOUTS:
        return [LAYOUTS[name]]
    if name in {"auto", "generic"}:
        return list(LAYOUTS.values())
    return []

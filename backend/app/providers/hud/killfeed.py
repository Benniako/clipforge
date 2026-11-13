"""Valorant kill feed: the streamer's own kills, deaths and multi-kills.

The kill feed sits in the top-right corner. Every row is

    [killer portrait] killer name  weapon  (headshot)  >  victim name [victim portrait]

with the killer's half in the killer's team colour and the victim's half in the
victim's. Allies are always teal and enemies always red, whichever side the
streamer plays. The game marks the rows the local player is part of with a pale
yellow outline: around the whole row when the player made the kill, around the
victim's end when the player died. Colour and outline are all this module
reads, so it does not depend on the game language or on the player's name.

A row stays on screen for about five seconds and rows queue top to bottom, so
the feed is sampled twice a second and each highlighted row is followed from
its first sighting to its last. Kills are then grouped into rounds (kills more
than ``ROUND_GAP_S`` apart cannot share a buy phase) to label multi-kills.

Geometry is in frame-height units and was measured on a 2560x1440 recording
with the German client: rows are 45 px tall, 52 px apart, and end 27 px from
the right edge.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

log = logging.getLogger("clipforge.hud.killfeed")

_M = 1440.0
ROW_H = 45 / _M
FEED_W = 760 / _M          # decoded region: this wide, anchored to the right edge
FEED_Y = 100 / _M
FEED_H = 400 / _M          # room for seven rows
EDGE_MARGIN = 27 / _M      # rows end this far from the right edge

SAMPLE_FPS = 2.0
ROW_GONE_S = 2.6           # a row unseen this long has left the feed
MIN_SIGHTINGS = 2          # samples a row must appear in to count
SAME_ROW = 0.20            # max share of differing signature bits for "the same row"
ROUND_GAP_S = 38.0         # own kills further apart than this are in different rounds
ROW_MAX_S = 7.0            # no row stays on screen longer than this
# Width of the yellow block left of a kill row, in row heights. Over 646 ally-kill
# row sightings in a 40-minute match the values split cleanly: 509 at <= 0.2
# (other players' kills), 5 at 0.3, 132 at >= 0.4 (the streamer's).
OWN_KILL_MIN = 0.36
OWN_KILL_MAX = 0.80
OWN_DEATH_MIN = 0.6        # outline strength on the victim's end of a death row (real ones read ~1.0)

PROFILES = frozenset({"valorant"})
MULTI_LABELS = {1: "kill", 2: "double_kill", 3: "triple_kill", 4: "quad_kill", 5: "ace"}

FrameSource = Callable[..., Iterator[tuple[float, object]]]


@dataclass
class FeedRow:
    """One kill-feed row as seen in a single frame."""
    y0: int
    y1: int
    killer: str | None            # "ally" | "enemy"
    victim: str | None
    own: str | None               # "kill" | "death" | None
    strength: float               # outline strength behind ``own`` (0..1)
    sig: object = None            # boolean text-mask signature
    headshot: float = 0.0         # match (0..1) of the headshot icon left of the victim


@dataclass
class KillEvent:
    t: float                      # when the row first appeared (s, source timeline)
    kind: str                     # "kill" | "death"
    confidence: float
    t_last: float = 0.0           # last sample the row was still visible
    round_index: int = 0          # 1-based group of kills (see ROUND_GAP_S)
    streak: int = 1               # n-th own kill of that round
    sources: list[str] = field(default_factory=lambda: ["feed"])   # "feed" and/or "banner"
    headshot: bool = False        # the feed row carried the headshot icon

    @property
    def detail(self) -> str:
        if self.kind == "death":
            return "Kill feed: the streamer was killed"
        shot = " (headshot)" if self.headshot else ""
        return f"Kill feed: kill {self.streak}{shot} of round group {self.round_index}"


@dataclass
class MultiKill:
    """The streamer's kills of one round."""
    index: int
    kills: list[KillEvent]
    won: bool | None = None       # round result off the top bar; None = not read
    after_plant: bool = False     # a kill fell while the spike was planted
    clutch: object | None = None  # valorant_rounds.Clutch when the streamer won this round alone

    @property
    def count(self) -> int:
        return len(self.kills)

    @property
    def label(self) -> str:
        return MULTI_LABELS.get(min(self.count, 5), "kill")

    @property
    def t_first(self) -> float:
        return self.kills[0].t

    @property
    def t_last(self) -> float:
        return self.kills[-1].t

    @property
    def span(self) -> float:
        return self.t_last - self.t_first

    @property
    def headshots(self) -> int:
        return sum(1 for k in self.kills if k.headshot)


@dataclass
class KillfeedResult:
    events: list[KillEvent]                 # kills and deaths, in time order
    rounds: list[MultiKill]
    samples: int
    sample_interval: float
    active_s: float                         # seconds with at least one feed row on screen
    healthy: bool
    diagnostics: dict = field(default_factory=dict)

    @property
    def kills(self) -> list[KillEvent]:
        return [e for e in self.events if e.kind == "kill"]

    @property
    def deaths(self) -> list[KillEvent]:
        return [e for e in self.events if e.kind == "death"]


def supports(profile: str | None) -> bool:
    return (profile or "").lower().replace(" ", "") in PROFILES


def feed_region(width: int, height: int) -> tuple[int, int, int, int]:
    """Even-aligned ``(x, y, w, h)`` pixel box of the kill feed."""
    w = min(int(round(FEED_W * height)), width)
    w -= w % 2
    x = width - w
    x -= x % 2
    y = int(round(FEED_Y * height))
    y -= y % 2
    h = min(int(round(FEED_H * height)), height - y)
    h -= h % 2
    return x, y, max(w, 2), max(h, 2)


# --- kill banner: a white skull in a ring, bottom centre, ~2 s after each kill ---
BANNER_CY = 1150 / _M      # centre of the ring (x is the frame centre)
BANNER_HALF = 90 / _M      # half-size of the decoded square
BANNER_RING = 68 / _M      # ring radius


def banner_region(width: int, height: int) -> tuple[int, int, int, int]:
    half = int(round(BANNER_HALF * height))
    half += half % 2
    x = width // 2 - half
    x -= x % 2
    y = int(round(BANNER_CY * height)) - half
    y -= y % 2
    return x, y, 2 * half, 2 * half


def banner_present(bgr, height: int) -> float:
    """0..1 confidence that the kill banner (skull in a ring) is in ``bgr``.

    ``bgr`` is the :func:`banner_region` square. The skull's cranium and jaw
    must be light, the gap between skull and ring dark, and the ring itself
    light: a bright wall or a flash fills the gap too and is rejected.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    light = (hsv[..., 1] <= 75) & (hsv[..., 2] >= 165)
    u = height / _M
    c = bgr.shape[0] // 2

    def share(x0, x1, y0, y1) -> float:
        patch = light[c + int(y0 * u):c + int(y1 * u), c + int(x0 * u):c + int(x1 * u)]
        return float(patch.mean()) if patch.size else 0.0

    cranium = share(-24, 24, -46, -30)
    jaw = share(-14, 14, 28, 44)
    gap = max(share(-46, -36, -18, 18), share(36, 46, -18, 18))
    ang = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    hits = []
    for r in (BANNER_RING * height - 1.5, BANNER_RING * height, BANNER_RING * height + 1.5):
        xs = np.clip((c + r * np.cos(ang)).astype(int), 0, bgr.shape[1] - 1)
        ys = np.clip((c + r * np.sin(ang)).astype(int), 0, bgr.shape[0] - 1)
        hits.append(light[ys, xs])
    ring = float(np.any(hits, axis=0).mean())
    if cranium < 0.75 or jaw < 0.6 or gap > 0.3 or ring < 0.4:
        return 0.0
    return round(min(1.0, 0.5 * cranium + 0.25 * jaw + 0.25 * ring), 3)


# --- buy-phase plate: top centre for the first 30 s of every round -------------
# Measured: plate x 1062..1497, y 180..375; the big word fills y 225..307.
PLATE = (1062 / _M, 180 / _M, 436 / _M, 196 / _M)
ROUND_START_GAP_S = 20.0   # plate sightings further apart start a new round
SPECTATE_FALLBACK_S = 45.0


def plate_region(width: int, height: int) -> tuple[int, int, int, int]:
    w = int(round(PLATE[2] * height))
    w += w % 2
    h = int(round(PLATE[3] * height))
    h += h % 2
    x = width // 2 - w // 2
    x -= x % 2
    y = int(round(PLATE[1] * height))
    y -= y % 2
    return x, y, w, h


def plate_present(bgr) -> bool:
    """True when the buy-phase plate is in ``bgr`` (the :func:`plate_region` box).

    The plate has a thin white frame line along its top and bottom, and one
    large white word in between. A bright wall is light everywhere, so the
    band of the word must be only partly light.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    # Measured: the letters read V 200-215; the big word then fills 35-53 % of
    # its band, while the small header and footer lines stay nearly empty.
    light = (hsv[..., 1] <= 70) & (hsv[..., 2] >= 200)
    h, w = light.shape
    xa, xb = int(0.08 * w), int(0.92 * w)
    word = float(light[int(0.25 * h):int(0.63 * h), xa:xb].mean())
    header = float(light[int(0.03 * h):int(0.20 * h), xa:xb].mean())
    footer = float(light[int(0.80 * h):int(0.97 * h), xa:xb].mean())
    return 0.25 <= word <= 0.62 and header <= 0.25 and footer <= 0.30


def round_starts(plate_times: list[float]) -> list[float]:
    """First sighting of each buy phase; isolated single sightings are dropped."""
    starts: list[float] = []
    run: list[float] = []
    for t in sorted(plate_times) + [float("inf")]:
        if run and t - run[-1] > ROUND_START_GAP_S:
            if len(run) >= 3:
                starts.append(run[0])
            run = []
        run.append(t)
    return starts


def drop_spectated(events: list[KillEvent], starts: list[float]) -> list[KillEvent]:
    """Remove banner-only kills seen while the streamer was dead.

    The kill banner also shows for the teammate being spectated. A banner kill
    is the streamer's own only if they have not died since the round started;
    without round starts (deathmatch) a fixed window after each death is used.
    """
    deaths = sorted(e.t for e in events if e.kind == "death")
    out = []
    for e in events:
        if e.kind == "kill" and "feed" not in e.sources:
            last = max((d for d in deaths if d < e.t), default=None)
            if last is not None:
                revived = any(last < s <= e.t for s in starts)
                if not revived and (starts or e.t - last <= SPECTATE_FALLBACK_S):
                    continue
        out.append(e)
    return out


def _masks(bgr):
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1]
    v = hsv[..., 2]
    teal = (h >= 70) & (h <= 95) & (s >= 55) & (v >= 110)
    red = ((h <= 8) | (h >= 172)) & (s >= 105) & (v >= 140)
    # The outline is pale yellow blended into the row colour: yellow-green on
    # teal (hue 30-65), measured on real frames.
    outline = (h >= 24) & (h <= 66) & (s >= 38) & (v >= 128)
    white = (s <= 60) & (v >= 185)
    # The local player's portrait box and arrow tab are a solid pale yellow.
    box = (h >= 18) & (h <= 36) & (s >= 50) & (v >= 120)
    # ... and the name plate next to it is a washed-out teal.
    pale = (h >= 70) & (h <= 140) & (s <= 58) & (v >= 140) & (v < 185)
    return teal, red, outline, white, box, pale


def _longest_run(flags) -> tuple[int, int]:
    """``(length, end)`` of the longest run of true values; ``end`` is exclusive."""
    best = cur = end = 0
    for i, f in enumerate(flags):
        cur = cur + 1 if f else 0
        if cur > best:
            best, end = cur, i + 1
    return best, end


def _bands(team, row_h: float) -> list[tuple[int, int]]:
    """Vertical extents of the feed rows in a team-colour mask."""
    need = int(2.6 * row_h)
    on = team.sum(axis=1) >= need
    runs: list[list[int]] = []
    start = None
    for y, flag in enumerate(list(on) + [False]):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            runs.append([start, y])
            start = None
    gap = max(2, int(round(0.10 * row_h)))   # white text can split a row in two
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if 0.7 * row_h <= b - a <= 1.35 * row_h]


def _two_line_share(mask, lines, xa: int, xb: int) -> float:
    """Second-highest share of set pixels among ``lines``: two lines must agree,
    so a one-pixel blend between a row and a bright wall behind it cannot pass."""
    height = mask.shape[0]
    vals = sorted((float(mask[y, xa:xb].mean()) for y in lines if 0 <= y < height),
                  reverse=True)
    return vals[1] if len(vals) > 1 else 0.0


# --- headshot icon: a reticle just left of the victim's half of a kill row -----
# Four bars around an empty centre and four short arcs on the diagonals,
# white, about 0.7 row heights across, its centre 0.6 row heights left of where
# the victim's colour starts (measured on 1440p rows: 26-29 px of 45).
# Score needed to call a row a headshot. The score is the template match times
# the mirror symmetry of the matched patch: the wall-penetration icon (an arrow
# into a bar) sits in the same spot and half-matches the template, but it is
# lopsided. Measured on a 40-minute 1440p match, 1,356 kill-row sightings: no
# icon (wall-penetration icons included) at most 0.29, the headshot icon 0.39
# to 0.78, and 0.70 or more on the streamer's own rows. Nothing in between.
HEADSHOT_MIN = 0.34
HEADSHOT_UNSURE = 0.05        # +/- around HEADSHOT_MIN reported as "unsure"


def _reticle(row_h: float):
    """The headshot icon drawn at ``row_h``, as a float32 image (0..1)."""
    import cv2
    import numpy as np

    u = row_h / 45.0
    size = int(round(36 * u)) | 1
    c = size // 2
    img = np.zeros((size, size), np.uint8)
    t = max(1, int(round(3 * u)))

    def seg(x0, y0, x1, y1):
        cv2.line(img, (int(round(c + x0 * u)), int(round(c + y0 * u))),
                 (int(round(c + x1 * u)), int(round(c + y1 * u))), 255, t)

    seg(0, -16, 0, -5)            # top bar
    seg(0, 5, 0, 16)              # bottom bar
    seg(-15, 0, -6, 0)            # left bar
    seg(6, 0, 15, 0)              # right bar
    for sx in (-1, 1):            # four arcs, tangent to a circle of radius 12.5
        for sy in (-1, 1):
            mx, my = sx * 8.8, sy * 8.8
            seg(mx - sy * 3.4, my + sx * 3.4, mx + sy * 3.4, my - sx * 3.4)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img.astype(np.float32) / 255.0


_RETICLES: dict[int, object] = {}


def _symmetry(patch) -> float:
    """How mirror-symmetric a patch is, left-right and top-bottom (-1..1)."""
    import numpy as np

    if patch.size == 0 or float(patch.std()) < 1e-6:
        return 0.0
    flat = patch.ravel()
    lr = float(np.corrcoef(flat, patch[:, ::-1].ravel())[0, 1])
    ud = float(np.corrcoef(flat, patch[::-1, :].ravel())[0, 1])
    return min(lr, ud)


def headshot_score(white, y0: int, y1: int, boundary: int, row_h: float) -> float:
    """How surely the strip left of ``boundary`` shows the headshot icon (0..1).

    ``white`` is the white-pixel mask of the feed image, ``boundary`` the first
    column of the victim's half of the row. The score is the best template
    match there times the symmetry of the patch it matched (see HEADSHOT_MIN).
    """
    import cv2
    import numpy as np

    key = int(round(row_h * 4))
    tpl = _RETICLES.get(key)
    if tpl is None:
        tpl = _RETICLES[key] = _reticle(row_h)
    pad = max(2, int(round(0.08 * row_h)))
    ya, yb = max(y0 - pad, 0), min(y1 + pad, white.shape[0])
    xa, xb = max(boundary - int(round(1.3 * row_h)), 0), min(boundary + pad, white.shape[1])
    win = white[ya:yb, xa:xb].astype(np.float32)
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1] or not win.any():
        return 0.0
    win = cv2.GaussianBlur(win, (3, 3), 0)
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _lo, match, _lo_at, (x, y) = cv2.minMaxLoc(res)
    if match <= 0.0:
        return 0.0
    th, tw = tpl.shape
    sym = _symmetry(win[y:y + th, x:x + tw])
    return float(match * max(sym, 0.0))


def _victim_start(lab, victim_code: int, start: int, row_h: float) -> int:
    """First column of the victim's colour, walking left from ``start``.

    White name text and the slanted edge interrupt the colour for a few
    columns, so gaps up to half a row height are bridged.
    """
    gap_max = int(0.5 * row_h)
    x, gap, bx = start, 0, start
    while x > 0:
        if lab[x] == victim_code:
            bx, gap = x, 0
        else:
            gap += 1
            if gap > gap_max:
                break
        x -= 1
    return bx


def read_rows(bgr, row_h: float, edge_x: int | None = None) -> list[FeedRow]:
    """Kill-feed rows in a colour image of the feed region, top to bottom.

    ``row_h`` is the row height in pixels and ``edge_x`` the column where rows
    end (default: the region's right edge minus the usual margin).
    """
    import cv2
    import numpy as np

    teal, red, outline, white, box, _pale = _masks(bgr)
    width = bgr.shape[1]
    if edge_x is None:
        edge_x = width - int(round(EDGE_MARGIN / ROW_H * row_h))
    k = max(2, int(round(0.09 * row_h)))
    rows: list[FeedRow] = []
    for y0, y1 in _bands(teal | red, row_h):
        ct = teal[y0:y1].mean(axis=0)
        cr = red[y0:y1].mean(axis=0)
        lab = np.zeros(width, np.int8)
        lab[(ct >= 0.35) & (ct > cr)] = 1
        lab[(cr >= 0.35) & (cr >= ct)] = 2

        # victim: the strip between the victim's name and portrait
        va = max(edge_x - int(2.9 * row_h), 0)
        vb = max(edge_x - int(1.85 * row_h), va + 1)
        n_ally = int((lab[va:vb] == 1).sum())
        n_enemy = int((lab[va:vb] == 2).sum())
        if n_ally + n_enemy < 0.3 * (vb - va):
            continue
        victim = "ally" if n_ally >= n_enemy else "enemy"
        other = 2 if victim == "ally" else 1
        # killer: the other colour must fill a good stretch left of the victim
        wa = max(va - int(14 * row_h), 0)
        xs = np.flatnonzero(lab[wa:va] == other) + wa
        if len(xs) >= 1.2 * row_h:
            killer = "enemy" if victim == "ally" else "ally"
            ka = int(np.percentile(xs, 15))
        else:
            killer, ka = victim, wa

        # The outline lines sit just outside the teal part of a row but inside
        # its red part, so the band's edge can fall on either side of them.
        top = range(y0 - k, y0 + k)
        bottom = range(y1 - k - 1, y1 + k)
        own, strength = None, 0.0
        if killer == "ally" and victim == "enemy":
            # Own kill: a solid yellow block (arrow tab + portrait box) left of
            # the killer's name. Measured ~1 row height wide; an agent portrait
            # or the small assist icon never fills a full-height block that wide.
            lo = max(ka - int(9 * row_h), 0)
            cols = box[y0:y1, lo:ka].mean(axis=0) >= 0.5
            strength = min(1.0, _longest_run(cols)[0] / row_h)
            # The player's tab is about half a row height wide (measured
            # 0.44-0.60). A wider yellow block is a teammate's yellow agent
            # portrait. This test alone still lets some teammates' rows
            # through, so a feed kill only ever confirms a kill banner (see
            # merge_banner_kills); it never counts as a kill on its own.
            if OWN_KILL_MIN <= strength <= OWN_KILL_MAX:
                own = "kill"
        elif killer == "enemy" and victim == "ally":
            strength = max(_two_line_share(outline, top, va, edge_x),
                           _two_line_share(outline, bottom, va, edge_x))
            if strength >= OWN_DEATH_MIN:
                own = "death"

        hs = 0.0
        if killer != victim:
            start = min(edge_x - int(2.0 * row_h), width - 1)
            boundary = _victim_start(lab, 1 if victim == "ally" else 2, start, row_h)
            hs = headshot_score(white, y0, y1, boundary, row_h)

        sa = max(edge_x - int(12 * row_h), 0)
        sb = max(edge_x - int(1.75 * row_h), sa + 2)
        patch = white[y0:y1, sa:sb].astype(np.uint8) * 255
        sig = cv2.resize(patch, (48, 6), interpolation=cv2.INTER_AREA) > 60
        rows.append(FeedRow(y0=y0, y1=y1, killer=killer, victim=victim, own=own,
                            strength=round(float(strength), 3), sig=sig,
                            headshot=round(hs, 3)))
    return rows


def _distance(a, b) -> float:
    return float((a != b).mean())


class FeedTracker:
    """Follows highlighted rows across samples and emits one event per row."""

    def __init__(self, interval: float = 1.0 / SAMPLE_FPS) -> None:
        self.interval = interval
        self._active: list[dict] = []
        self.events: list[KillEvent] = []

    def push(self, t: float, rows: list[FeedRow]) -> None:
        taken: set[int] = set()
        own = [r for r in rows if r.own]
        pending: list[FeedRow] = []
        for attempt in (0, 1):
            # Pass 0 pairs rows with entries by signature. Pass 1 gives a row
            # that matched nothing to a still-unpaired entry of its kind: the
            # same row whose text mask changed (fading, moving up a slot), not
            # a new kill, as long as that entry is young enough to still be up.
            todo, pending = (own if attempt == 0 else pending), []
            for row in todo:
                best, best_d = None, (SAME_ROW if attempt == 0 else 1.0)
                for i, a in enumerate(self._active):
                    if i in taken or a["kind"] != row.own:
                        continue
                    if attempt == 1 and t - a["first"] > ROW_MAX_S:
                        continue
                    d = _distance(a["sig"], row.sig)
                    if d <= best_d:
                        best, best_d = i, d
                if best is None and attempt == 0:
                    pending.append(row)
                    continue
                self._pair(t, row, best, taken)
        self._expire(t)

    def _pair(self, t: float, row: FeedRow, best: int | None, taken: set[int]) -> None:
        hs = 1 if row.headshot >= HEADSHOT_MIN else 0
        if best is None:
            self._active.append({"kind": row.own, "sig": row.sig, "first": t,
                                 "last": t, "seen": 1, "strength": row.strength, "hs": hs})
            taken.add(len(self._active) - 1)
            return
        a = self._active[best]
        a["last"] = t
        a["seen"] += 1
        a["hs"] += hs
        a["strength"] = max(a["strength"], row.strength)
        a["sig"] = row.sig
        taken.add(best)

    def _expire(self, t: float | None) -> None:
        keep = []
        for a in self._active:
            if t is not None and t - a["last"] <= ROW_GONE_S:
                keep.append(a)
            elif a["seen"] >= MIN_SIGHTINGS:
                # The row appeared somewhere in the sample gap before "first".
                # The icon stays on the row its whole life; a majority of the
                # sightings rides out a fade or a frame where it was covered.
                self.events.append(KillEvent(
                    t=round(max(a["first"] - self.interval / 2.0, 0.0), 3), kind=a["kind"],
                    confidence=round(min(0.99, 0.6 + 0.4 * a["strength"]), 3),
                    t_last=round(a["last"], 3),
                    headshot=a["hs"] > 0 and 2 * a["hs"] >= a["seen"]))
        self._active = keep

    def finish(self) -> list[KillEvent]:
        self._expire(None)
        self.events.sort(key=lambda e: e.t)
        return self.events


def merge_banner_kills(events: list[KillEvent], banner_runs: list[list[float]], *,
                       interval: float, window_s: float = 3.0) -> list[KillEvent]:
    """Combine kill-feed kills with kill-banner sightings.

    Every kill needs the banner: it shows for each kill of the player on screen
    and is far more reliable than the feed's highlight (verified on a 40-minute
    match: 11 banners, 8 own kills + 3 of a spectated teammate; the feed test
    alone also fired on 15 teammates' rows). A feed kill within ``window_s``
    marks the banner as confirmed by the feed, which is what tells an own kill
    from a spectated teammate's when round starts were missed.
    """
    out = [e for e in events if e.kind != "kill"]
    feed = [e for e in events if e.kind == "kill"]
    used: set[int] = set()
    previous: KillEvent | None = None
    for first, last, conf in banner_runs:
        t = max(first - interval / 2.0, 0.0)
        near = [i for i, k in enumerate(feed) if i not in used and abs(k.t - t) <= window_s]
        # A screen flash can hide the banner for one sample, which splits one
        # banner into two sightings. A sighting that resumes a single missed
        # sample after the previous one, with no feed row of its own, is the
        # same banner; a real second kill brings its own feed row.
        if (previous is not None and not near
                and first - previous.t_last <= 2.0 * interval + 0.05):
            previous.t_last = round(last, 3)
            continue
        sources = ["banner"]
        confidence = 0.6 + 0.3 * conf
        headshot = False
        if near:
            match = min(near, key=lambda i: abs(feed[i].t - t))
            used.add(match)
            sources.append("feed")
            confidence = min(0.99, confidence + 0.08)
            # Only the feed row shows whether it was a headshot.
            headshot = feed[match].headshot
        previous = KillEvent(t=round(t, 3), kind="kill", confidence=round(confidence, 3),
                             t_last=round(last, 3), sources=sources, headshot=headshot)
        out.append(previous)
    out.sort(key=lambda e: e.t)
    return out


def group_rounds(events: list[KillEvent], *, gap_s: float = ROUND_GAP_S,
                 starts: list[float] | None = None) -> list[MultiKill]:
    """Group own kills into rounds and number them (mutates the events).

    With ``starts`` (buy-phase sightings) a round is the time between two of
    them; without, kills more than ``gap_s`` apart are taken as separate rounds.
    """
    import bisect

    starts = sorted(starts or [])
    rounds: list[MultiKill] = []
    last_key = None
    for ev in sorted((e for e in events if e.kind == "kill"), key=lambda e: e.t):
        key = bisect.bisect_right(starts, ev.t) if starts else None
        new = (key != last_key) if starts else (not rounds or ev.t - rounds[-1].kills[-1].t > gap_s)
        last_key = key
        if not rounds or new:
            rounds.append(MultiKill(index=len(rounds) + 1, kills=[]))
        rounds[-1].kills.append(ev)
        ev.round_index = rounds[-1].index
        ev.streak = len(rounds[-1].kills)
    return rounds


def _has_nvidia() -> bool:
    try:
        from ...config import get_settings

        return bool(get_settings().has_nvidia)
    except Exception:
        return False


def track_killfeed(src_path: str, info, *, frame_source: FrameSource | None = None,
                   fps: float = SAMPLE_FPS, start: float | None = None,
                   duration: float | None = None) -> KillfeedResult | None:
    """Read the kill feed of ``src_path``; ``None`` when the video cannot be read.

    ``info`` needs ``width``, ``height`` and ``duration``. ``frame_source``
    replaces ffmpeg in tests; it is called as
    ``source(region, keyframes=False, fps=..., pix_fmt="bgr24", ...)``.
    """
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except Exception as e:
        log.info("killfeed: OpenCV/NumPy unavailable, skipped (%s)", e)
        return None
    width, height = int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return None
    started = time.monotonic()
    region = feed_region(width, height)
    row_h = ROW_H * height
    edge_x = region[2] - int(round(EDGE_MARGIN * height))
    tracker = FeedTracker(interval=1.0 / fps)
    samples = with_rows = own_rows = 0
    hs_bands = [0, 0, 0]                     # kill-row sightings: no / unsure / yes icon
    banner = banner_region(width, height)
    banner_runs: list[list[float]] = []      # [first_t, last_t, best confidence]
    plate = plate_region(width, height)
    plate_times: list[float] = []

    def analyse(stacked):
        """What one sample shows: (feed rows, banner confidence, plate on)."""
        frame = stacked[:region[3]]
        y_b = region[3]
        y_p = y_b + banner[3]
        conf = banner_present(stacked[y_b:y_p, :banner[2]], height)
        plate_on = plate_present(stacked[y_p:y_p + plate[3], :plate[2]])
        # Most samples show no feed at all: a quarter-size colour test
        # skips them before the full-resolution read.
        small = _masks(frame[::4, ::4])
        busy = int((small[0] | small[1]).sum()) >= 0.35 * row_h * row_h / 16.0
        return (read_rows(frame, row_h, edge_x) if busy else []), conf, plate_on

    # Reading a sample needs no history, so the video is decoded in parallel
    # chunks; only the tracking below has to see the samples in order.
    readings: list[tuple[float, tuple]] = []
    try:
        if frame_source is not None:
            for t, stacked in frame_source(region, keyframes=False, fps=fps, start=start,
                                           duration=duration, hwaccel=_has_nvidia(),
                                           pix_fmt="bgr24", stack=[banner, plate]):
                readings.append((float(t), analyse(stacked)))
        else:
            from . import scan

            t0 = float(start or 0.0)
            length = float(duration) if duration is not None else \
                max(float(getattr(info, "duration", 0.0) or 0.0) - t0, 0.0)
            readings = scan.map_frames(str(src_path), region, analyse, fps=fps, start=t0,
                                       duration=length, hwaccel=_has_nvidia(),
                                       pix_fmt="bgr24", stack=[banner, plate])
    except Exception as e:
        log.warning("killfeed: scan failed after %d samples: %s", len(readings), e)
        if not readings:
            return None
    for t, (rows, conf, plate_on) in readings:
        if plate_on:
            plate_times.append(float(t))
        if conf > 0.0:
            if banner_runs and float(t) - banner_runs[-1][1] <= 1.5 / fps + 0.05:
                banner_runs[-1][1] = float(t)
                banner_runs[-1][2] = max(banner_runs[-1][2], conf)
            else:
                banner_runs.append([float(t), float(t), conf])
        samples += 1
        with_rows += bool(rows)
        own_rows += sum(1 for r in rows if r.own)
        for r in rows:
            if r.killer != r.victim:
                # Icon scores split cleanly on real footage; many rows close
                # to the cut-off would mean the icon or the layout has changed.
                if abs(r.headshot - HEADSHOT_MIN) < HEADSHOT_UNSURE:
                    hs_bands[1] += 1
                else:
                    hs_bands[2 if r.headshot >= HEADSHOT_MIN else 0] += 1
        tracker.push(float(t), rows)
    starts = round_starts(plate_times)
    events = drop_spectated(
        merge_banner_kills(tracker.finish(), banner_runs, interval=1.0 / fps), starts)
    rounds = group_rounds(events, starts=starts)
    active_s = with_rows / fps
    result = KillfeedResult(
        events=events, rounds=rounds, samples=samples, sample_interval=1.0 / fps,
        active_s=round(active_s, 1),
        # A real match shows feed rows for minutes in total; a few stray
        # teal/red shapes in other footage do not.
        healthy=samples >= 20 and active_s >= 20.0,
        diagnostics={"own_row_sightings": own_rows, "round_starts": [round(s, 1) for s in starts],
                     "banner_sightings": len(banner_runs),
                     "headshot_icon_rows": {"no": hs_bands[0], "unsure": hs_bands[1],
                                            "yes": hs_bands[2]},
                     "seconds": round(time.monotonic() - started, 1)})
    log.info("kill feed: %d kill(s), %d death(s), %d round group(s), feed visible %.0fs, read in %.0fs",
             len(result.kills), len(result.deaths), len(rounds), active_s,
             result.diagnostics["seconds"])
    return result

"""Valorant round results, spike plants and clutches, read off the top bar.

The top bar shows the own team's score left of the round timer and the
enemy's right of it, as white digits on translucent plates. When the spike is
planted the timer is replaced by a red spike icon. Further out sit the
portraits of the players still alive: a dead player's portrait is removed and
the others move up towards the centre.

* **Scores** are read with the digit OCR on every keyframe (about every 4 s).
  Reads on the translucent plates are noisy (measured: 40-60 % come back
  empty, a few gain or lose a digit, e.g. 3 read as 13). A score never falls
  within a match, so the longest series of reads that never falls is kept and
  everything else is a misread; a value needs two reads, or one read that
  fills a +1 step exactly.
* **Spike plants** need the red icon on two keyframes in a row.
* **Alive counts** are the filled portrait slots counted from the centre
  outwards: a portrait has strong edges, the empty band almost none
  (measured mean |Sobel| 80-200 against 5-30). A **clutch** is a round in
  which the streamer was the last one alive on two samples in a row, had not
  died, and the team won.

Checked on a 40-minute match that ended 8-13: the 20 score changes up to 8-12
decode in order (the final 13 never shows before the end screen; spot-checked
by eye at 1:40, 3:20 and 5:00), 10 plants are found, and the alive counts match
the frames at six moments checked by eye.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterator

log = logging.getLogger("clipforge.hud.valorant_rounds")

_M = 1440.0
BAR_Y = 42 / _M
BAR_H = 52 / _M
BAR_W = 440 / _M
SCORE_DX = 183 / _M        # score plates sit this far left / right of the centre
SCORE_HALF = 26 / _M
ICON_HALF = 40 / _M
OCR_EVERY = 1              # keyframes between two score reads (OCR here is ~10 ms a read)
MIN_OCR_CONF = 0.4
MAX_SCORE = 30
SPIKE_MIN_RED = 0.10       # share of red pixels where the timer normally is
SPIKE_GAP_S = 9.0
RESULT_WINDOW_S = 150.0    # a round's result shows up at most this long after its last kill
POST_ROUND_S = 10.0        # kills can still happen for a few seconds after a round ends
MIN_READS = 2              # reads a score value needs before it counts
RESET_MIN_SUM = 4          # a 0-0 read after this many rounds starts a new match
MATCH_GAP_S = 180.0        # samples further apart than this belong to different matches
# The whole top bar: portraits, score plates and timer.
STRIP_Y = 36 / _M
STRIP_H = 58 / _M
STRIP_W = 1380 / _M
# Portrait slot k (0 = next to the score plate) is centred this far from the middle.
SLOT_DX = 310 / _M
SLOT_PITCH = 87.5 / _M
SLOT_HALF = 24 / _M
SLOT_Y0 = 38 / _M
SLOT_Y1 = 88 / _M
PORTRAIT_MIN_EDGE = 60.0   # mean |Sobel| in a slot: portraits read 80-200, the empty band 5-30
CLUTCH_CONFIRM_S = 10.0    # the two samples showing the streamer alone must be this close
# A death this long before the last full-team sample still counts for the round:
# the feed row and the portrait leaving do not line up exactly with a keyframe
# (seen: a death at 16:52 and a full team on the 16:52 keyframe). The previous
# round's deaths are at least the 30 s buy phase earlier.
DEATH_SLACK_S = 5.0

Span = tuple[float, float]
FrameSource = Callable[..., Iterator[tuple[float, object]]]


@dataclass
class RoundResult:
    t: float                       # first sample that showed the new score
    winner: str                    # "ally" | "enemy"
    score: tuple[int, int]         # (ally, enemy) after the round
    t_low: float = 0.0             # last sample that still showed the old score
    exact: bool = True             # False when a value was never read (two rounds at once)


@dataclass
class BarSample:
    t: float
    ally: int | None
    enemy: int | None
    spike: bool
    alive_ally: int | None = None      # players alive per team; None when unsure
    alive_enemy: int | None = None


@dataclass
class Clutch:
    t: float                       # first sample with the streamer as the last one alive
    enemies: int                   # enemies alive then: the N of 1vN
    t_end: float                   # first sample showing the round's result
    won: bool
    t_live: float = 0.0            # last sample with the whole team alive (the round was on)


def strip_region(width: int, height: int) -> tuple[int, int, int, int]:
    """The whole top bar; :func:`bar_region` lies inside it."""
    w = min(int(STRIP_W * height), width)
    w -= w % 2
    h = max(int(STRIP_H * height), 2)
    h -= h % 2
    x = (width - w) // 2
    x -= x % 2
    y = int(STRIP_Y * height)
    y -= y % 2
    return x, y, w, h


def alive_counts(strip, height: int) -> tuple[int | None, int | None]:
    """Players alive per team in a :func:`strip_region` image; None where unsure.

    A count is the number of filled slots from the centre outwards. A filled
    slot beyond an empty one cannot happen in the game, so that reads as None;
    so does everything when the team bands are not on screen (menus, flashes).
    """
    import cv2
    import numpy as np

    from .segments import topbar_present

    if not topbar_present(strip, height):
        return None, None
    gray = cv2.cvtColor(np.ascontiguousarray(strip), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = (np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
           + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)))
    w = strip.shape[1]
    c = w // 2
    y0 = max(int((SLOT_Y0 - STRIP_Y) * height), 0)
    y1 = min(int((SLOT_Y1 - STRIP_Y) * height), strip.shape[0])

    def filled(sign: int, k: int) -> bool:
        cx = c + sign * (SLOT_DX + SLOT_PITCH * k) * height
        x0, x1 = int(cx - SLOT_HALF * height), int(cx + SLOT_HALF * height)
        if x0 < 0 or x1 > w or y1 <= y0:
            return False
        return float(mag[y0:y1, x0:x1].mean()) >= PORTRAIT_MIN_EDGE

    counts: list[int | None] = []
    for sign in (-1, 1):
        flags = [filled(sign, k) for k in range(5)]
        n = flags.index(False) if False in flags else 5
        counts.append(None if any(flags[n:]) else n)
    return counts[0], counts[1]


def bar_region(width: int, height: int) -> tuple[int, int, int, int]:
    w = min(int(BAR_W * height), width)
    w -= w % 2
    h = max(int(BAR_H * height), 2)
    h -= h % 2
    x = (width - w) // 2
    x -= x % 2
    y = int(BAR_Y * height)
    y -= y % 2
    return x, y, w, h


def _cols(width: int, height: int, dx: float, half: float) -> slice:
    c = width // 2 + int(dx * height)
    return slice(max(c - int(half * height), 0), min(c + int(half * height), width))


def digit_image(bgr):
    """Black-on-white image of the white digit in ``bgr``, or None when there is none."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] <= 70) & (hsv[..., 2] >= 170)
    if not 0.04 <= float(white.mean()) <= 0.55:
        return None
    img = np.where(white, 0, 255).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


def spike_icon(bgr, height: int) -> bool:
    """True when the red spike icon sits where the round timer normally is."""
    import cv2
    import numpy as np

    patch = np.ascontiguousarray(bgr)[:, _cols(bgr.shape[1], height, 0.0, ICON_HALF)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    red = ((h <= 8) | (h >= 172)) & (hsv[..., 1] >= 110) & (hsv[..., 2] >= 120)
    return float(red.mean()) >= SPIKE_MIN_RED


def _read_score(reader, bgr) -> int | None:
    from .readers import clean_digits

    img = digit_image(bgr)
    if img is None:
        return None
    text, conf = reader.read_digits(img)
    value = clean_digits(text)
    if value is None or conf < MIN_OCR_CONF or value > MAX_SCORE:
        return None
    return value


def read_bar(src_path: str, info, *, spans: list[Span] | None = None, reader=None,
             frame_source: FrameSource | None = None) -> list[BarSample]:
    """One keyframe pass over the top bar; samples outside ``spans`` are skipped."""
    width, height = int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return []
    if reader is None:
        from .readers import pick_readers

        reader = pick_readers()[0]
    region = strip_region(width, height)
    bar = bar_region(width, height)
    rows = slice(max(bar[1] - region[1], 0), min(bar[1] - region[1] + bar[3], region[3]))
    cols = slice(bar[0] - region[0], bar[0] - region[0] + bar[2])
    if frame_source is None:
        from . import scan

        def frame_source(reg, **kw):
            return scan.iter_frames(src_path, reg, **kw)

    ally = _cols(bar[2], height, -SCORE_DX, SCORE_HALF)
    enemy = _cols(bar[2], height, SCORE_DX, SCORE_HALF)
    out: list[BarSample] = []
    n = 0
    for t, strip in frame_source(region, keyframes=True, pix_fmt="bgr24"):
        t = float(t)
        if spans and not any(a <= t <= b for a, b in spans):
            continue
        frame = strip[rows, cols]
        a = e = None
        if reader is not None and n % OCR_EVERY == 0:
            a, e = _read_score(reader, frame[:, ally]), _read_score(reader, frame[:, enemy])
        n += 1
        alive_a, alive_e = alive_counts(strip, height)
        out.append(BarSample(t=t, ally=a, enemy=e, spike=spike_icon(frame, height),
                             alive_ally=alive_a, alive_enemy=alive_e))
    return out


def _monotone(reads: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """The longest series of reads whose values never fall.

    A score only rises during a match, so every read outside that series is a
    misread (a dropped or an extra digit: 12 for 2, 1 for 11).
    """
    import bisect

    tails: list[int] = []      # tails[k]: smallest last value of a series of length k + 1
    ends: list[int] = []       # index of that last read
    prev = [-1] * len(reads)
    for i, (_t, v) in enumerate(reads):
        k = bisect.bisect_right(tails, v)
        if k == len(tails):
            tails.append(v)
            ends.append(i)
        else:
            tails[k] = v
            ends[k] = i
        prev[i] = ends[k - 1] if k else -1
    out: list[tuple[float, int]] = []
    i = ends[-1] if ends else -1
    while i >= 0:
        out.append(reads[i])
        i = prev[i]
    return out[::-1]


def _runs(reads: list[tuple[float, int]]) -> list[list]:
    """``[value, first_t, last_t, n]`` per value.

    A value read fewer than MIN_READS times only counts when it fills the +1
    step between its neighbours exactly: that keeps a first round whose 0 was
    hard to read and the final score, which is on screen only briefly before
    the end screen, but drops a lone misread that happens not to fall.
    """
    runs: list[list] = []
    for t, v in reads:
        if runs and runs[-1][0] == v:
            runs[-1][2] = t
            runs[-1][3] += 1
        else:
            runs.append([v, t, t, 1])
    strong = [r[3] >= MIN_READS for r in runs]
    keep: list[list] = []
    for i, r in enumerate(runs):
        if strong[i]:
            keep.append(r)
            continue
        before = next((runs[j][0] for j in range(i - 1, -1, -1) if strong[j]), None)
        after = next((runs[j][0] for j in range(i + 1, len(runs)) if strong[j]), None)
        if (before, after) != (None, None) and before in (None, r[0] - 1) \
                and after in (None, r[0] + 1):
            keep.append(r)
    return keep


def _split_matches(samples: list[BarSample]) -> list[list[BarSample]]:
    """Cut where the score is back at 0-0 after a match was under way, and at long gaps."""
    parts: list[list[BarSample]] = [[]]
    progressed = 0
    zeros: list[int] = []
    last_t: float | None = None
    for s in samples:
        if last_t is not None and s.t - last_t > MATCH_GAP_S and parts[-1]:
            parts.append([])
            progressed, zeros = 0, []
        last_t = s.t
        both = s.ally is not None and s.enemy is not None
        if both and s.ally + s.enemy >= RESET_MIN_SUM:
            progressed += 1
            zeros = []
        elif both and s.ally == 0 and s.enemy == 0 and progressed >= MIN_READS:
            zeros.append(len(parts[-1]))
            if len(zeros) >= MIN_READS:
                cut = zeros[0]
                parts.append(parts[-1][cut:])
                parts[-2] = parts[-2][:cut]
                progressed, zeros = 0, []
        parts[-1].append(s)
    return [p for p in parts if p]


def _decode_match(samples: list[BarSample]) -> list[RoundResult]:
    runs: dict[str, list[list]] = {}
    for side in ("ally", "enemy"):
        reads = [(s.t, getattr(s, side)) for s in samples if getattr(s, side) is not None]
        runs[side] = _runs(_monotone(reads))
    if not runs["ally"] or not runs["enemy"]:
        return []
    steps = []
    for side, rs in runs.items():
        for (v0, _f0, last0, _n0), (v1, first1, _l1, _n1) in zip(rs, rs[1:]):
            # A value that was never read hides one round: both are placed at
            # the first read of the higher value.
            for value in range(v0 + 1, v1 + 1):
                steps.append((first1, last0, side, value, v1 - v0 == 1))
    steps.sort(key=lambda st: (st[0], st[3]))
    score = {"ally": runs["ally"][0][0], "enemy": runs["enemy"][0][0]}
    out: list[RoundResult] = []
    for t, t_low, side, value, exact in steps:
        score[side] = value
        out.append(RoundResult(t=round(t, 3), winner=side,
                               score=(score["ally"], score["enemy"]),
                               t_low=round(t_low, 3), exact=exact))
    return out


def decode_rounds(samples: list[BarSample]) -> list[RoundResult]:
    """Round results from noisy score reads (see the module docstring)."""
    out: list[RoundResult] = []
    for part in _split_matches(samples):
        out.extend(_decode_match(part))
    out.sort(key=lambda r: r.t)
    return out


def spike_plants(samples: list[BarSample]) -> list[Span]:
    """``(first, last)`` sighting of each planted spike; single sightings are dropped."""
    runs: list[list[float]] = []
    for s in samples:
        if not s.spike:
            continue
        if runs and s.t - runs[-1][1] <= SPIKE_GAP_S:
            runs[-1][1] = s.t
            runs[-1][2] += 1
        else:
            runs.append([s.t, s.t, 1])
    return [(a, b) for a, b, n in runs if n >= 2]


def _result_for(rounds: list[RoundResult], t: float, own_start: float | None,
                nxt: float | None) -> RoundResult | None:
    """The score change that ended the round live at ``t``; None when unsure.

    It only counts when the old score was still read during this round (after
    its buy phase) and the change came before the next buy phase; otherwise
    the round stays unknown rather than getting a neighbouring round's result.
    """
    after = next((r for r in rounds if r.t >= t - POST_ROUND_S), None)
    if after is None or after.t > t + RESULT_WINDOW_S:
        return None
    if nxt is not None and after.t_low >= nxt:
        return None
    if own_start is not None and after.t_low < own_start - POST_ROUND_S:
        return None
    return after


def find_clutches(samples: list[BarSample], rounds: list[RoundResult],
                  deaths: list[float]) -> list[Clutch]:
    """Rounds in which the streamer was the last one alive, won or lost.

    Rounds are told apart by the portraits themselves: every buy phase brings
    the whole team back. (The buy-phase plate is no good here: the round-end
    banner in the same spot makes it fire 8-26 s early, still inside the old
    round.) One portrait left is the streamer only when they have not died
    since the whole team was last alive. The old score must still have been
    read after the streamer was left alone, so a teammate dying after the
    round was already decided is not taken for a clutch.
    """
    valid = [s for s in samples if s.alive_ally is not None and s.alive_enemy is not None]
    if not valid:
        return []
    full = max(s.alive_ally for s in valid)
    if full < 2:
        return []
    deaths = sorted(float(d) for d in deaths)
    out: list[Clutch] = []
    t_full: float | None = None
    judged = False                     # this round's last-one-alive moment was already looked at
    for i, a in enumerate(valid):
        if a.alive_ally >= full:
            t_full, judged = a.t, False
            continue
        if judged or t_full is None or i + 1 >= len(valid):
            continue
        b = valid[i + 1]
        if not (a.alive_ally == 1 and b.alive_ally == 1 and a.alive_enemy >= 1
                and b.t - a.t <= CLUTCH_CONFIRM_S):
            continue
        judged = True
        if any(t_full - DEATH_SLACK_S < d <= b.t for d in deaths):
            continue                       # the one left is a teammate
        nxt = next((s.t for s in valid[i + 1:] if s.alive_ally >= full), None)
        res = _result_for(rounds, a.t, t_full, nxt)
        if res is not None and res.t_low >= a.t - 2.0:
            out.append(Clutch(t=a.t, enemies=int(a.alive_enemy), t_end=res.t,
                              won=res.winner == "ally", t_live=t_full))
    return out


def annotate(result, rounds: list[RoundResult], plants: list[Span],
             clutches: list[Clutch] = ()) -> None:
    """Mark each kill round of a kill-feed ``result``: won / lost, after a
    plant, and the won clutch it belongs to (a kill while that round was on)."""
    import bisect

    starts = sorted(float(s) for s in result.diagnostics.get("round_starts", []))
    for rnd in result.rounds:
        t_first, t_last = float(rnd.t_first), float(rnd.t_last)
        i = bisect.bisect_right(starts, t_first)
        own_start = starts[i - 1] if i else None
        nxt = next((s for s in starts if s > t_last), None)
        after = _result_for(rounds, t_last, own_start, nxt)
        rnd.won = (after.winner == "ally") if after is not None else None
        rnd.after_plant = any(a - 2.0 <= float(k.t) <= b + 8.0 for a, b in plants for k in rnd.kills)
        rnd.clutch = None
    rows = []
    for c in clutches:
        match = None
        if c.won:
            match = next((rnd for rnd in result.rounds
                          if any(c.t_live <= float(k.t) <= c.t_end + POST_ROUND_S
                                 for k in rnd.kills)), None)
        if match is not None:
            match.clutch, match.won = c, True
        rows.append({"t": c.t, "t_end": c.t_end, "enemies": c.enemies, "won": c.won,
                     "round": match.index if match is not None else None})
    result.diagnostics["round_results"] = [
        {"t": r.t, "t_low": r.t_low, "winner": r.winner, "score": list(r.score),
         "exact": r.exact} for r in rounds]
    result.diagnostics["spike_plants"] = [[round(a, 1), round(b, 1)] for a, b in plants]
    result.diagnostics["clutches"] = rows


def enrich(src_path: str, info, result, spans: list[Span] | None = None, **kw
           ) -> tuple[list[RoundResult], list[Span]]:
    """Read the top bar and annotate ``result``; never raises."""
    started = time.monotonic()
    try:
        samples = read_bar(src_path, info, spans=spans, **kw)
        rounds, plants = decode_rounds(samples), spike_plants(samples)
        clutches = find_clutches(samples, rounds, [float(d.t) for d in result.deaths])
        annotate(result, rounds, plants, clutches)
    except Exception as e:
        log.warning("valorant rounds: top bar could not be read: %s", e)
        return [], []
    final = rounds[-1].score if rounds else None
    log.info("valorant rounds: %d result(s), last score %s, %d spike plant(s), %d clutch(es) "
             "(%d won), read in %.0fs", len(rounds), final, len(plants), len(clutches),
             sum(c.won for c in clutches), time.monotonic() - started)
    return rounds, plants

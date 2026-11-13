"""Valorant round results and spike plants, read off the top bar.

The top bar shows the own team's score left of the round timer and the
enemy's right of it, as white digits on translucent plates. When the spike is
planted the timer is replaced by a red spike icon.

* **Scores** are read with the digit OCR on every keyframe (about every 4 s).
  Reads on the translucent plates are noisy (measured: 40-60 % come back
  empty, a few gain or lose a digit, e.g. 3 read as 13). A score never falls
  within a match, so the longest series of reads that never falls is kept and
  everything else is a misread; a value needs two reads, or one read that
  fills a +1 step exactly.
* **Spike plants** need the red icon on two keyframes in a row.

Checked on a 40-minute match that ended 8-13: the 20 score changes up to 8-12
decode in order (the final 13 never shows before the end screen; spot-checked
by eye at 1:40, 3:20 and 5:00), and 10 plants are found.
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
    """One keyframe pass over the top bar; scores are read inside ``spans`` only."""
    width, height = int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return []
    if reader is None:
        from .readers import pick_readers

        reader = pick_readers()[0]
    region = bar_region(width, height)
    if frame_source is None:
        from . import scan

        def frame_source(reg, **kw):
            return scan.iter_frames(src_path, reg, **kw)

    ally = _cols(region[2], height, -SCORE_DX, SCORE_HALF)
    enemy = _cols(region[2], height, SCORE_DX, SCORE_HALF)
    out: list[BarSample] = []
    n = 0
    for t, frame in frame_source(region, keyframes=True, pix_fmt="bgr24"):
        t = float(t)
        if spans and not any(a <= t <= b for a, b in spans):
            continue
        a = e = None
        if reader is not None and n % OCR_EVERY == 0:
            a, e = _read_score(reader, frame[:, ally]), _read_score(reader, frame[:, enemy])
        n += 1
        out.append(BarSample(t=t, ally=a, enemy=e, spike=spike_icon(frame, height)))
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


def annotate(result, rounds: list[RoundResult], plants: list[Span]) -> None:
    """Mark each kill round of a kill-feed ``result`` as won / lost / after a plant.

    A round's result is the first score change after its last kill. It only
    counts when the old score was still read during this round (after its buy
    phase) and the change came before the next buy phase; otherwise the round
    stays unknown rather than getting a neighbouring round's result.
    """
    import bisect

    starts = sorted(float(s) for s in result.diagnostics.get("round_starts", []))
    for rnd in result.rounds:
        t_first, t_last = float(rnd.t_first), float(rnd.t_last)
        i = bisect.bisect_right(starts, t_first)
        own_start = starts[i - 1] if i else None
        nxt = next((s for s in starts if s > t_last), None)
        after = next((r for r in rounds if r.t >= t_last - POST_ROUND_S), None)
        ok = (after is not None and after.t <= t_last + RESULT_WINDOW_S
              and (nxt is None or after.t_low < nxt)
              and (own_start is None or after.t_low >= own_start - POST_ROUND_S))
        rnd.won = (after.winner == "ally") if ok else None
        rnd.after_plant = any(a - 2.0 <= float(k.t) <= b + 8.0 for a, b in plants for k in rnd.kills)
    result.diagnostics["round_results"] = [
        {"t": r.t, "t_low": r.t_low, "winner": r.winner, "score": list(r.score),
         "exact": r.exact} for r in rounds]
    result.diagnostics["spike_plants"] = [[round(a, 1), round(b, 1)] for a, b in plants]


def enrich(src_path: str, info, result, spans: list[Span] | None = None, **kw
           ) -> tuple[list[RoundResult], list[Span]]:
    """Read the top bar and annotate ``result``; never raises."""
    started = time.monotonic()
    try:
        samples = read_bar(src_path, info, spans=spans, **kw)
        rounds, plants = decode_rounds(samples), spike_plants(samples)
        annotate(result, rounds, plants)
    except Exception as e:
        log.warning("valorant rounds: top bar could not be read: %s", e)
        return [], []
    final = rounds[-1].score if rounds else None
    log.info("valorant rounds: %d result(s), last score %s, %d spike plant(s), read in %.0fs",
             len(rounds), final, len(plants), time.monotonic() - started)
    return rounds, plants

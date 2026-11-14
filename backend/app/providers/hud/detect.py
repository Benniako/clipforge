"""Scoreboard tracker — read the score off the HUD and report every goal.

The pipeline, end to end:

1. **Scan.** Decode only the scoreboard region, keyframes only
   (:mod:`.scan`). A four-hour recording becomes a few thousand tiny images.
2. **Gate.** Keep the samples whose pixels have a scoreboard's structure
   (:func:`.glyphs.structure_ok`). Menus, replays and cutscenes drop out here.
3. **Cluster and label.** Binarise both score cells, group identical glyphs,
   give each group its digit (:mod:`.labeling`).
4. **Track.** Turn the per-sample scores into matches and goals
   (:mod:`.tracker`).
5. **Refine.** For the goals that matter, re-read a few seconds at 5 fps to
   find the moment the digit changed, and sweep the last seconds of every
   match for a goal the coarse pass could not have seen.
6. **Describe.** Read the match clock at each goal, work out which side is
   the streamer's, and rate how much each goal mattered.

The result carries no clips and no scores out of 100 — turning goals into
clips is the gameplay detector's job (``detect_gameplay``).
"""
from __future__ import annotations

import logging
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import glyphs as G
from .labeling import label_clusters
from .layouts import HudLayout, layouts_for
from .readers import CLOCK_CHARS, match_minute, parse_clock, pick_readers
from .segments import cluster_spans
from .tracker import Goal, Sample, Segment, covered, in_play_intervals, step_team, track

log = logging.getLogger("clipforge.hud")

MAX_KEYFRAME_GAP_S = 8.0     # sparser keyframes than this -> fall back to a timed decode
FALLBACK_FPS = 0.5
DENSE_FPS = 5.0
MIN_CONFIRM_FRAMES = 3       # dense frames that must show a new score to confirm it
BRIDGE_S = 45.0              # scoreboard absences shorter than this stay "in play"
MIN_HEALTHY_SECONDS = 60.0

FrameSource = Callable[..., Iterator[tuple[float, object]]]


@dataclass
class HudEvent:
    """One goal read off the scoreboard."""
    t: float                               # estimated moment of the goal (s, source timeline)
    team: str                              # "a" = first team on the scoreboard, "b" = second
    score_before: tuple[int, int]
    score_after: tuple[int, int]
    t_change: float                        # when the new score was first on screen
    window: tuple[float, float]            # coarse bracket: last old reading, first new reading
    hidden: bool = False                   # scoreboard was off screen around the goal
    confidence: float = 0.95
    label: str = "goal"
    minute: int | None = None              # football minute (12:29 -> 13)
    clock: str = ""                        # clock text as read, e.g. "12:29" or "90:56+2"
    inferred: bool = False                 # one of two goals that arrived between two readings
    refined: bool = False                  # timing pinned down by the 5 fps pass
    streamer: bool | None = None           # the streamer's team scored (None = unknown)
    importance: float = 0.0                # how much the goal mattered, in score points
    reasons: list[str] = field(default_factory=list)

    @property
    def detail(self) -> str:
        minute = f" ({self.minute}')" if self.minute is not None else ""
        return (f"Scoreboard {self.score_before[0]}-{self.score_before[1]} -> "
                f"{self.score_after[0]}-{self.score_after[1]}{minute}")


@dataclass
class MatchInfo:
    t_start: float
    t_end: float
    start_score: tuple[int, int]
    final_score: tuple[int, int]
    goals: int
    # When play started: the first frame with the scoreboard on screen. The
    # scoreboard appears at kick-off (its clock reads 00:00), so this is up to a
    # coarse sample earlier than ``t_start``. None when it was not measured.
    kickoff: float | None = None

    @property
    def play_start(self) -> float:
        return self.kickoff if self.kickoff is not None else self.t_start


@dataclass
class HudResult:
    layout: str
    events: list[HudEvent]
    matches: list[MatchInfo]
    in_play: list[tuple[float, float]]     # spans with the scoreboard on screen (short gaps bridged)
    coverage: float                        # share of the video those spans cover, 0..1
    healthy: bool                          # the scoreboard was found and read reliably
    samples: int
    sample_interval: float
    streamer_side: str | None = None
    diagnostics: dict = field(default_factory=dict)
    # Stretches of the recording that show this game at all (matches plus the
    # menus between them). Empty means "the whole recording".
    game_spans: list[tuple[float, float]] = field(default_factory=list)

    def is_in_play(self, t: float, *, pad: float = 2.0) -> bool:
        return covered(self.in_play, t, pad=pad)

    def allows(self, t: float, *, pad: float = 2.0) -> bool:
        """False only for moments of this game with no scoreboard on screen.

        A moment outside ``game_spans`` belongs to another game in the same
        recording; this tracker has no say about it.
        """
        if self.game_spans and not covered(self.game_spans, t):
            return True
        return self.is_in_play(t, pad=pad)

    @property
    def game_coverage(self) -> float:
        """Share of this game's own stretches with the scoreboard on screen."""
        total = sum(b - a for a, b in self.game_spans)
        if total <= 0:
            return self.coverage
        return min(sum(b - a for a, b in self.in_play) / total, 1.0)


@dataclass
class _Boxes:
    score: dict[str, tuple[int, int, int, int]]
    clock: tuple[int, int, int, int]
    rows: tuple[int, int, int, int]
    name: dict[str, tuple[int, int, int, int] | None]


@dataclass
class _Pass:
    """Per-sample readings from one scan of the region."""
    times: list[float] = field(default_factory=list)
    present: list[bool] = field(default_factory=list)
    glyph: dict[str, list] = field(default_factory=lambda: {"a": [], "b": []})
    clock: list = field(default_factory=list)
    names: dict[str, list] = field(default_factory=lambda: {"a": [], "b": []})


def _boxes(lay: HudLayout, height: int, region: tuple[int, int, int, int]) -> _Boxes:
    rel = lambda c: lay.cell_in_region(c, height, region)  # noqa: E731
    return _Boxes(score={"a": rel(lay.score_a), "b": rel(lay.score_b)},
                  clock=rel(lay.clock), rows=rel(lay.rows),
                  name={"a": rel(lay.name_a) if lay.name_a else None,
                        "b": rel(lay.name_b) if lay.name_b else None})


def _read_frames(frames: Iterator[tuple[float, object]], boxes: _Boxes, lay: HudLayout,
                 *, with_names: bool) -> _Pass:
    out = _Pass()
    for t, gray in frames:
        ok = G.structure_ok(G.structure_levels(gray, boxes.rows, boxes.clock),
                            lay.min_structure_contrast)
        out.times.append(float(t))
        out.present.append(bool(ok))
        for side in ("a", "b"):
            out.glyph[side].append(G.extract_glyph(G.crop(gray, boxes.score[side])) if ok else None)
            box = boxes.name[side]
            out.names[side].append(G.signature(G.crop(gray, box)) if (ok and with_names and box) else None)
        out.clock.append(G.crop(gray, boxes.clock).copy() if ok else None)
    return out


def _median_interval(times: list[float]) -> float:
    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    return gaps[len(gaps) // 2] if gaps else 0.0


def _default_source(src_path: str) -> FrameSource:
    from . import scan

    def source(region, **kw):
        return scan.iter_frames(src_path, region, **kw)

    return source


def _has_nvidia() -> bool:
    try:
        from ...config import get_settings

        return bool(get_settings().has_nvidia)
    except Exception:
        return False


def track_scoreboard(src_path: str, info, settings=None, *, layout: HudLayout | None = None,
                     readers: tuple | None = None, frame_source: FrameSource | None = None,
                     refine_limit: int | None = 40, max_goal_gap_s: float = 90.0
                     ) -> HudResult | None:
    """Track the scoreboard of ``src_path`` and return its goals.

    ``info`` needs ``width``, ``height`` and ``duration``. The layout comes
    from ``layout`` or from ``settings.game_profile``. ``readers`` is a
    ``(digit_reader, line_reader)`` pair (default: whatever OCR is installed;
    pass ``(None, None)`` to use the game rule alone). ``frame_source`` replaces
    ffmpeg in tests. ``refine_limit`` caps how many goals get the 5 fps timing
    pass (``None`` = all of them).

    Returns ``None`` when no layout applies or the video cannot be read; a
    result with ``healthy=False`` when the scoreboard was not found.
    """
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except Exception as e:
        log.info("hud: OpenCV/NumPy unavailable, scoreboard tracking skipped (%s)", e)
        return None
    width, height = int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    if width <= 0 or height <= 0 or duration <= 0:
        return None
    candidates = [layout] if layout is not None else layouts_for(getattr(settings, "game_profile", None))
    if not candidates:
        return None
    source = frame_source or _default_source(str(src_path))
    best: HudResult | None = None
    for lay in candidates:
        try:
            res = _track_layout(lay, source, width, height, duration, readers,
                                refine_limit, max_goal_gap_s)
        except Exception as e:
            log.warning("hud: tracking with layout %s failed: %s", lay.name, e)
            res = None
        if res is not None and (best is None or res.coverage > best.coverage):
            best = res
    return best


def _track_layout(lay: HudLayout, source: FrameSource, width: int, height: int, duration: float,
                  readers: tuple | None, refine_limit: int | None, max_goal_gap_s: float
                  ) -> HudResult | None:
    started = time.monotonic()
    region = lay.region_px(width, height)
    boxes = _boxes(lay, height, region)

    coarse = _read_frames(source(region, keyframes=True), boxes, lay, with_names=True)
    interval = _median_interval(coarse.times)
    mode = "keyframes"
    if len(coarse.times) < 3 or interval > MAX_KEYFRAME_GAP_S:
        # Too few keyframes to follow a score: decode at a fixed rate instead.
        coarse = _read_frames(source(region, keyframes=False, fps=FALLBACK_FPS, hwaccel=_has_nvidia()),
                              boxes, lay, with_names=True)
        interval = _median_interval(coarse.times)
        mode = "fps"
    n = len(coarse.times)
    if n == 0:
        return None
    interval = interval or 1.0

    # Cluster the score glyphs.
    clusters = G.GlyphClusters()
    seqs: dict[str, list[int]] = {"a": [-1] * n, "b": [-1] * n}
    for i in range(n):
        for side in ("a", "b"):
            bm = coarse.glyph[side][i]
            if bm is not None:
                seqs[side][i] = clusters.add(bm, (i, side))

    n_present = sum(coarse.present)
    digit_reader = line_reader = None
    if n_present:
        digit_reader, line_reader = readers if readers is not None else pick_readers()
    label_diag = label_clusters(clusters.clusters, seqs, coarse.times,
                                lambda ref: coarse.glyph[ref[1]][ref[0]], digit_reader,
                                max_goal_gap_s=max_goal_gap_s)

    def value(side: str, i: int) -> int | None:
        cid = seqs[side][i]
        return clusters.clusters[cid].label if cid >= 0 else None

    samples = [Sample(t=coarse.times[i], present=coarse.present[i], a=value("a", i), b=value("b", i))
               for i in range(n)]

    # Another game in the same recording can pass the structure test without
    # ever showing a readable score (Valorant's corner HUD did so for 13 of 16
    # minutes). A span like that is not this game: its samples are taken out
    # before anything is tracked or judged.
    foreign = 0
    for a, b in in_play_intervals(samples, bridge_s=BRIDGE_S):
        idx = [i for i, s in enumerate(samples) if s.present and a <= s.t <= b]
        if sum(1 for i in idx if samples[i].score is not None) < 0.5 * len(idx):
            foreign += len(idx)
            for i in idx:
                samples[i].present = False
                coarse.present[i] = False
    n_present = sum(coarse.present)
    segments = track(samples, max_goal_gap_s=max_goal_gap_s)

    in_play = in_play_intervals(samples, bridge_s=BRIDGE_S, pad_s=interval)
    in_play = [(a, min(b, duration)) for a, b in in_play]
    in_play_s = sum(b - a for a, b in in_play)
    decoded = sum(1 for s in samples if s.score is not None)
    healthy = (in_play_s >= MIN_HEALTHY_SECONDS and n_present >= 10
               and decoded >= 0.8 * n_present)

    dense_calls = 0

    def dense(t0: float, t1: float) -> list[tuple[float, bool, int | None, int | None, object]]:
        """Readings at 5 fps for [t0, t1]: (t, present, a, b, clock strip)."""
        nonlocal dense_calls
        t0 = max(t0, 0.0)
        t1 = min(t1, duration)
        if t1 - t0 < 0.2:
            return []
        dense_calls += 1
        try:
            p = _read_frames(source(region, keyframes=False, fps=DENSE_FPS, start=t0, duration=t1 - t0),
                             boxes, lay, with_names=False)
        except Exception as e:
            log.debug("hud: dense read %.1f-%.1f failed: %s", t0, t1, e)
            return []
        rows = []
        for k in range(len(p.times)):
            ga, gb = p.glyph["a"][k], p.glyph["b"][k]
            rows.append((p.times[k], p.present[k],
                         clusters.classify(ga) if ga is not None else None,
                         clusters.classify(gb) if gb is not None else None,
                         p.clock[k]))
        return rows

    # A goal in the last seconds of a match may never reach a coarse sample.
    tail_goals: dict[int, tuple[float, float]] = {}       # id(goal) -> (t_change, t_old)
    tail_clock: dict[int, object] = {}
    if healthy:
        for seg in segments:
            found = _tail_sweep(seg, dense(seg.t_end - 0.4, seg.t_end + interval + 0.6))
            if found is not None:
                goal, t_change, t_old, strip = found
                seg.goals.append(goal)
                seg.final_score = goal.after
                tail_goals[id(goal)] = (t_change, t_old)
                tail_clock[id(goal)] = strip

    # Kick-off per match: walk back over coarse samples that still show the
    # scoreboard (digits unreadable or not), then find its first frame in the
    # gap before. Clips of early goals start there instead of in the intro.
    kickoffs: list[float | None] = []
    for seg in segments:
        i = seg.i_first
        while i > 0 and coarse.present[i - 1] and coarse.times[i] - coarse.times[i - 1] <= 1.5 * interval:
            i -= 1
        t_seen = coarse.times[i]
        t_before = coarse.times[i - 1] if i > 0 else max(t_seen - interval, 0.0)
        kick = max(t_before, t_seen - 2.0 * interval)  # never more than the gap before
        if healthy:
            rows = dense(kick - 0.2, t_seen + 0.2)
            first = next((row[0] for row in rows if row[1]), None)
            if first is not None:
                kick = first
        kickoffs.append(round(min(kick, t_seen), 3))

    # Clock and minute per goal.
    clock_cache: dict[int, tuple[int, int, int] | None] = {}

    def read_clock(strip) -> tuple[int, int, int] | None:
        if line_reader is None or strip is None:
            return None
        try:
            text, _conf = line_reader.read_line(G.line_image(strip), CLOCK_CHARS)
        except Exception as e:
            log.debug("hud: clock read failed: %s", e)
            return None
        return parse_clock(text)

    def clock_at(i: int) -> tuple[int, int, int] | None:
        if i < 0 or i >= n:
            return None
        if i not in clock_cache:
            clock_cache[i] = read_clock(coarse.clock[i])
        return clock_cache[i]

    streamer_side = _streamer_side(coarse, segments)

    def set_clock(ev: HudEvent, clock: tuple[int, int, int] | None) -> None:
        if clock is None:
            return
        ev.minute = match_minute(clock)
        ev.clock = "%02d:%02d" % (clock[0], clock[1]) + (f"+{clock[2]}" if clock[2] else "")

    events: list[tuple[HudEvent, Goal, Segment]] = []
    deficits: list[int] = []            # per event: worst deficit the scoring team had faced
    for seg in segments:
        deficit = {"a": 0, "b": 0}      # worst deficit each team has faced so far in this match
        deficit["a"] = max(0, seg.start_score[1] - seg.start_score[0])
        deficit["b"] = max(0, seg.start_score[0] - seg.start_score[1])
        for goal in seg.goals:
            if id(goal) in tail_goals:
                t_change, _t_old = tail_goals[id(goal)]
                clock = read_clock(tail_clock.get(id(goal)))
                ev = HudEvent(t=max(t_change - lay.score_delay_s, 0.0), team=goal.team,
                              score_before=goal.before, score_after=goal.after, t_change=t_change,
                              window=(goal.lo, goal.hi), confidence=0.85, refined=True)
            else:
                clock = None
                for i in (goal.i_hi, goal.i_hi + 1, bisect_right(coarse.times, goal.lo) - 1):
                    clock = clock_at(i)
                    if clock is not None:
                        break
                ev = HudEvent(t=_coarse_time(goal, interval, lay), team=goal.team,
                              score_before=goal.before, score_after=goal.after, t_change=goal.hi,
                              window=(goal.lo, goal.hi), hidden=goal.hidden, inferred=goal.inferred,
                              confidence=0.6 if (goal.inferred or goal.tentative) else 0.95)
            if clock is not None:
                set_clock(ev, clock)
            elif line_reader is not None:
                ev.confidence = round(max(ev.confidence - 0.1, 0.3), 2)
            if streamer_side is not None:
                ev.streamer = ev.team == streamer_side
            ev.importance, ev.reasons = goal_importance(ev, max_deficit=deficit[ev.team])
            deficits.append(deficit[ev.team])
            for team in ("a", "b"):
                own, other = (0, 1) if team == "a" else (1, 0)
                deficit[team] = max(deficit[team], goal.after[other] - goal.after[own])
            events.append((ev, goal, seg))

    # Timing pass for the goals most likely to become clips, and for every
    # goal whose new score was seen in a single coarse sample.
    order = sorted(range(len(events)), key=lambda k: events[k][0].importance, reverse=True)
    budget = len(order) if refine_limit is None else max(int(refine_limit), 0)
    chosen = set(order[:budget]) | {k for k, (_e, g, _s) in enumerate(events) if g.tentative}
    dropped: set[int] = set()
    for k in sorted(chosen):
        ev, goal, _seg = events[k]
        if ev.refined or goal.inferred:
            continue
        hi_end = (goal.hi + interval) if goal.tentative else goal.hi
        rows = dense(goal.lo - 0.6, min(hi_end, goal.lo + 16.0) + 0.6)
        if not rows:
            continue
        t_change, t_old, t_hidden, n_new = _locate_change(goal, rows)
        if goal.tentative:
            if n_new < MIN_CONFIRM_FRAMES:
                dropped.add(k)
                continue
            ev.confidence = 0.9
        if t_change is not None and t_old is not None and (t_change - t_old) <= 3.0:
            ev.t, ev.t_change, ev.hidden, ev.refined = max(t_change - lay.score_delay_s, 0.0), t_change, False, True
        elif t_hidden is not None:
            ev.t, ev.hidden, ev.refined = max(t_hidden - lay.cut_delay_s, 0.0), True, True
            if t_change is not None:
                ev.t_change = t_change
        elif t_change is not None:
            ev.t, ev.t_change, ev.refined = max(t_change - lay.score_delay_s, 0.0), t_change, True
        if ev.refined:
            # Read the clock at the goal moment itself. The coarse sample can
            # be half a match-minute late, which is enough to report a goal
            # scored at 38:55 as the 40th minute instead of the 39th.
            near = [row for row in rows if row[1] and row[4] is not None and row[0] <= ev.t_change]
            if near:
                strip = min(near, key=lambda row: abs(row[0] - ev.t))[4]
                before = ev.minute
                set_clock(ev, read_clock(strip))
                if ev.minute != before:
                    ev.importance, ev.reasons = goal_importance(ev, max_deficit=deficits[k])

    final = sorted((ev for k, (ev, _g, _s) in enumerate(events) if k not in dropped),
                   key=lambda e: e.t)
    matches = [MatchInfo(t_start=s.t_start, t_end=s.t_end, start_score=s.start_score,
                         final_score=s.final_score, goals=len(s.goals), kickoff=k)
               for s, k in zip(segments, kickoffs)]
    diagnostics = {
        "mode": mode,
        "region": list(region),
        "present_samples": int(n_present),
        "foreign_samples": int(foreign),
        "decoded_samples": int(decoded),
        "clusters": len(clusters.clusters),
        "digit_reader": getattr(digit_reader, "name", None),
        "line_reader": getattr(line_reader, "name", None),
        "dense_reads": dense_calls,
        "tentative_dropped": len(dropped),
        "elapsed_s": round(time.monotonic() - started, 2),
        **label_diag,
    }
    return HudResult(layout=lay.name, events=final, matches=matches, in_play=in_play,
                     coverage=min(in_play_s / duration, 1.0) if duration > 0 else 0.0,
                     healthy=bool(healthy), samples=n, sample_interval=float(interval),
                     streamer_side=streamer_side, diagnostics=diagnostics,
                     game_spans=cluster_spans(in_play, duration))


def _coarse_time(goal: Goal, interval: float, lay: HudLayout) -> float:
    """Best guess at the goal moment from the coarse bracket alone."""
    if goal.hidden and (goal.hi - goal.lo) > 2.5 * interval:
        # The scoreboard vanished right after the last old reading: the goal
        # was scored just before that.
        return max(goal.lo + 0.5 * interval - lay.cut_delay_s, 0.0)
    return max(goal.hi - 0.5 * interval - lay.score_delay_s, 0.0)


def _locate_change(goal: Goal, rows: list) -> tuple[float | None, float | None, float | None, int]:
    """Find the digit change inside dense readings.

    Returns ``(t_change, t_old, t_hidden, n_new)``: the first frame showing
    the new digit (followed by agreeing frames), the last frame still showing
    the old one, the first frame after it where the scoreboard was gone, and
    how many frames showed the new digit.
    """
    side = 0 if goal.team == "a" else 1
    new_v, old_v = goal.after[side], goal.before[side]

    def val(row) -> int | None:
        return (row[2], row[3])[side] if row[1] else None

    t_change = None
    for k, row in enumerate(rows):
        if val(row) != new_v:
            continue
        later = [val(r) for r in rows[k + 1:] if val(r) is not None]
        if not later or later[0] == new_v:
            t_change = row[0]
            break
    olds = [row[0] for row in rows if val(row) == old_v and (t_change is None or row[0] < t_change)]
    t_old = max(olds) if olds else None
    t_hidden = None
    if t_old is not None:
        gone = [row[0] for row in rows
                if row[0] > t_old and not row[1] and (t_change is None or row[0] < t_change)]
        t_hidden = min(gone) if gone else None
    n_new = sum(1 for row in rows if val(row) == new_v)
    return t_change, t_old, t_hidden, n_new


def _tail_sweep(seg: Segment, rows: list) -> tuple[Goal, float, float, object] | None:
    """A goal in the final seconds of a match, seen only in dense readings."""
    if not rows:
        return None
    final = seg.final_score
    counts: dict[str, list[int]] = {"a": [], "b": []}
    for k, row in enumerate(rows):
        if not row[1] or row[2] is None or row[3] is None:
            continue
        team = step_team(final, (row[2], row[3]))
        if team is not None:
            counts[team].append(k)
    team = max(counts, key=lambda tm: len(counts[tm]))
    hits = counts[team]
    if len(hits) < MIN_CONFIRM_FRAMES:
        return None
    k0 = hits[0]
    olds = [row[0] for row in rows[:k0] if row[1] and (row[2], row[3]) == final]
    t_old = max(olds) if olds else seg.t_end
    after = (final[0] + 1, final[1]) if team == "a" else (final[0], final[1] + 1)
    goal = Goal(team=team, before=final, after=after, lo=t_old, hi=rows[k0][0])
    return goal, rows[k0][0], t_old, rows[hits[-1]][4]


def _streamer_side(coarse: _Pass, segments: list[Segment]) -> str | None:
    """Which scoreboard row belongs to the streamer.

    Opponents change from match to match; the streamer's own team name does
    not. So the name on the streamer's side looks alike in any two matches,
    while the other side's rarely does. Decided on the median similarity over
    all pairs of matches; needs at least two matches, ``None`` when unclear.
    """
    import numpy as np

    alike: dict[str, float] = {}
    for side in ("a", "b"):
        per_match = []
        for seg in segments[:40]:
            sigs = [coarse.names[side][i] for i in range(seg.i_first, min(seg.i_last + 1, len(coarse.times)))
                    if coarse.names[side][i] is not None]
            if len(sigs) >= 3:
                per_match.append((np.mean(np.stack(sigs), axis=0) >= 0.5).astype(np.uint8))
        if len(per_match) < 2:
            return None
        sims = sorted(G.glyph_iou(x, y) for i, x in enumerate(per_match) for y in per_match[i + 1:])
        alike[side] = sims[len(sims) // 2]
    for side, other in (("b", "a"), ("a", "b")):
        if alike[side] >= 0.75 and alike[other] <= 0.6 and (alike[side] - alike[other]) >= 0.2:
            return side
    return None


def goal_importance(ev: HudEvent, *, max_deficit: int = 0) -> tuple[float, list[str]]:
    """How much a goal mattered, in score points, with the reasons.

    Deliberately simple and explainable: what the goal did to the scoreline,
    how late it came, whether it capped a comeback, and whose goal it was.
    ``max_deficit`` is the largest deficit the scoring team faced earlier in
    the match.
    """
    own = 0 if ev.team == "a" else 1
    before = ev.score_before[own] - ev.score_before[1 - own]
    after = before + 1
    points = 0.0
    reasons: list[str] = []
    if after == 0:
        points += 5
        reasons.append("Equaliser")
    elif after == 1:
        points += 5
        reasons.append("Go-ahead goal")
    elif after >= 3:
        points -= 4
        reasons.append("Extends a big lead")
    late = ev.minute is not None and ev.minute >= 85
    if late:
        points += 3
        reasons.append(f"Late goal ({ev.minute}')")
        if after in (0, 1):
            points += 4
    if after in (0, 1) and max_deficit >= 2:
        points += 4
        reasons.append(f"Comeback from {max_deficit} down")
    if ev.streamer is True:
        points += 4
        reasons.append("Streamer's team scores")
    elif ev.streamer is False:
        points -= 3
        reasons.append("Opponent scores")
    return points, reasons

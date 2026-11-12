"""From per-sample scoreboard readings to goals and match segments.

Pure Python, no image or OCR dependencies: the input is a time-ordered list of
:class:`Sample` (was the scoreboard on screen, and what did it read), the
output is the list of goals and the matches they belong to.

The rules encode how a football scoreboard behaves:

* a score only ever goes up, one goal at a time;
* a new score that is real stays on screen, a misread does not;
* when a celebration or replay hides the scoreboard, a goal scored just
  before shows up as "+1" once it returns;
* anything else (both digits dropping to 0, a jump of several goals, a long
  break) is a new match, not a goal.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Sample:
    t: float
    present: bool                 # scoreboard structure visible in this frame
    a: int | None = None          # first team's score, None when unreadable
    b: int | None = None

    @property
    def score(self) -> tuple[int, int] | None:
        if self.present and self.a is not None and self.b is not None:
            return self.a, self.b
        return None


@dataclass
class Run:
    """A stretch of samples all reading the same score."""
    score: tuple[int, int]
    i_first: int
    i_last: int
    t_first: float
    t_last: float
    n: int = 1
    tentative: bool = False       # seen in one sample only; needs confirming


@dataclass
class Goal:
    team: str                     # "a" (first team on the scoreboard) or "b"
    before: tuple[int, int]
    after: tuple[int, int]
    lo: float                     # last sample still showing the old score
    hi: float                     # first sample showing the new score
    i_hi: int = -1                # index of that first sample
    hidden: bool = False          # scoreboard was off screen in between
    inferred: bool = False        # two goals arrived between two readings
    tentative: bool = False       # the new score was seen in a single sample


@dataclass
class Segment:
    """One match: a monotonic run of scores."""
    t_start: float
    t_end: float
    start_score: tuple[int, int]
    final_score: tuple[int, int]
    i_first: int = 0
    i_last: int = 0
    goals: list[Goal] = field(default_factory=list)


def step_team(prev: tuple[int, int], cur: tuple[int, int]) -> str | None:
    """Which team scored if ``cur`` is exactly one goal ahead of ``prev``."""
    if cur == (prev[0] + 1, prev[1]):
        return "a"
    if cur == (prev[0], prev[1] + 1):
        return "b"
    return None


def build_runs(samples: list[Sample], *, max_break_gap_s: float = 240.0) -> list[Run]:
    """Group consecutive readable samples by score.

    Unreadable and absent samples never split a run on their own; only a
    different score does — or the same score returning after a long break,
    which is treated as a new stretch so two matches are not glued together.
    """
    runs: list[Run] = []
    for i, s in enumerate(samples):
        sc = s.score
        if sc is None:
            continue
        last = runs[-1] if runs else None
        if last is not None and last.score == sc and (s.t - last.t_last) <= max_break_gap_s:
            last.i_last, last.t_last, last.n = i, s.t, last.n + 1
        else:
            runs.append(Run(score=sc, i_first=i, i_last=i, t_first=s.t, t_last=s.t))
    return runs


def _settle(runs: list[Run], *, max_goal_gap_s: float, max_break_gap_s: float) -> list[Run]:
    """Drop one-sample blips, keeping only those that could be a real goal.

    A score seen in a single sample is noise unless it is exactly one goal
    ahead of the score before it — a goal scored seconds before the final
    whistle leaves precisely that trace, and it is the kind of goal worth
    clipping. Such runs are kept as ``tentative`` for the caller to confirm.
    """
    kept: list[Run] = []
    for r in runs:
        if r.n >= 2:
            if (kept and kept[-1].tentative and len(kept) >= 2
                    and kept[-2].score == r.score
                    and (r.t_first - kept[-2].t_last) <= max_break_gap_s):
                kept.pop()                       # A, blip, A  ->  the blip was a misread
            if kept and kept[-1].score == r.score and (r.t_first - kept[-1].t_last) <= max_break_gap_s:
                last = kept[-1]
                last.i_last, last.t_last = r.i_last, r.t_last
                last.n += r.n
                last.tentative = False
            else:
                kept.append(r)
        elif (kept and not kept[-1].tentative
              and step_team(kept[-1].score, r.score) is not None
              and (r.t_first - kept[-1].t_last) <= max_goal_gap_s):
            r.tentative = True
            kept.append(r)
    return kept


def track(samples: list[Sample], *, max_goal_gap_s: float = 90.0,
          max_break_gap_s: float = 240.0) -> list[Segment]:
    """Segment the samples into matches and find every goal in them."""
    runs = _settle(build_runs(samples, max_break_gap_s=max_break_gap_s),
                   max_goal_gap_s=max_goal_gap_s, max_break_gap_s=max_break_gap_s)
    segments: list[Segment] = []
    seg: Segment | None = None
    prev: Run | None = None
    for r in runs:
        goals: list[Goal] = []
        new_match = seg is None or prev is None
        if not new_match:
            gap = r.t_first - prev.t_last
            hidden = any(not samples[i].present for i in range(prev.i_last + 1, r.i_first))
            skipped = (r.i_first - prev.i_last) > 1
            da, db = r.score[0] - prev.score[0], r.score[1] - prev.score[1]
            team = step_team(prev.score, r.score)
            if team is not None and gap <= max_goal_gap_s:
                goals.append(Goal(team=team, before=prev.score, after=r.score, lo=prev.t_last,
                                  hi=r.t_first, i_hi=r.i_first, hidden=hidden,
                                  tentative=r.tentative))
            elif (da >= 0 and db >= 0 and da + db == 2 and gap <= max_goal_gap_s
                  and (hidden or skipped)):
                # Two goals between two readings (the scoreboard was hidden or
                # unreadable for a while). The order is unknown for 1+1.
                first = "a" if da >= 1 else "b"
                mid = (prev.score[0] + 1, prev.score[1]) if first == "a" else (prev.score[0], prev.score[1] + 1)
                second = step_team(mid, r.score) or first
                for tm, before, after in ((first, prev.score, mid), (second, mid, r.score)):
                    goals.append(Goal(team=tm, before=before, after=after, lo=prev.t_last,
                                      hi=r.t_first, i_hi=r.i_first, hidden=hidden, inferred=True))
            else:
                new_match = True
        if new_match:
            seg = Segment(t_start=r.t_first, t_end=r.t_last, start_score=r.score,
                          final_score=r.score, i_first=r.i_first, i_last=r.i_last)
            segments.append(seg)
        else:
            seg.goals.extend(goals)
            seg.t_end, seg.final_score, seg.i_last = r.t_last, r.score, r.i_last
        prev = r
    return segments


def in_play_intervals(samples: list[Sample], *, bridge_s: float = 45.0,
                      pad_s: float = 0.0) -> list[tuple[float, float]]:
    """Time spans with the scoreboard on screen, bridging short absences.

    A celebration or replay hides the scoreboard for well under a minute; a
    trip through the menus takes longer. Bridging gaps up to ``bridge_s``
    keeps the former inside the match and the latter out of it.
    """
    out: list[list[float]] = []
    for s in samples:
        if not s.present:
            continue
        if out and (s.t - out[-1][1]) <= bridge_s:
            out[-1][1] = s.t
        else:
            out.append([s.t, s.t])
    return [(max(a - pad_s, 0.0), b + pad_s) for a, b in out]


def covered(intervals: list[tuple[float, float]], t: float, *, pad: float = 0.0) -> bool:
    """True when ``t`` falls inside one of the intervals (optionally padded)."""
    return any((a - pad) <= t <= (b + pad) for a, b in intervals)

"""Local auto B-roll — smart cutaways during voiceover.

When the speaker keeps talking but the frame is static (talking head holding
still), cutting to a *related* strong visual moment keeps the viewer's eye
moving. Commercial editors (OpusClip, Submagic) do this with stock libraries;
ClipForge does it *locally* by mining the same source video for its strongest
visual moments (scene cuts, high-motion regions) and inserting them as brief
picture-in-picture cutaways or hard cuts.

This module is the *selection* logic — it picks candidate B-roll windows from
the source. The actual composite (overlay or hard cut) is applied at render
time by the caller. Pure functions throughout, fully unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BRollWindow:
    """A candidate cutaway: a [start, end] span with a visual-strength score."""
    start: float
    end: float
    score: float          # 0..1, relative strength of this visual moment
    kind: str             # "cut" | "motion" | "zoom"  (where the moment came from)


def candidates_from_cuts(cuts: list[float], *, window: float = 1.2,
                         clip_end: float | None = None) -> list[BRollWindow]:
    """Each scene cut is a candidate cutaway of length ``window``.

    Cuts are the strongest visual punctuation; a 1.2s window after a cut is a
    natural B-roll beat. Score is uniform (1.0) — all cuts rank equally until
    merged with motion scores. Spans past ``clip_end`` are clamped/dropped.
    """
    out: list[BRollWindow] = []
    for t in cuts:
        if clip_end is not None and t >= clip_end:
            continue
        end = t + window
        if clip_end is not None:
            end = min(end, clip_end)
        if end - t < 0.3:
            continue
        out.append(BRollWindow(start=float(t), end=end, score=1.0, kind="cut"))
    return out


def candidates_from_motion(motion: list[tuple[float, float]],
                           *, window: float = 1.0,
                           threshold: float = 0.4) -> list[BRollWindow]:
    """High-motion spans are candidate B-roll (something visually happening).

    ``motion`` is a (t, intensity 0..1) series. Consecutive samples above
    ``threshold`` collapse into one window. Score is the peak intensity.
    """
    out: list[BRollWindow] = []
    run_start: float | None = None
    peak = 0.0
    prev_t: float | None = None
    for t, m in motion:
        if m >= threshold:
            if run_start is None or (prev_t is not None and t - prev_t > 0.5):
                if run_start is not None:
                    out.append(BRollWindow(start=run_start,
                                           end=min(prev_t or run_start, run_start + window),
                                           score=peak, kind="motion"))
                run_start = t
                peak = m
            else:
                peak = max(peak, m)
        else:
            if run_start is not None:
                out.append(BRollWindow(start=run_start,
                                       end=min(t, run_start + window),
                                       score=peak, kind="motion"))
                run_start = None
                peak = 0.0
        prev_t = t
    if run_start is not None:
        out.append(BRollWindow(start=run_start, end=run_start + window,
                               score=peak, kind="motion"))
    return out


def select_broll(candidates: list[BRollWindow], *,
                 gaps: list[tuple[float, float]],
                 max_per_clip: int = 2,
                 min_duration: float = 0.8) -> list[BRollWindow]:
    """Pick the best B-roll windows to fill ``gaps`` (static spans in the clip).

    A "gap" is a [start, end] span where the speaker talks but the frame is
    static — the prime cutaway opportunity. We rank candidates by score and
    assign at most ``max_per_clip`` total, never overlapping a gap's speaker.
    Returns the chosen windows sorted by time.
    """
    if not candidates or not gaps:
        return []
    # Rank globally; fill the earliest gap first (viewer attention is highest
    # at the start, so the strongest cutaway goes where it matters most).
    ranked = sorted(candidates, key=lambda c: -c.score)
    chosen: list[BRollWindow] = []
    filled_gaps = 0
    for gap_start, gap_end in sorted(gaps):
        if filled_gaps >= max_per_clip:
            break
        # Best candidate that fits inside this gap and isn't already used.
        for c in ranked:
            if c in chosen:
                continue
            if c.start >= gap_start and c.end <= gap_end and (c.end - c.start) >= min_duration:
                chosen.append(c)
                filled_gaps += 1
                break
    chosen.sort(key=lambda c: c.start)
    return chosen

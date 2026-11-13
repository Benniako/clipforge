"""Goals read off the scoreboard, grouped by match — for picking clips by hand.

The pipeline only renders the highest-ranked goals, but a football session
holds a hundred of them and which ones are worth posting is the streamer's
call. This module turns the goal events stored on a project into a list of
matches, first to last, each with its goals in order, and builds a clip for
any goal the user picks.

Pure functions over the project's events; no video access.
"""
from __future__ import annotations

from .models import Clip, ClipStatus, DetectedEvent, Project, ScoreFactor

# A goal's base score and floor, matching what the detector gives its own
# goal clips (detect_gameplay.HUD_GOAL_FLOOR and the 84-point base there).
_BASE_SCORE = 84.0
_SCORE_FLOOR = 80.0


def is_goal(e: DetectedEvent) -> bool:
    return (e.data or {}).get("detector") == "hud" and e.label == "goal"


def _pair(text) -> tuple[int, int] | None:
    try:
        a, b = str(text).split("-")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return None


def _clip_for(project: Project, t: float) -> Clip | None:
    """The gameplay clip that already shows the moment ``t``, if any."""
    hits = [c for c in project.clips if c.start <= t <= c.end]
    return min(hits, key=lambda c: abs((c.features or {}).get("climax_t", c.start) - t)) if hits else None


def goals_by_match(project: Project) -> list[dict]:
    """Matches in playing order, each with its goals in order.

    A new match starts wherever a goal does not continue the previous goal's
    score (the scoreboard went back to 0-0 in between).
    """
    matches: list[dict] = []
    last_after: tuple[int, int] | None = None
    for e in sorted((e for e in project.events if is_goal(e)), key=lambda e: e.t):
        d = e.data or {}
        before, after = _pair(d.get("score_before")), _pair(d.get("score_after"))
        if not matches or before is None or before != last_after:
            matches.append({"index": len(matches) + 1, "start": round(float(e.t), 3),
                            "end": round(float(e.t), 3), "final_score": "", "goals": []})
        m = matches[-1]
        clip = _clip_for(project, float(e.t))
        m["goals"].append({
            "t": round(float(e.t), 3),
            "team": d.get("team"),
            "score_before": d.get("score_before") or "",
            "score_after": d.get("score_after") or "",
            "minute": d.get("minute"),
            "clock": d.get("clock") or "",
            "streamer": d.get("streamer"),
            "importance": float(d.get("importance") or 0.0),
            "confidence": float(e.confidence),
            "clip_id": clip.id if clip else None,
            "clip_status": clip.status.value if clip else None,
        })
        m["end"] = round(float(e.t), 3)
        m["final_score"] = d.get("score_after") or ""
        last_after = after
    return matches


def find_goal(project: Project, t: float, *, tolerance: float = 1.0) -> DetectedEvent | None:
    goals = [e for e in project.events if is_goal(e) and abs(float(e.t) - t) <= tolerance]
    return min(goals, key=lambda e: abs(float(e.t) - t)) if goals else None


def clip_for_goal(project: Project, goal: DetectedEvent, *, lead: float, tail: float) -> Clip:
    """A not-yet-rendered clip around ``goal`` (``lead`` seconds of build-up,
    ``tail`` seconds after), shaped like the detector's own goal clips."""
    d = goal.data or {}
    t = float(goal.t)
    duration = float(project.source.duration) if project.source else t + tail
    st = project.settings
    start, end = max(0.0, t - lead), min(duration, t + tail)
    if end - start < st.min_len:
        start = max(0.0, end - st.min_len)
    if end - start > st.max_len:
        start = max(0.0, end - st.max_len)
    importance = float(d.get("importance") or 0.0)
    score = int(round(max(_SCORE_FLOOR, min(99.0, _BASE_SCORE + importance))))
    minute = f" ({d.get('minute')}')" if d.get("minute") is not None else ""
    m, s = divmod(int(t), 60)
    return Clip(
        start=round(start, 3), end=round(end, 3), kind="gameplay", status=ClipStatus.pending,
        title=f"Goal {d.get('score_after') or ''} — {m}:{s:02d}".replace("  ", " "),
        score=score,
        factors=[ScoreFactor(
            label=f"Scoreboard {d.get('score_before')} → {d.get('score_after')}{minute}",
            weight=round(float(goal.confidence) * 20, 1), detail="Picked by hand from the goal list")],
        features={"ocr": round(float(goal.confidence), 4), "hud": 1.0,
                  "hud_importance": importance, "climax_t": round(t, 3),
                  "evidence_t": round(t, 3)},
        raw_start=round(start, 3), raw_end=round(end, 3))

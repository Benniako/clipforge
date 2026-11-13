"""Tests for the goal list (``app.goals``) and its two API routes.

    cd backend && python -m pytest tests/test_goals.py -q
"""
from __future__ import annotations

import os
import tempfile

# Keep the real data directory out of reach before anything imports settings.
os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

from app import goals  # noqa: E402
from app.models import (Clip, DetectedEvent, ImportSettings, Project,  # noqa: E402
                        SourceMedia)
from app.pipeline import orchestrator  # noqa: E402


def _goal(t: float, before: str, after: str, **data) -> DetectedEvent:
    payload = {"detector": "hud", "team": "a", "score_before": before, "score_after": after,
               "minute": 10, "clock": "09:30", "streamer": True, "importance": 5.0}
    payload.update(data)
    return DetectedEvent(t=t, source="ocr", label="goal", detail="x", confidence=0.95, data=payload)


def _project(events, clips=()) -> Project:
    return Project(id="proj_goals", settings=ImportSettings(min_len=15, max_len=60),
                   source=SourceMedia(filename="v.mp4", path="proj_goals/source.mp4", duration=3000.0),
                   events=list(events), clips=list(clips))


def test_goals_are_grouped_into_matches_in_playing_order():
    events = [_goal(700.0, "0-0", "0-1", team="b"),            # second match, listed first on purpose
              _goal(100.0, "0-0", "1-0"), _goal(160.0, "1-0", "1-1", team="b", streamer=False),
              _goal(240.0, "1-1", "2-1"),
              DetectedEvent(t=50.0, source="cue", label="goal", detail="whistle", confidence=0.9),
              _goal(900.0, "0-1", "1-1")]
    matches = goals.goals_by_match(_project(events))
    assert [(m["index"], m["final_score"], len(m["goals"])) for m in matches] == [(1, "2-1", 3), (2, "1-1", 2)]
    assert [g["score_after"] for g in matches[0]["goals"]] == ["1-0", "1-1", "2-1"]
    assert (matches[0]["start"], matches[0]["end"]) == (100.0, 240.0)
    assert matches[0]["goals"][1]["streamer"] is False and matches[0]["goals"][0]["clip_id"] is None


def test_two_matches_ending_and_starting_level_are_still_split():
    # 0-0 -> 1-0, then a new match: its first goal starts from 0-0 again.
    matches = goals.goals_by_match(_project([_goal(10.0, "0-0", "1-0"), _goal(900.0, "0-0", "1-0")]))
    assert len(matches) == 2


def test_a_goal_knows_the_clip_that_already_shows_it():
    clip = Clip(id="c1", start=80.0, end=110.0, kind="gameplay")
    matches = goals.goals_by_match(_project([_goal(100.0, "0-0", "1-0"), _goal(300.0, "1-0", "2-0")], [clip]))
    first, second = matches[0]["goals"]
    assert (first["clip_id"], first["clip_status"]) == ("c1", "pending") and second["clip_id"] is None


def test_clip_for_goal_has_the_goal_near_its_end_and_a_verified_score():
    project = _project([_goal(100.0, "0-0", "1-0", importance=9.0, minute=13)])
    goal = goals.find_goal(project, 100.4)
    assert goal is not None and goals.find_goal(project, 105.0) is None
    clip = goals.clip_for_goal(project, goal, lead=29.0, tail=8.0)
    assert (clip.start, clip.end, clip.kind) == (71.0, 108.0, "gameplay")
    assert clip.score == 93 and clip.title == "Goal 1-0 — 1:40"
    assert clip.features["hud"] == 1.0 and clip.features["climax_t"] == 100.0
    assert all(isinstance(v, float) for v in clip.features.values())
    early = goals.clip_for_goal(_project([_goal(5.0, "0-0", "1-0")]), _goal(5.0, "0-0", "1-0"), lead=29.0, tail=8.0)
    assert early.start == 0.0 and early.end - early.start >= 13.0
    weak = goals.clip_for_goal(project, _goal(100.0, "5-0", "6-0", importance=-7.0), lead=29.0, tail=8.0)
    assert weak.score == 80


def test_every_scoreboard_goal_is_kept_on_the_project():
    accepted = [_goal(100.0, "0-0", "1-0")]
    detected = [*accepted, _goal(200.0, "1-0", "2-0"),
                DetectedEvent(t=150.0, source="audio", label="cheer", confidence=0.5)]
    kept = orchestrator._events_to_keep(detected, accepted)
    assert [(e.t, e.label) for e in kept] == [(100.0, "goal"), (200.0, "goal")]


def test_goal_routes(monkeypatch):
    from fastapi.testclient import TestClient

    from app import store
    from app.main import app

    store.init_db()
    project = _project([_goal(100.0, "0-0", "1-0"), _goal(160.0, "1-0", "2-0")])
    store.save(project)
    rendered: list[str] = []
    monkeypatch.setattr(orchestrator.engine, "rerender_clip", lambda pid, cid: rendered.append(cid))
    with TestClient(app) as client:
        listing = client.get(f"/api/projects/{project.id}/goals").json()
        assert listing["goals"] == 2 and listing["matches"][0]["final_score"] == "2-0"
        made = client.post(f"/api/projects/{project.id}/goals/clip", params={"t": 100.0}).json()
        assert made["created"] is True
        again = client.post(f"/api/projects/{project.id}/goals/clip", params={"t": 100.0}).json()
        assert again == {"ok": True, "clip_id": made["clip_id"], "created": False}
        assert client.post(f"/api/projects/{project.id}/goals/clip", params={"t": 500.0}).status_code == 404
        assert client.get("/api/projects/nope/goals").status_code == 404
        listing = client.get(f"/api/projects/{project.id}/goals").json()
        assert listing["matches"][0]["goals"][0]["clip_id"] == made["clip_id"]
    assert len(store.get(project.id).clips) == 1

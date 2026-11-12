"""Tests for the offline evaluation harness and the ``hud`` / ``eval`` CLI commands.

No video, no ffmpeg, no OCR: label files live in pytest's tmp_path and the
detectors are replaced by fakes, so the whole file runs in about a second.

    cd backend && python -m pytest tests/test_evaluation.py -q
"""
from __future__ import annotations

import importlib
import io
import json
import os
import random
import sys
import tempfile
import types
from pathlib import Path

import pytest

# Keep the settings (and anything they create on disk) away from the real data
# dir. Must happen before anything imports app.config.
if "app.config" not in sys.modules:
    os.environ["CLIPFORGE_DATA_DIR"] = tempfile.mkdtemp(prefix="clipforge-eval-test-")

from app import evaluation as ev
from app.evaluation import LabelEvent, LabelSet
from app.providers.hud import HudEvent, HudResult, MatchInfo


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _valid() -> dict:
    return {
        "version": 1,
        "video": "video.mp4",
        "game_profile": "eafc",
        "status": "verified",
        "duration": 600.0,
        "labelled_ranges": [[0, 250], [300, 600]],
        "events": [
            {"t": 159.2, "label": "goal", "team": "a", "score": "1-0", "minute": 11, "note": ""},
            {"t": 420.5, "label": "goal", "team": "b", "score": "1-1"},
        ],
    }


def _write(tmp_path: Path, data, name: str = "labels.json") -> Path:
    p = tmp_path / name
    p.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return p


def _hud_result(*, healthy: bool = True) -> HudResult:
    events = [
        HudEvent(t=159.23, team="a", score_before=(0, 0), score_after=(1, 0), t_change=161.0,
                 window=(158.0, 162.0), minute=11, clock="10:29", refined=True, streamer=True,
                 importance=9.0, confidence=0.95),
        HudEvent(t=300.07, team="b", score_before=(1, 0), score_after=(1, 1), t_change=302.0,
                 window=(280.0, 302.0), inferred=True, streamer=False, importance=2.0,
                 confidence=0.5),
        HudEvent(t=420.5, team="b", score_before=(1, 1), score_after=(1, 2), t_change=440.0,
                 window=(419.0, 440.0), hidden=True, refined=True, minute=88, importance=-3.0,
                 confidence=0.85),
    ]
    return HudResult(layout="eafc_stacked", events=events,
                     matches=[MatchInfo(t_start=30.0, t_end=3725.0, start_score=(0, 0),
                                        final_score=(1, 2), goals=3)],
                     in_play=[(28.0, 562.0)], coverage=0.89, healthy=healthy, samples=280,
                     sample_interval=2.0, streamer_side="a",
                     diagnostics={"mode": "keyframes", "region": [200, 70, 200, 120]})


def _brute(truth: list[float], pred: list[float], tol: float) -> tuple[int, float]:
    """(largest number of pairs, smallest total |dt| at that number) by trying everything."""
    best = [0, 0.0]

    def rec(i: int, used: frozenset, count: int, cost: float) -> None:
        if i == len(truth):
            if count > best[0] or (count == best[0] and cost < best[1]):
                best[0], best[1] = count, cost
            return
        rec(i + 1, used, count, cost)
        for j, p in enumerate(pred):
            if j not in used and abs(truth[i] - p) <= tol + 1e-9:
                rec(i + 1, used | {j}, count + 1, cost + abs(truth[i] - p))

    rec(0, frozenset(), 0, 0.0)
    return best[0], best[1]


def _goals(*times: float) -> list[LabelEvent]:
    return [LabelEvent(t=t, label="goal") for t in times]


# --------------------------------------------------------------------------- #
# Label files
# --------------------------------------------------------------------------- #
def test_labels_round_trip_is_sorted_stable_and_readable(tmp_path):
    labels = LabelSet(
        video="video.mp4", game_profile="eafc", status="verified", duration=13978.75,
        labelled_ranges=[(0.0, 3600.0), (7200.0, 9000.0)],
        events=[
            LabelEvent(t=420.5, label="goal", team="b", score="1-1", minute=55, note="Abstoß, knapp"),
            LabelEvent(t=159.2, label="goal", team="a", score="1-0", minute=11),
            LabelEvent(t=3000.0, label="save"),
        ])
    path = ev.save_labels(labels, tmp_path / "sub" / "labels.json")
    raw = path.read_bytes()
    text = raw.decode("utf-8")

    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert b"\r" not in raw
    assert "Abstoß" in text                                  # UTF-8, not \u escapes
    assert json.loads(text)["version"] == 1
    lines = text.splitlines()
    assert lines[1] == '  "version": 1,'                     # 2-space indent
    assert '  "labelled_ranges": [[0.0, 3600.0], [7200.0, 9000.0]],' in lines
    event_lines = [l for l in lines if l.startswith('    {"t"')]
    assert len(event_lines) == 3                             # one event per line
    assert event_lines[0] == '    {"t": 159.2, "label": "goal", "team": "a", "score": "1-0", "minute": 11},'
    assert event_lines[2] == '    {"t": 3000.0, "label": "save"}'
    assert not (tmp_path / "sub" / "labels.json.tmp").exists()

    loaded = ev.load_labels(path)
    assert [e.t for e in loaded.events] == [159.2, 420.5, 3000.0]
    assert loaded == LabelSet(video="video.mp4", game_profile="eafc", status="verified",
                              duration=13978.75, labelled_ranges=[(0.0, 3600.0), (7200.0, 9000.0)],
                              events=sorted(labels.events, key=lambda e: e.t))
    assert loaded.verified
    # Saving what was loaded gives the same bytes.
    assert ev.save_labels(loaded, tmp_path / "again.json").read_bytes() == raw


def test_load_minimal_file_with_bom_and_unsorted_events(tmp_path):
    data = {"version": 1, "video": "v.mp4", "game_profile": "eafc", "status": "draft",
            "events": [{"t": 50, "label": "goal"}, {"t": 10.5, "label": "goal", "score": "",
                                                    "team": None, "minute": None, "note": None}]}
    p = tmp_path / "bom.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(data).encode("utf-8"))
    labels = ev.load_labels(p)
    assert labels.status == "draft" and not labels.verified
    assert labels.duration is None and labels.labelled_ranges is None
    assert labels.events == [LabelEvent(t=10.5, label="goal"), LabelEvent(t=50.0, label="goal")]


def test_save_writes_no_events_as_an_empty_list(tmp_path):
    p = ev.save_labels(LabelSet(video="v.mp4", game_profile="eafc"), tmp_path / "empty.json")
    assert '"events": []' in p.read_text(encoding="utf-8")
    assert ev.load_labels(p).events == []


def test_load_rejects_bad_top_level_fields(tmp_path):
    def drop(key):
        return lambda d: d.pop(key)

    def put(key, value):
        return lambda d: d.__setitem__(key, value)

    cases = [
        (put("version", 2), ["version", "2"]),
        (put("version", "1"), ["version"]),
        (put("version", True), ["version"]),
        (drop("version"), ["missing key", "version"]),
        (drop("video"), ["missing key", "video"]),
        (drop("game_profile"), ["missing key", "game_profile"]),
        (drop("status"), ["missing key", "status"]),
        (drop("events"), ["missing key", "events"]),
        (put("video", ""), ["video"]),
        (put("video", 5), ["video"]),
        (put("game_profile", " "), ["game_profile"]),
        (put("status", "checked"), ["status", "checked", "draft", "verified"]),
        (put("duration", 0), ["duration"]),
        (put("duration", "600"), ["duration"]),
        (put("duration", -5), ["duration"]),
        (put("events", {"t": 1}), ["events", "list"]),
        (put("labeled_ranges", [[0, 600]]), ["unknown key", "labeled_ranges"]),
    ]
    for mutate, needles in cases:
        data = _valid()
        mutate(data)
        with pytest.raises(ValueError) as err:
            ev.load_labels(_write(tmp_path, data))
        msg = str(err.value)
        assert msg.startswith("labels.json: "), msg
        for needle in needles:
            assert needle in msg, (needles, msg)


def test_load_rejects_bad_labelled_ranges(tmp_path):
    cases = [
        ([[300, 600], [0, 250]], ["labelled_ranges[1]", "unsorted"]),      # unsorted
        ([[0, 300], [250, 600]], ["labelled_ranges[1]", "overlaps"]),      # overlapping
        ([[0, 250], [300, 601]], ["labelled_ranges[1]", "beyond duration"]),
        ([[-1, 250], [300, 600]], ["labelled_ranges[0]", "before 0"]),
        ([[250, 250], [300, 600]], ["labelled_ranges[0]", "end after"]),
        ([[0, 250], [300]], ["labelled_ranges[1]", "pair"]),
        ([[0, "250"], [300, 600]], ["labelled_ranges[0]", "pair"]),
        ("0-600", ["labelled_ranges", "list"]),
        ([[0, 100], [300, 600]], ["events[0]", "outside labelled_ranges"]),  # 159.2 not covered
    ]
    for ranges, needles in cases:
        data = _valid()
        data["labelled_ranges"] = ranges
        with pytest.raises(ValueError) as err:
            ev.load_labels(_write(tmp_path, data))
        for needle in needles:
            assert needle in str(err.value), (needles, str(err.value))
    # Ranges that merely touch are fine, and so is an event sitting on a boundary.
    data = _valid()
    data["labelled_ranges"] = [[0, 159.2], [159.2, 600]]
    assert len(ev.load_labels(_write(tmp_path, data)).events) == 2


def test_load_rejects_bad_events_and_names_the_index(tmp_path):
    def with_event(**fields):
        def mutate(data):
            data["events"][1].update(fields)
        return mutate

    def drop(key):
        return lambda data: data["events"][1].pop(key)

    cases = [
        (with_event(t="420.5"), ["events[1].t", "number"]),
        (with_event(t=None), ["events[1].t", "number"]),
        (with_event(t=True), ["events[1].t", "number"]),
        (with_event(t=-0.1), ["events[1].t", "negative"]),
        (with_event(t=600.5), ["events[1].t", "beyond duration"]),
        (with_event(t=275), ["events[1].t", "outside labelled_ranges"]),
        (drop("t"), ["events[1]", "missing key", "'t'"]),
        (drop("label"), ["events[1]", "missing key", "'label'"]),
        (with_event(label=""), ["events[1].label"]),
        (with_event(label=7), ["events[1].label"]),
        (with_event(team="home"), ["events[1].team", "home"]),
        (with_event(score="2:1"), ["events[1].score", "2:1"]),
        (with_event(score=21), ["events[1].score"]),
        (with_event(minute="11"), ["events[1].minute"]),
        (with_event(minute=-1), ["events[1].minute"]),
        (with_event(minute=11.5), ["events[1].minute"]),
        (with_event(note=3), ["events[1].note"]),
        (with_event(time=420.5), ["events[1]", "unknown key", "time"]),
        (lambda data: data["events"].__setitem__(1, [420.5, "goal"]), ["events[1]", "object"]),
    ]
    for mutate, needles in cases:
        data = _valid()
        mutate(data)
        with pytest.raises(ValueError) as err:
            ev.load_labels(_write(tmp_path, data))
        for needle in needles:
            assert needle in str(err.value), (needles, str(err.value))


def test_load_rejects_nan_times(tmp_path):
    text = json.dumps(_valid()).replace("159.2", "NaN")
    with pytest.raises(ValueError, match=r"events\[0\]\.t"):
        ev.load_labels(_write(tmp_path, text))


def test_load_rejects_files_that_are_not_label_json(tmp_path):
    with pytest.raises(ValueError, match="not a valid UTF-8 JSON"):
        ev.load_labels(_write(tmp_path, "{ this is not json"))
    with pytest.raises(ValueError, match="JSON object"):
        ev.load_labels(_write(tmp_path, "[1, 2, 3]"))
    bad = tmp_path / "latin.json"
    bad.write_bytes(b'{"video": "\xff\xfe"}')
    with pytest.raises(ValueError, match="not a valid UTF-8 JSON"):
        ev.load_labels(bad)
    with pytest.raises(OSError):
        ev.load_labels(tmp_path / "does-not-exist.json")


def test_save_refuses_a_set_that_would_not_load(tmp_path):
    target = tmp_path / "bad.json"
    for labels, needle in [
        (LabelSet(video="v.mp4", status="final"), "status"),
        (LabelSet(video="v.mp4", events=[LabelEvent(t=-1.0, label="goal")]), "events[0].t"),
        (LabelSet(video="v.mp4", duration=10.0, events=[LabelEvent(t=11.0, label="goal")]), "beyond duration"),
        (LabelSet(video="v.mp4", labelled_ranges=[(0.0, 5.0)],
                  events=[LabelEvent(t=6.0, label="goal")]), "outside labelled_ranges"),
    ]:
        with pytest.raises(ValueError) as err:
            ev.save_labels(labels, target)
        assert needle in str(err.value)
    assert not target.exists()


def test_resolve_video_and_video_reference(tmp_path):
    label_path = tmp_path / "labels" / "match.labels.json"
    rel = LabelSet(video="../videos/match.mp4")
    assert ev.resolve_video(rel, label_path) == tmp_path / "videos" / "match.mp4"
    assert ev.resolve_video(rel, label_path).is_absolute()
    absolute = tmp_path / "elsewhere" / "match.mp4"
    assert ev.resolve_video(LabelSet(video=str(absolute)), label_path) == absolute

    # A video next to (or below) the label file is referenced relatively ...
    assert ev.video_reference(tmp_path / "labels" / "match.mp4", label_path) == "match.mp4"
    assert ev.video_reference(tmp_path / "labels" / "raw" / "match.mp4", label_path) == "raw/match.mp4"
    # ... anything else by its absolute path; both resolve back to the same file.
    ref = ev.video_reference(absolute, label_path)
    assert Path(ref).is_absolute()
    assert ev.resolve_video(LabelSet(video=ref), label_path) == absolute.resolve()


# --------------------------------------------------------------------------- #
# Drafts from the scoreboard tracker
# --------------------------------------------------------------------------- #
def test_draft_from_hud_builds_one_goal_per_event(tmp_path):
    draft = ev.draft_from_hud(_hud_result(), video="video.mp4", game_profile="eafc", duration=600.04)
    assert draft.status == "draft" and not draft.verified
    assert (draft.video, draft.game_profile, draft.duration) == ("video.mp4", "eafc", 600.04)
    assert draft.labelled_ranges is None
    assert [e.label for e in draft.events] == ["goal", "goal", "goal"]
    assert [e.t for e in draft.events] == [159.2, 300.1, 420.5]            # rounded to 0.1 s
    assert [e.team for e in draft.events] == ["a", "b", "b"]
    assert [e.score for e in draft.events] == ["1-0", "1-1", "1-2"]        # score after the goal
    assert [e.minute for e in draft.events] == [11, None, 88]
    assert draft.events[0].note == ""
    assert "low confidence (0.50)" in draft.events[1].note
    assert "inferred" in draft.events[1].note
    assert "hidden" in draft.events[2].note and "low confidence" not in draft.events[2].note
    # A draft is a valid label file.
    assert ev.load_labels(ev.save_labels(draft, tmp_path / "draft.json")) == draft


def test_draft_from_hud_keeps_times_inside_the_video_and_handles_no_events(tmp_path):
    res = _hud_result()
    res.events[2].t = 599.99                    # would round up past the end of the video
    res.events[0].t = -0.3
    draft = ev.draft_from_hud(res, video="v.mp4", game_profile="eafc", duration=599.97)
    assert [e.t for e in draft.events] == [0.0, 300.1, 599.9]
    ev.save_labels(draft, tmp_path / "d.json")  # passes the "t beyond duration" check

    res.events = []
    empty = ev.draft_from_hud(res, video="v.mp4", game_profile="eafc", duration=None)
    assert empty.events == [] and empty.duration is None and empty.status == "draft"


# --------------------------------------------------------------------------- #
# match_events
# --------------------------------------------------------------------------- #
def test_match_events_empty_sides():
    assert ev.match_events([], [], 6.0) == []
    assert ev.match_events([10.0], [], 6.0) == []
    assert ev.match_events([], [10.0], 6.0) == []


def test_match_events_exactly_at_the_tolerance_counts():
    assert ev.match_events([10.0], [16.0], 6.0) == [(0, 0)]
    assert ev.match_events([16.0], [10.0], 6.0) == [(0, 0)]
    assert ev.match_events([159.2], [165.2], 6.0) == [(0, 0)]      # 165.2 - 159.2 != 6.0 in binary
    assert ev.match_events([0.1], [6.1], 6.0) == [(0, 0)]
    assert ev.match_events([10.0], [16.01], 6.0) == []
    assert ev.match_events([10.0], [10.0], 0.0) == [(0, 0)]
    assert ev.match_events([10.0], [10.001], 0.0) == []


def test_match_events_ties_and_duplicates_stay_one_to_one():
    # One truth, two predictions equally far away: exactly one of them is used.
    pairs = ev.match_events([10.0], [8.0, 12.0], 6.0)
    assert len(pairs) == 1 and pairs[0][0] == 0
    assert pairs == ev.match_events([10.0], [8.0, 12.0], 6.0)      # and always the same one
    # Identical timestamps on both sides pair off completely.
    pairs = ev.match_events([10.0, 10.0], [10.0, 10.0], 6.0)
    assert sorted(i for i, _ in pairs) == [0, 1] and sorted(j for _, j in pairs) == [0, 1]
    # Three truths at one moment, two predictions: two pairs, no prediction reused.
    pairs = ev.match_events([5.0, 5.0, 5.0], [5.0, 6.0], 2.0)
    assert len(pairs) == 2 and len({j for _, j in pairs}) == 2


def test_match_events_two_truths_competing_for_one_prediction():
    pairs = ev.match_events([10.0, 14.0], [12.0], 3.0)
    assert len(pairs) == 1 and pairs[0][1] == 0
    # The nearer truth wins it.
    assert ev.match_events([10.0, 13.0], [12.0], 3.0) == [(1, 0)]
    assert ev.match_events([11.5, 14.0], [12.0], 3.0) == [(0, 0)]


def test_match_events_prefers_the_closest_partner_but_never_at_the_cost_of_a_match():
    # A duplicate 5 s early must not take the pair away from the exact hit.
    assert ev.match_events([10.0], [5.0, 10.0], 6.0) == [(0, 1)]
    # Pairing the two nearest (5 and 4) would strand 0 and 9; the maximum is two pairs.
    assert ev.match_events([0.0, 5.0], [4.0, 9.0], 5.0) == [(0, 0), (1, 1)]
    # Nothing matches across a gap wider than the tolerance.
    assert ev.match_events([0.0, 100.0], [50.0], 6.0) == []


def test_match_events_returns_indices_into_unsorted_inputs():
    assert ev.match_events([50.0, 10.0], [11.0, 49.0, 300.0], 6.0) == [(1, 0), (0, 1)]


def test_match_events_rejects_a_negative_tolerance():
    with pytest.raises(ValueError):
        ev.match_events([1.0], [1.0], -1.0)
    with pytest.raises(ValueError):
        ev.match_events([1.0], [1.0], float("nan"))


def _check_against_brute_force(*, minimal_error: bool) -> None:
    rng = random.Random(20240611)
    for _ in range(600):
        # Half-second grid: plenty of ties and exact-tolerance cases, exact float sums.
        truth = [rng.randrange(0, 40) / 2.0 for _ in range(rng.randrange(0, 6))]
        pred = [rng.randrange(0, 40) / 2.0 for _ in range(rng.randrange(0, 6))]
        tol = rng.choice([0.0, 0.5, 1.0, 2.5, 6.0])
        pairs = ev.match_events(truth, pred, tol)
        case = (truth, pred, tol, pairs)
        assert len({i for i, _ in pairs}) == len(pairs), case          # one-to-one
        assert len({j for _, j in pairs}) == len(pairs), case
        assert all(abs(truth[i] - pred[j]) <= tol for i, j in pairs), case
        best_count, best_cost = _brute(truth, pred, tol)
        assert len(pairs) == best_count, case                          # maximum
        if minimal_error:
            assert sum(abs(truth[i] - pred[j]) for i, j in pairs) == pytest.approx(best_cost), case


def test_match_events_is_maximum_and_closest_against_brute_force():
    _check_against_brute_force(minimal_error=True)


def test_match_events_sweep_fallback_is_still_maximum(monkeypatch):
    # Force the linear sweep that replaces the alignment table on huge clusters.
    monkeypatch.setattr(ev, "_DP_CELL_LIMIT", 0)
    _check_against_brute_force(minimal_error=False)
    assert ev.match_events([0.0, 5.0], [4.0, 9.0], 5.0) == [(0, 0), (1, 1)]


# --------------------------------------------------------------------------- #
# score_events
# --------------------------------------------------------------------------- #
def test_score_events_hand_computed():
    truth = _goals(100.0, 200.0, 300.0, 400.0)
    predicted = [(101.0, "goal"), (197.0, "goal"), (250.0, "goal"), (407.0, "goal")]
    m = ev.score_events(truth, predicted)                 # default tolerance 6 s
    assert m.tolerance == 6.0
    g = m.per_label["goal"]
    assert (g.truth, g.predicted, g.tp, g.fp, g.fn) == (4, 4, 2, 2, 2)
    assert g.precision == 0.5 and g.recall == 0.5 and g.f1 == 0.5
    assert g.mean_abs_error == pytest.approx(2.0)         # |+1|, |-3|
    assert g.median_abs_error == pytest.approx(2.0)
    assert g.bias == pytest.approx(-1.0)                  # predictions early on average
    assert [e.t for e in g.missed] == [300.0, 400.0]
    assert [e.t for e in g.extra] == [250.0, 407.0]
    assert g.pairs == [(100.0, 101.0), (200.0, 197.0)]
    assert list(m.per_label) == ["goal"]
    o = m.overall
    assert (o.label, o.tp, o.fp, o.fn, o.precision, o.recall) == ("all", 2, 2, 2, 0.5, 0.5)
    assert m.ignored_predictions == 0 and m.ignored_truth == 0

    # A wider tolerance turns the 7 s miss into a hit.
    wide = ev.score_events(truth, predicted, tolerance=7.0).overall
    assert (wide.tp, wide.fp, wide.fn) == (3, 1, 1)
    assert wide.precision == 0.75 and wide.recall == 0.75
    assert wide.mean_abs_error == pytest.approx((1 + 3 + 7) / 3)
    assert wide.median_abs_error == pytest.approx(3.0)
    assert wide.bias == pytest.approx((1 - 3 + 7) / 3)


def test_score_events_matches_within_a_label_only():
    truth = [LabelEvent(t=100.0, label="goal", score="1-0", minute=11), LabelEvent(t=200.0, label="save")]
    predicted = [(100.5, "goal"), (200.0, "goal"), (900.0, "kill")]
    m = ev.score_events(truth, predicted)
    goal, save, kill = m.per_label["goal"], m.per_label["save"], m.per_label["kill"]
    assert (goal.tp, goal.fp, goal.fn) == (1, 1, 0)       # the 200.0 "goal" is not the save
    assert (goal.precision, goal.recall) == (0.5, 1.0)
    assert goal.f1 == pytest.approx(2 / 3)
    assert (save.tp, save.fp, save.fn, save.precision, save.recall, save.f1) == (0, 0, 1, 1.0, 0.0, 0.0)
    assert (kill.tp, kill.fp, kill.fn, kill.precision, kill.recall, kill.f1) == (0, 1, 0, 0.0, 1.0, 0.0)
    o = m.overall
    assert (o.tp, o.fp, o.fn) == (1, 2, 1)
    assert o.precision == pytest.approx(1 / 3) and o.recall == 0.5
    assert o.f1 == pytest.approx(0.4)
    assert [(e.t, e.label) for e in o.missed] == [(200.0, "save")]
    assert [(e.t, e.label) for e in o.extra] == [(200.0, "goal"), (900.0, "kill")]

    # Restricted to one label, the others vanish from both sides.
    only = ev.score_events(truth, predicted, labels=("goal",))
    assert list(only.per_label) == ["goal"]
    assert (only.overall.tp, only.overall.fp, only.overall.fn) == (1, 1, 0)
    # The truth events themselves come back, so a report can show their details.
    assert ev.score_events(truth, [], labels=("goal",)).overall.missed == [truth[0]]


def test_score_events_labelled_ranges():
    ranges = [(0.0, 150.0), (1000.0, 1100.0)]
    truth = _goals(100.0, 150.0, 1050.0)
    predicted = [
        (101.0, "goal"),      # hit
        (120.0, "goal"),      # inside a checked range, nothing there: false positive
        (152.0, "goal"),      # 2 s outside the range, but it found the goal at 150: hit
        (500.0, "goal"),      # never checked by a human: ignored
        (2000.0, "goal"),     # ditto
    ]
    m = ev.score_events(truth, predicted, labelled_ranges=ranges)
    o = m.overall
    assert (o.tp, o.fp, o.fn) == (2, 1, 1)
    assert m.ignored_predictions == 2
    assert o.precision == pytest.approx(2 / 3) and o.recall == pytest.approx(2 / 3)
    assert [e.t for e in o.extra] == [120.0]
    assert [e.t for e in o.missed] == [1050.0]
    assert m.labelled_ranges == ranges

    # Without the ranges the same predictions are all on the hook.
    plain = ev.score_events(truth, predicted).overall
    assert (plain.tp, plain.fp, plain.fn) == (2, 3, 1)

    # Truth outside the checked ranges is left out too (a label file never has
    # any, but callers may pass their own lists).
    m = ev.score_events(_goals(100.0, 600.0), [(100.0, "goal")], labelled_ranges=ranges)
    assert (m.overall.tp, m.overall.fn, m.ignored_truth) == (1, 0, 1)


def test_score_events_empty_cases():
    both = ev.score_events([], []).overall
    assert (both.tp, both.fp, both.fn) == (0, 0, 0)
    assert (both.precision, both.recall, both.f1) == (1.0, 1.0, 1.0)
    assert both.mean_abs_error is None and both.median_abs_error is None and both.bias is None

    no_pred = ev.score_events(_goals(10.0, 20.0), []).overall
    assert (no_pred.tp, no_pred.fp, no_pred.fn) == (0, 0, 2)
    assert (no_pred.precision, no_pred.recall, no_pred.f1) == (1.0, 0.0, 0.0)

    no_truth = ev.score_events([], [(10.0, "goal")]).overall
    assert (no_truth.tp, no_truth.fp, no_truth.fn) == (0, 1, 0)
    assert (no_truth.precision, no_truth.recall, no_truth.f1) == (0.0, 1.0, 0.0)

    # A label asked for by name gets its row even when nothing carries it.
    named = ev.score_events([], [], labels=("goal",))
    assert named.per_label["goal"].precision == 1.0 and named.per_label["goal"].recall == 1.0

    # Everything predicted lies outside the checked ranges: same as no predictions.
    outside = ev.score_events(_goals(10.0), [(500.0, "goal")], labelled_ranges=[(0.0, 100.0)])
    assert (outside.overall.precision, outside.overall.recall) == (1.0, 0.0)
    assert outside.ignored_predictions == 1


def test_score_events_accepts_plain_tuples_as_truth():
    m = ev.score_events([(10.0, "goal")], [(12.0, "goal")], tolerance=2.0)
    assert (m.overall.tp, m.overall.bias) == (1, pytest.approx(2.0))


# --------------------------------------------------------------------------- #
# score_clips
# --------------------------------------------------------------------------- #
def test_score_clips_hand_computed():
    truth = [LabelEvent(t=100.0, label="goal"), LabelEvent(t=200.0, label="goal"),
             LabelEvent(t=300.0, label="save")]
    clips = [(290.0, 310.0), (90.0, 110.0), (95.0, 105.0), (150.0, 160.0)]
    c = ev.score_clips(truth, clips)
    assert (c.clips, c.hits, c.events, c.events_covered) == (4, 3, 3, 2)
    assert c.clip_hit_rate == 0.75
    assert c.event_coverage == pytest.approx(2 / 3)
    assert c.empty_clips == [(150.0, 160.0)]
    assert [e.t for e in c.uncovered] == [200.0]
    assert c.ignored_clips == 0

    goals_only = ev.score_clips(truth, clips, labels=("goal",))
    assert (goals_only.clips, goals_only.hits, goals_only.events, goals_only.events_covered) == (4, 2, 2, 1)
    assert goals_only.clip_hit_rate == 0.5 and goals_only.event_coverage == 0.5
    assert goals_only.empty_clips == [(150.0, 160.0), (290.0, 310.0)]


def test_score_clips_labelled_ranges_and_boundaries():
    truth = [LabelEvent(t=100.0, label="goal"), LabelEvent(t=200.0, label="goal")]
    clips = [(90.0, 100.0),      # the event sits exactly on the clip's end: inside
             (240.0, 260.0),     # straddles the end of the checked range: scored, empty
             (251.0, 280.0),     # entirely outside: ignored
             (400.0, 420.0)]     # entirely outside: ignored
    c = ev.score_clips(truth, clips, labelled_ranges=[(0.0, 250.0)])
    assert (c.clips, c.hits, c.ignored_clips) == (2, 1, 2)
    assert c.clip_hit_rate == 0.5
    assert (c.events, c.events_covered, c.event_coverage) == (2, 1, 0.5)
    assert c.empty_clips == [(240.0, 260.0)]

    # A clip given back to front is still a clip.
    assert ev.score_clips(truth, [(110.0, 90.0)]).hits == 1


def test_score_clips_empty_cases():
    none = ev.score_clips([LabelEvent(t=1.0, label="goal")], [])
    assert (none.clips, none.clip_hit_rate, none.event_coverage) == (0, 1.0, 0.0)
    no_truth = ev.score_clips([], [(0.0, 10.0)])
    assert (no_truth.clip_hit_rate, no_truth.event_coverage) == (0.0, 1.0)
    assert no_truth.empty_clips == [(0.0, 10.0)]
    nothing = ev.score_clips([], [])
    assert (nothing.clip_hit_rate, nothing.event_coverage) == (1.0, 1.0)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def test_format_hms():
    assert ev.format_hms(0) == "0:00:00"
    assert ev.format_hms(159.2) == "0:02:39"
    assert ev.format_hms(59.99) == "0:00:59"
    assert ev.format_hms(3725.0) == "1:02:05"
    assert ev.format_hms(13978.75) == "3:52:58"
    assert ev.format_hms(-3) == "0:00:00"


def test_format_report_content():
    truth = [LabelEvent(t=100.0, label="goal"), LabelEvent(t=200.0, label="goal"),
             LabelEvent(t=300.0, label="goal", score="2-1", minute=67, note="penalty"),
             LabelEvent(t=3725.0, label="goal")]
    predicted = [(101.0, "goal"), (197.0, "goal"), (250.0, "goal"), (3732.0, "goal")]
    metrics = ev.score_events(truth, predicted)
    clips = ev.score_clips(truth, [(90.0, 110.0), (500.0, 530.0)])
    report = ev.format_report(metrics, clips, title="hud detector vs verified labels: v.mp4",
                              warnings=["these labels are a DRAFT"])
    lines = report.splitlines()
    assert lines[0] == "hud detector vs verified labels: v.mp4"
    assert lines[1] == "WARNING: these labels are a DRAFT"
    assert "tolerance: +/-6 s" in lines
    header = next(l for l in lines if l.startswith("label"))
    assert header.split() == ["label", "truth", "pred", "tp", "fp", "fn", "precision", "recall", "f1"]
    row = next(l for l in lines if l.startswith("goal"))
    assert row.split() == ["goal", "4", "4", "2", "2", "2", "0.500", "0.500", "0.500"]
    assert not any(l.startswith("all") for l in lines)      # one label: no separate total row
    assert ("timing: 2 matched, mean |dt| 2.00 s, median |dt| 2.00 s, bias -1.00 s "
            "(positive = detected late)") in lines
    assert "missed (2 labelled, not detected):" in lines
    assert "  0:05:00  goal  2-1  67'  (penalty)" in lines
    assert "  1:02:05  goal" in lines
    assert "extra (2 detected, not labelled):" in lines
    assert "  0:04:10  goal" in lines and "  1:02:12  goal" in lines
    assert ("clips: 2 scored, 1 contain a labelled event (hit rate 0.500); 1 of 4 labelled "
            "events are inside a clip (coverage 0.250)") in lines
    assert "empty clips (1):" in lines and "  0:08:20 - 0:08:50" in lines

    # Without clips or a title those parts are simply absent.
    bare = ev.format_report(metrics)
    assert "clips:" not in bare and "WARNING" not in bare and bare.startswith("tolerance")


def test_format_report_lists_at_most_fifteen_and_handles_nothing_matched():
    truth = _goals(*[100.0 * k for k in range(1, 21)])                  # 20 missed
    predicted = [(50.0 + 100.0 * k, "goal") for k in range(1, 18)]      # 17 extra
    report = ev.format_report(ev.score_events(truth, predicted))
    lines = report.splitlines()
    assert "timing: no matched events" in lines
    start = lines.index("missed (20 labelled, not detected):")
    assert lines[start + 1] == "  0:01:40  goal" and lines[start + 15] == "  0:25:00  goal"
    assert lines[start + 16] == "  ... and 5 more"
    start = lines.index("extra (17 detected, not labelled):")
    assert lines[start + 15] == "  0:25:50  goal" and lines[start + 16] == "  ... and 2 more"
    assert report.count("  0:") == 30

    # Several labels: one row each plus the total; ignored predictions are mentioned.
    multi = ev.score_events([LabelEvent(t=10.0, label="goal"), LabelEvent(t=50.0, label="save")],
                            [(11.0, "goal"), (900.0, "goal")], labelled_ranges=[(0.0, 100.0)])
    lines = ev.format_report(multi).splitlines()
    assert next(l for l in lines if l.startswith("goal")).split()[1:6] == ["1", "1", "1", "0", "0"]
    assert next(l for l in lines if l.startswith("save")).split()[1:6] == ["1", "0", "0", "0", "1"]
    assert next(l for l in lines if l.startswith("all")).split()[1:6] == ["2", "1", "1", "0", "1"]
    assert "ignored: 1 unmatched predictions outside the labelled ranges" in lines

    empty = ev.format_report(ev.score_events([], []))
    assert "missed" not in empty and "extra" not in empty


def test_to_json_is_plain_and_complete():
    truth = _goals(*[100.0 * k for k in range(1, 21)]) + [LabelEvent(t=5000.0, label="goal", score="3-2")]
    metrics = ev.score_events(truth, [(101.5, "goal"), (9000.0, "goal")], labelled_ranges=[(0.0, 8000.0)])
    clips = ev.score_clips(truth, [(90.0, 110.0), (6000.0, 6030.0)])
    data = json.loads(json.dumps(ev.to_json(metrics, clips, title="t", warnings=["w"])))
    assert data["title"] == "t" and data["warnings"] == ["w"] and data["tolerance"] == 6.0
    assert data["overall"]["tp"] == 1 and data["overall"]["fn"] == 20 and data["overall"]["fp"] == 0
    assert data["overall"]["recall"] == round(1 / 21, 4)
    assert data["overall"]["mean_abs_error"] == 1.5 and data["overall"]["bias"] == 1.5
    assert len(data["overall"]["missed"]) == 20                      # not cut at 15
    assert data["overall"]["missed"][-1] == {"t": 5000.0, "time": "1:23:20", "label": "goal", "score": "3-2"}
    assert data["labels"]["goal"]["tp"] == 1
    assert data["ignored_predictions"] == 1
    assert data["labelled_ranges"] == [[0.0, 8000.0]]
    assert data["clips"]["clip_hit_rate"] == 0.5 and data["clips"]["hits"] == 1
    assert data["clips"]["empty_clips"] == [{"start": 6000.0, "end": 6030.0, "time": "1:40:00 - 1:40:30"}]
    assert len(data["clips"]["uncovered"]) == 20
    assert ev.to_json(ev.score_events([], []))["clips"] is None
    assert ev.to_json(ev.score_events([], []))["overall"]["mean_abs_error"] is None


# --------------------------------------------------------------------------- #
# CLI: hud and eval, with the detectors replaced by fakes
# --------------------------------------------------------------------------- #
class _Fakes:
    """Stand-ins for probe, track_scoreboard and detect_gameplay that record their calls."""

    def __init__(self, monkeypatch, *, hud="default", clips=(), events=(), duration=600.0):
        import cli

        self.cli = cli
        self.hud = _hud_result() if hud == "default" else hud
        self.clips = list(clips)
        self.events = list(events)
        self.probed: list[str] = []
        self.tracked: list[dict] = []
        self.detected: list[dict] = []
        info = types.SimpleNamespace(duration=duration, width=1920, height=1080, fps=60.0,
                                     has_audio=True, has_video=True, codec="h264")

        def probe(path):
            self.probed.append(str(path))
            return info

        def track_scoreboard(src_path, media_info, settings=None, **kw):
            self.tracked.append({"src": str(src_path), "info": media_info, "settings": settings, **kw})
            return self.hud

        def detect_gameplay(src_path, media_info, settings, *, weights=None, wav_path=None,
                            events_out=None, warnings_out=None):
            self.detected.append({"src": str(src_path), "info": media_info, "settings": settings})
            if events_out is not None:
                events_out.extend(self.events)
            return list(self.clips)

        monkeypatch.setattr(importlib.import_module("app.media.ffmpeg"), "probe", probe)
        monkeypatch.setattr(importlib.import_module("app.providers.hud"), "track_scoreboard",
                            track_scoreboard)
        monkeypatch.setattr(importlib.import_module("app.providers.detect_gameplay"),
                            "detect_gameplay", detect_gameplay)
        # main() would otherwise attach a log handler to pytest's captured
        # stderr, which is closed again when the test ends.
        monkeypatch.setattr(cli.logging, "basicConfig", lambda **kw: None)

    def run(self, *argv) -> int:
        return self.cli.main([str(a) for a in argv])


def _video(tmp_path: Path) -> Path:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"not a real video")
    return video


def _label_file(tmp_path: Path, *, status: str = "verified", events=None, **extra) -> Path:
    _video(tmp_path)
    data = {"version": 1, "video": "video.mp4", "game_profile": "eafc", "status": status,
            "duration": 600.0,
            "events": events if events is not None else [
                {"t": 160.0, "label": "goal", "team": "a", "score": "1-0", "minute": 11},
                {"t": 298.0, "label": "goal", "team": "b", "score": "1-1"},
                {"t": 500.0, "label": "goal", "team": "a", "score": "2-1", "minute": 90},
            ], **extra}
    return _write(tmp_path, data, "video.labels.json")


def test_cli_hud_prints_summary_matches_and_goals(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    video = _video(tmp_path)
    assert fakes.run("hud", "--input", video) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "layout:    eafc_stacked" in lines
    assert "healthy:   yes" in lines
    assert any(l.startswith("coverage:  89.0% of 0:10:00") for l in lines)
    assert any(l.startswith("streamer:  a") for l in lines)
    assert "matches (1):" in lines and "goals (3):" in lines
    match_row = lines[lines.index("matches (1):") + 2].split()
    assert match_row == ["1", "0:00:30", "1:02:05", "1-2", "3"]       # start, end, final score, goals
    first, second, third = (l.split() for l in lines[lines.index("goals (3):") + 2:][:3])
    # time, score before -> after, minute, team, streamer, importance, confidence
    assert first[:9] == ["0:02:39", "0-0", "->", "1-0", "11'", "a", "yes", "+9.0", "0.95"]
    assert second[:9] == ["0:05:00", "1-0", "->", "1-1", "-", "b", "no", "+2.0", "0.50"]
    assert "inferred" in " ".join(second[9:])
    # The third goal carries no streamer flag: the column says so instead of guessing.
    assert third[:9] == ["0:07:00", "1-1", "->", "1-2", "88'", "b", "unknown", "-3.0", "0.85"]
    assert "hidden" in " ".join(third[9:])

    # Probed and tracked once, on the real path, with the eafc profile and no refine cap.
    assert fakes.probed == [str(video.resolve())]
    assert len(fakes.tracked) == 1
    call = fakes.tracked[0]
    assert call["src"] == str(video.resolve())
    assert call["settings"].game_profile == "eafc"
    assert call["refine_limit"] is None
    assert not fakes.detected


def test_cli_hud_streamer_unknown_and_options(tmp_path, monkeypatch, capsys):
    hud = _hud_result()
    hud.streamer_side = None
    for e in hud.events:
        e.streamer = None
    fakes = _Fakes(monkeypatch, hud=hud)
    assert fakes.run("hud", "-i", _video(tmp_path), "--profile", "fifa", "--refine", "5") == 0
    out = capsys.readouterr().out
    assert "streamer:  unknown" in out.splitlines()
    assert out.count(" unknown ") == 3                                # one per goal row
    assert fakes.tracked[0]["settings"].game_profile == "fifa"
    assert fakes.tracked[0]["refine_limit"] == 5

    assert fakes.run("hud", "-i", tmp_path / "video.mp4", "--refine", "ALL") == 0
    assert fakes.tracked[-1]["refine_limit"] is None
    for bad in ("some", "-1", "2.5"):
        with pytest.raises(SystemExit) as err:
            fakes.run("hud", "-i", tmp_path / "video.mp4", "--refine", bad)
        assert err.value.code == 2


def test_cli_hud_exit_code_one_when_unhealthy_or_no_result(tmp_path, monkeypatch, capsys):
    video = _video(tmp_path)
    fakes = _Fakes(monkeypatch, hud=_hud_result(healthy=False))
    assert fakes.run("hud", "--input", video) == 1
    out = capsys.readouterr().out
    assert "healthy:   NO" in out and "goals (3):" in out             # still prints what it has

    fakes = _Fakes(monkeypatch, hud=None)
    assert fakes.run("hud", "--input", video) == 1
    assert "result:    none" in capsys.readouterr().out
    assert fakes.run("hud", "--input", video, "--json") == 1
    data = json.loads(capsys.readouterr().out)
    assert data["result"] is None and data["healthy"] is False


def test_cli_hud_missing_input_or_unreadable_video(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    assert fakes.run("hud", "--input", tmp_path / "nope.mp4") == 2
    assert not fakes.probed and not fakes.tracked

    def broken_probe(path):
        raise RuntimeError("moov atom not found")

    monkeypatch.setattr(importlib.import_module("app.media.ffmpeg"), "probe", broken_probe)
    assert fakes.run("hud", "--input", _video(tmp_path)) == 2
    assert not fakes.tracked
    assert capsys.readouterr().out == ""


def test_cli_hud_json_is_valid_json(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    video = _video(tmp_path)
    assert fakes.run("hud", "--input", video, "--json") == 0
    data = json.loads(capsys.readouterr().out)                        # nothing but JSON on stdout
    assert data["video"] == str(video.resolve())
    assert data["game_profile"] == "eafc" and data["duration"] == 600.0 and data["healthy"] is True
    assert data["labels_out"] is None
    result = data["result"]
    assert result["layout"] == "eafc_stacked" and result["healthy"] is True
    assert result["coverage"] == 0.89 and result["streamer_side"] == "a"
    assert len(result["events"]) == 3 and len(result["matches"]) == 1
    assert result["events"][0]["score_after"] == [1, 0] and result["events"][0]["minute"] == 11
    assert result["events"][1]["inferred"] is True
    assert result["matches"][0]["final_score"] == [1, 2]
    assert result["diagnostics"]["mode"] == "keyframes"


def test_cli_hud_labels_out_writes_a_draft(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    video = _video(tmp_path)
    out_file = tmp_path / "video.labels.json"
    assert fakes.run("hud", "--input", video, "--labels-out", out_file) == 0
    assert "draft labels:" in capsys.readouterr().out
    labels = ev.load_labels(out_file)
    assert labels.status == "draft" and labels.game_profile == "eafc" and labels.duration == 600.0
    assert labels.video == "video.mp4"                                # relative: it sits next to the file
    assert ev.resolve_video(labels, out_file) == video.resolve()
    assert [(e.t, e.score) for e in labels.events] == [(159.2, "1-0"), (300.1, "1-1"), (420.5, "1-2")]

    # A draft may be replaced by a newer draft without --force.
    fakes.hud.events.pop()
    assert fakes.run("hud", "--input", video, "--labels-out", out_file, "--json") == 0
    assert json.loads(capsys.readouterr().out)["labels_out"] == str(out_file)
    assert len(ev.load_labels(out_file).events) == 2

    # Elsewhere on disk the video is referenced by its absolute path.
    other = tmp_path / "labels" / "x.json"
    assert fakes.run("hud", "--input", video, "--labels-out", other) == 0
    assert ev.resolve_video(ev.load_labels(other), other) == video.resolve()

    # No result: nothing is written.
    fakes.hud = None
    missing = tmp_path / "none.json"
    assert fakes.run("hud", "--input", video, "--labels-out", missing) == 1
    assert not missing.exists()


def test_cli_hud_never_overwrites_verified_labels_without_force(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    video = _video(tmp_path)
    verified = _label_file(tmp_path, status="verified")
    before = verified.read_bytes()

    assert fakes.run("hud", "--input", video, "--labels-out", verified) == 2
    assert verified.read_bytes() == before
    assert not fakes.tracked                         # refused before the (slow) scan
    assert capsys.readouterr().out == ""

    # A file that cannot be parsed might be verified labels with a typo: protected too.
    broken = _write(tmp_path, '{"status": "verified", "events": [', "broken.json")
    assert fakes.run("hud", "--input", video, "--labels-out", broken) == 2
    assert broken.read_text(encoding="utf-8").startswith('{"status": "verified"')
    assert not fakes.tracked

    assert fakes.run("hud", "--input", video, "--labels-out", verified, "--force") == 0
    replaced = ev.load_labels(verified)
    assert replaced.status == "draft" and len(replaced.events) == 3
    assert fakes.run("hud", "--input", video, "--labels-out", broken, "--force") == 0
    assert ev.load_labels(broken).status == "draft"


def test_cli_eval_hud_reports_and_exits_zero(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    labels = _label_file(tmp_path)        # truth 160, 298, 500; the tracker says 159.23, 300.07, 420.5
    assert fakes.run("eval", "--labels", labels) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "hud detector vs verified labels: video.mp4"
    assert "WARNING" not in out and "DRAFT" not in out
    assert next(l for l in lines if l.startswith("goal")).split() == [
        "goal", "3", "3", "2", "1", "1", "0.667", "0.667", "0.667"]
    assert "tolerance: +/-6 s" in lines
    assert "  0:08:20  goal  2-1  90'" in lines        # the missed goal, with its details
    assert "  0:07:00  goal" in lines                  # the extra one
    assert "clips:" not in out
    # The video named in the label file was probed and tracked with the file's profile.
    video = (tmp_path / "video.mp4").resolve()
    assert fakes.probed == [str(video)]
    assert fakes.tracked[0]["src"] == str(video)
    assert fakes.tracked[0]["settings"].game_profile == "eafc"
    assert fakes.tracked[0]["refine_limit"] is None
    assert not fakes.detected

    # A tighter tolerance changes the numbers: 300.07 vs 298 no longer matches.
    assert fakes.run("eval", "--labels", labels, "--tolerance", "1", "--refine", "0") == 0
    lines = capsys.readouterr().out.splitlines()
    assert next(l for l in lines if l.startswith("goal")).split()[3:6] == ["1", "2", "2"]
    assert fakes.tracked[-1]["refine_limit"] == 0


def test_cli_eval_warns_about_draft_labels(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    labels = _label_file(tmp_path, status="draft")
    assert fakes.run("eval", "--labels", labels) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "hud detector vs draft labels: video.mp4"
    warning = lines[1]
    assert warning.startswith("WARNING: ") and "DRAFT" in warning
    assert "not trustworthy" in warning and "verified" in warning

    assert fakes.run("eval", "--labels", labels, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["labels_status"] == "draft"
    assert len(data["warnings"]) == 1 and "DRAFT" in data["warnings"][0]


def test_cli_eval_warns_when_the_tracker_has_nothing(tmp_path, monkeypatch, capsys):
    labels = _label_file(tmp_path)
    fakes = _Fakes(monkeypatch, hud=None)
    assert fakes.run("eval", "--labels", labels) == 0       # no gate given: still exit code 0
    out = capsys.readouterr().out
    assert "WARNING: the scoreboard tracker returned no result" in out
    assert next(l for l in out.splitlines() if l.startswith("goal")).split()[1:6] == ["3", "0", "0", "0", "3"]

    fakes = _Fakes(monkeypatch, hud=_hud_result(healthy=False))
    assert fakes.run("eval", "--labels", labels) == 0
    assert "WARNING: the scoreboard was not read reliably" in capsys.readouterr().out


def test_cli_eval_gates_set_the_exit_code(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    labels = _label_file(tmp_path)                           # precision = recall = 2/3
    assert fakes.run("eval", "--labels", labels, "--min-recall", "0.6", "--min-precision", "0.6") == 0
    assert "FAILED" not in capsys.readouterr().out

    assert fakes.run("eval", "--labels", labels, "--min-recall", "0.9") == 1
    out = capsys.readouterr().out
    assert "FAILED: recall 0.667 is below the required 0.9" in out
    assert "precision 0.667 is below" not in out

    assert fakes.run("eval", "--labels", labels, "--min-precision", "0.7") == 1
    assert "FAILED: precision 0.667 is below the required 0.7" in capsys.readouterr().out

    assert fakes.run("eval", "--labels", labels, "--min-precision", "0.7", "--min-recall", "0.7", "--json") == 1
    data = json.loads(capsys.readouterr().out)
    assert data["passed"] is False and len(data["failed"]) == 2
    assert data["min_precision"] == 0.7 and data["min_recall"] == 0.7

    # A gate at exactly the measured value is met.
    hud = _hud_result()
    hud.events = hud.events[:1]                              # one prediction, one hit: precision 1.0
    fakes = _Fakes(monkeypatch, hud=hud)
    assert fakes.run("eval", "--labels", labels, "--min-precision", "1.0") == 0
    assert fakes.run("eval", "--labels", labels, "--min-precision", "1.0", "--min-recall", "0.5") == 1


def test_cli_eval_exit_code_two_for_bad_labels_or_missing_video(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    assert fakes.run("eval", "--labels", tmp_path / "missing.json") == 2

    assert fakes.run("eval", "--labels", _write(tmp_path, "{ nope", "garbage.json")) == 2

    invalid = _valid()
    invalid["events"][0]["t"] = -5
    assert fakes.run("eval", "--labels", _write(tmp_path, invalid, "invalid.json")) == 2

    # A valid label file whose video is not there.
    assert fakes.run("eval", "--labels", _write(tmp_path, _valid(), "orphan.json")) == 2

    # Gates do not turn these into exit code 1.
    assert fakes.run("eval", "--labels", tmp_path / "missing.json", "--min-recall", "0.5") == 2

    labels = _label_file(tmp_path)
    assert fakes.run("eval", "--labels", labels, "--tolerance", "-1") == 2
    assert not fakes.probed and not fakes.tracked and not fakes.detected
    assert capsys.readouterr().out == ""                     # no report was printed for any of them

    def broken_probe(path):
        raise RuntimeError("moov atom not found")

    monkeypatch.setattr(importlib.import_module("app.media.ffmpeg"), "probe", broken_probe)
    assert fakes.run("eval", "--labels", labels) == 2
    assert not fakes.tracked


def test_cli_eval_reports_the_reason_for_a_rejected_label_file(tmp_path, monkeypatch, caplog):
    fakes = _Fakes(monkeypatch)
    invalid = _valid()
    invalid["events"][1]["t"] = "soon"
    with caplog.at_level("ERROR", logger="clipforge.cli"):
        assert fakes.run("eval", "--labels", _write(tmp_path, invalid, "invalid.json")) == 2
    assert "invalid.json" in caplog.text and "events[1].t" in caplog.text


def test_cli_eval_json_is_valid_json(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    labels = _label_file(tmp_path)
    assert fakes.run("eval", "--labels", labels, "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["detector"] == "hud" and data["labels_status"] == "verified"
    assert data["labels_file"] == str(labels.resolve())
    assert data["video"] == str((tmp_path / "video.mp4").resolve())
    assert data["game_profile"] == "eafc" and data["tolerance"] == 6.0
    assert data["passed"] is True and data["failed"] == [] and data["warnings"] == []
    goal = data["labels"]["goal"]
    assert (goal["tp"], goal["fp"], goal["fn"]) == (2, 1, 1)
    assert goal["precision"] == 0.6667 and goal["recall"] == 0.6667
    assert data["overall"]["mean_abs_error"] == round((0.77 + 2.07) / 2, 3)
    assert data["overall"]["bias"] == round((-0.77 + 2.07) / 2, 3)
    assert [e["time"] for e in data["overall"]["missed"]] == ["0:08:20"]
    assert [e["t"] for e in data["overall"]["extra"]] == [420.5]
    assert data["clips"] is None


def test_cli_eval_gameplay_scores_goal_events_and_clips(tmp_path, monkeypatch, capsys):
    from app.models import ContentType, DetectedEvent

    events = [
        DetectedEvent(t=161.0, source="hud", label="goal", detail="Scoreboard 0-0 -> 1-0", confidence=0.95),
        DetectedEvent(t=163.0, source="audio", label="goal", detail="crowd", confidence=0.5),  # duplicate
        DetectedEvent(t=299.0, source="hud", label="goal", confidence=0.6),
        DetectedEvent(t=50.0, source="ocr", label="victory", confidence=0.9),                  # not a goal
        DetectedEvent(t=499.0, source="cue", label="kill", confidence=0.9),                    # not a goal
    ]
    clips = [types.SimpleNamespace(start=150.0, end=175.0, score=90),
             types.SimpleNamespace(start=40.0, end=60.0, score=70),
             types.SimpleNamespace(start=480.0, end=510.0, score=60)]
    fakes = _Fakes(monkeypatch, events=events, clips=clips)
    labels = _label_file(tmp_path)                           # truth 160, 298, 500
    assert fakes.run("eval", "--labels", labels, "--detector", "gameplay") == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "gameplay detector vs verified labels: video.mp4"
    # goals: 161 and 299 hit, 163 is extra, 500 is missed; "victory" and "kill" are not scored
    assert next(l for l in lines if l.startswith("goal")).split() == [
        "goal", "3", "3", "2", "1", "1", "0.667", "0.667", "0.667"]
    assert not any(l.startswith(("victory", "kill", "all")) for l in lines)
    assert "  0:02:43  goal" in lines
    assert ("clips: 3 scored, 2 contain a labelled event (hit rate 0.667); 2 of 3 labelled "
            "events are inside a clip (coverage 0.667)") in lines
    assert "empty clips (1):" in lines and "  0:00:40 - 0:01:00" in lines

    assert len(fakes.detected) == 1 and not fakes.tracked
    call = fakes.detected[0]
    assert call["src"] == str((tmp_path / "video.mp4").resolve())
    assert call["settings"].game_profile == "eafc"
    assert call["settings"].content_type == ContentType.gameplay

    assert fakes.run("eval", "--labels", labels, "--detector", "gameplay", "--json",
                     "--min-recall", "0.6") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["detector"] == "gameplay"
    assert (data["clips"]["clips"], data["clips"]["hits"], data["clips"]["events_covered"]) == (3, 2, 2)
    assert data["clips"]["clip_hit_rate"] == 0.6667


def test_cli_eval_honours_labelled_ranges(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)                              # tracker: 159.23, 300.07, 420.5
    labels = _label_file(tmp_path, labelled_ranges=[[0, 350]],
                         events=[{"t": 160.0, "label": "goal"}, {"t": 298.0, "label": "goal"}])
    assert fakes.run("eval", "--labels", labels, "--min-precision", "1", "--min-recall", "1") == 0
    out = capsys.readouterr().out
    assert next(l for l in out.splitlines() if l.startswith("goal")).split()[1:6] == ["2", "2", "2", "0", "0"]
    assert "ignored: 1 unmatched predictions outside the labelled ranges" in out


def test_cli_eval_exit_code_two_when_the_detector_crashes(tmp_path, monkeypatch, capsys):
    fakes = _Fakes(monkeypatch)
    labels = _label_file(tmp_path)

    def boom(*a, **kw):
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(importlib.import_module("app.providers.hud"), "track_scoreboard", boom)
    assert fakes.run("eval", "--labels", labels, "--min-recall", "0.5") == 2
    assert capsys.readouterr().out == ""


def test_cli_eval_report_survives_a_console_that_cannot_show_a_note(tmp_path, monkeypatch):
    # A traceback here would exit with code 1, which is reserved for "gate not met".
    fakes = _Fakes(monkeypatch, hud=None)                    # nothing detected: the label is listed as missed
    labels = _label_file(tmp_path, events=[{"t": 160.0, "label": "goal", "note": "Abstoß → Tor"}])
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii", newline="\n"))
    code = fakes.run("eval", "--labels", labels)
    sys.stdout.flush()
    assert code == 0
    assert "  0:02:40  goal  (Absto? ? Tor)" in raw.getvalue().decode("ascii").splitlines()

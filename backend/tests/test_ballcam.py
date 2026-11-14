"""Tests for the ball-following crop (``app.pipeline.ballcam``).

The detector and the frame source are replaced by fakes, so these run with no
model, no GPU and no video.

    cd backend && python -m pytest tests/test_ballcam.py -q
"""
from __future__ import annotations

import os
import tempfile
import types

os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

from app.models import ImportSettings, LayoutType, Rect, ReframeKeyframe  # noqa: E402
from app.pipeline import ballcam as B  # noqa: E402
from app.pipeline import orchestrator as O  # noqa: E402

W, H = 2560, 1440
INFO = types.SimpleNamespace(width=W, height=H, duration=600.0)
C = B.Candidate


def box(x: float, y: float, size: float = 14.0, conf: float = 0.6):
    """A ``size``-px detection box centred on frame fractions (x, y)."""
    cx, cy = x * W, y * H
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, conf)


def flat_box(x: float, y: float, conf: float = 0.79):
    """A 14x7-px box: how the penalty spot, a white disc on the grass seen from the
    raised camera, reads in EA FC 27 at 1440p (the ball itself reads 14x14)."""
    cx, cy = x * W, y * H
    return (cx - 7.0, cy - 3.5, cx + 7.0, cy + 3.5, conf)


def test_follows_football_only(monkeypatch):
    assert B.follows("eafc") and B.follows("FIFA") and B.follows("EA FC")
    assert not B.follows("valorant") and not B.follows(None)
    monkeypatch.setenv("CLIPFORGE_BALLCAM", "off")
    assert not B.follows("eafc")


def test_candidates_skip_hud_shapes_and_big_round_things():
    boxes = [box(0.5, 0.3),                 # the ball
             box(0.5, 0.80),                # the ball icon in the radar
             box(0.5, 0.95),                # player name bars
             box(0.05, 0.05),               # scoreboard
             box(0.6, 0.3, size=44.0),      # a name-tag icon: too big for the ball
             box(0.7, 0.4, conf=0.05)]      # too unsure
    assert B.candidates_from_boxes(boxes, W, H) == [C(0.5, 0.3, 0.6)]


def test_candidates_skip_flat_pitch_markings():
    # In the goal clips the penalty spot read as a ball at up to 0.79, more sure than
    # the ball itself; it is twice as wide as tall.
    boxes = [flat_box(0.24, 0.47), box(0.51, 0.27, conf=0.45)]
    assert B.candidates_from_boxes(boxes, W, H) == [C(0.51, 0.27, 0.45)]


def test_best_track_prefers_the_ball_over_a_brief_white_boot():
    samples = []
    for i in range(20):
        cs = [C(0.40 + 0.01 * i, 0.30, 0.7)]                 # the ball, drifting right
        if 5 <= i < 10:
            cs.append(C(0.05 + 0.04 * (i - 5), 0.42, 0.5))    # a boot sprinting on the left
        if i in (12, 13):
            cs = []                                          # hidden behind a player
        samples.append((i * 0.25, cs))
    track = B.best_track(samples)
    assert [round(x, 3) for _t, x in track] == [
        round(0.40 + 0.01 * i, 3) for i in range(20) if i not in (12, 13)]


def test_best_track_follows_the_ball_through_a_camera_cut():
    samples = [(i * 0.25, [C(0.30, 0.4, 0.8)]) for i in range(8)]
    samples += [(2.0 + i * 0.25, [C(0.75, 0.5, 0.8)]) for i in range(8)]
    track = B.best_track(samples)
    assert len(track) == 16 and track[0][1] == 0.30 and track[-1][1] == 0.75
    assert B.best_track([]) == [] and B.best_track([(0.0, [])]) == []


def _path(xs, conf: float = 0.5, t0: float = 0.0):
    """A path through one candidate per 0.25-s sample."""
    return [(t0 + i * 0.25, C(round(x, 3), 0.4, conf)) for i, x in enumerate(xs)]


def test_drop_detours_skips_short_visits_and_keeps_real_moves():
    ramp = [0.40 + 0.01 * i for i in range(12)]
    # One sample on a white boot 0.15 to the right, two on something 0.2 to the left.
    visits = [x + 0.15 if i == 4 else x - 0.20 if i in (8, 9) else x for i, x in enumerate(ramp)]
    assert B.drop_detours(_path(visits)) == [p for i, p in enumerate(_path(ramp))
                                             if i not in (4, 8, 9)]
    # Followed: a camera cut (the ball reappears elsewhere and stays), a ball that stays
    # out longer than DETOUR_S, a wobble smaller than DETOUR_X, and a short visit to a
    # sure ball: the shot into the net just before the game cuts to the celebration.
    cut = _path([0.30 if i < 6 else 0.75 for i in range(12)])
    held = _path([0.60 if 4 <= i < 8 else 0.40 for i in range(12)])
    wobble = _path([0.52 if i % 2 else 0.50 for i in range(12)])
    shot = _path(ramp[:6]) + [(1.5, C(0.25, 0.42, B.SURE_CONF))] + _path([0.45] * 4, t0=1.75)
    for path in (cut, held, wobble, shot):
        assert B.drop_detours(path) == path
    assert B.drop_detours([]) == [] and B.drop_detours(_path(ramp[:2])) == _path(ramp[:2])


def test_camera_path_holds_in_the_dead_zone_pans_at_a_limit_and_stays_in_frame():
    times = [i * 0.25 for i in range(41)]
    half = 0.158
    still = B.camera_path([(0.0, 0.50), (10.0, 0.53)], times, half)
    assert all(abs(c - 0.50) < 1e-9 for _t, c in still)        # drift inside the dead zone
    fast = B.camera_path([(0.0, 0.30), (1.0, 0.30), (1.25, 0.80), (10.0, 0.80)], times, half)
    steps = [b - a for (_, a), (_, b) in zip(fast, fast[1:])]
    assert max(abs(s) for s in steps) <= B.MAX_SPEED * 0.25 + 1e-9
    assert abs(fast[-1][1] - (0.80 - B.DEAD_ZONE)) < 0.01       # ends with the ball in view
    edge = B.camera_path([(0.0, 0.02)], times, half)
    assert all(abs(c - half) < 1e-9 for _t, c in edge)          # never past the frame edge
    assert B.camera_path([], times, half) == []


def _source(start: float, duration: float, n: int):
    def source(region, **kw):
        assert region == (0, 0, W, H) and kw["keyframes"] is False
        assert kw["fps"] == B.BALL_FPS and kw["pix_fmt"] == "bgr24"
        assert kw["start"] == start and kw["duration"] == duration
        for i in range(n):
            yield start + i / B.BALL_FPS, i                     # the "frame" is its index
    return source


def test_ball_keyframes_pan_with_the_ball():
    def detector(frames):
        assert len(frames) <= B.BATCH
        return [[box(0.30 + 0.005 * f, 0.4)] if f % 4 else [] for f in frames]

    kfs = B.ball_keyframes("x.mp4", 100.0, 120.0, INFO, frame_source=_source(100.0, 20.0, 80),
                           detector=detector)
    assert kfs and kfs[0].t == 0.0 and kfs[-1].t <= 20.0
    assert kfs[-1].cx > kfs[0].cx + 0.25                        # followed the ball to the right


def test_ball_keyframes_never_pan_out_and_back_for_a_moment():
    # The ball drifts slowly right. At sample 30 it is hidden and a white boot 0.15 to
    # its right reads as a ball; at samples 50-51 the penalty spot 0.2 to its left reads
    # as a surer ball than the ball itself. The crop visits neither: it only moves right.
    def detector(frames):
        out = []
        for f in frames:
            x = 0.40 + 0.002 * f
            if f == 30:
                out.append([box(x + 0.15, 0.45, conf=0.5)])
            elif f in (50, 51):
                out.append([box(x, 0.40, conf=0.45), flat_box(x - 0.20, 0.47)])
            else:
                out.append([box(x, 0.40)])
        return out

    kfs = B.ball_keyframes("x.mp4", 100.0, 120.0, INFO, frame_source=_source(100.0, 20.0, 80),
                           detector=detector)
    cxs = [k.cx for k in kfs]
    assert all(b >= a for a, b in zip(cxs, cxs[1:])), cxs
    assert cxs[-1] > cxs[0] + 0.08                              # and still follows the ball


def test_ball_keyframes_give_up_without_enough_ball():
    rare = B.ball_keyframes("x.mp4", 100.0, 120.0, INFO, frame_source=_source(100.0, 20.0, 80),
                            detector=lambda fs: [[box(0.5, 0.4)] if f % 10 == 0 else [] for f in fs])
    assert rare is None                                         # on 10 % of the samples only
    assert B.ball_keyframes("x.mp4", 0.0, 0.5, INFO, detector=lambda fs: []) is None
    flat = types.SimpleNamespace(width=0, height=0, duration=10.0)
    assert B.ball_keyframes("x.mp4", 0.0, 10.0, flat, detector=lambda fs: []) is None


def test_gameplay_reframe_follows_the_ball_in_football_only(monkeypatch):
    kfs = [ReframeKeyframe(t=0.0, cx=0.4), ReframeKeyframe(t=5.0, cx=0.6)]
    calls = []
    monkeypatch.setattr(O.ballcam_mod, "ball_keyframes",
                        lambda *a, **kw: calls.append(a) or kfs)
    monkeypatch.setattr(O.facecam_mod, "action_center", lambda *a, **kw: 0.55)
    engine = O.Engine.__new__(O.Engine)                        # no __init__ side effects
    clip = types.SimpleNamespace(id="c1", start=10.0, end=40.0)
    eafc = ImportSettings(game_profile="eafc")

    r = engine._gameplay_reframe("m.mp4", clip, None, eafc, vertical=True, info=INFO)
    assert r.layout == LayoutType.fill and r.tracked and r.keyframes == kfs
    assert calls == [("m.mp4", 10.0, 40.0, INFO)]
    # Not followed: another game, a landscape output, a facecam layout, no media info.
    other = engine._gameplay_reframe("m.mp4", clip, None, ImportSettings(game_profile="fortnite"),
                                     vertical=True, info=INFO)
    assert other.layout == LayoutType.center and other.keyframes[0].cx == 0.55
    assert not engine._gameplay_reframe("m.mp4", clip, None, eafc, vertical=False,
                                        info=INFO).tracked
    cam = Rect(x=0.0, y=0.0, w=0.2, h=0.2)
    assert engine._gameplay_reframe("m.mp4", clip, cam, eafc, vertical=True,
                                    info=INFO).layout == LayoutType.split
    assert not engine._gameplay_reframe("m.mp4", clip, None, eafc, vertical=True).tracked
    assert len(calls) == 1
    # The ball was not seen often enough: the motion centre, as before.
    monkeypatch.setattr(O.ballcam_mod, "ball_keyframes", lambda *a, **kw: None)
    r = engine._gameplay_reframe("m.mp4", clip, None, eafc, vertical=True, info=INFO)
    assert r.layout == LayoutType.center and not r.tracked and r.keyframes[0].cx == 0.55
    # A crash in the ball cam never fails the clip.
    monkeypatch.setattr(O.ballcam_mod, "ball_keyframes",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no GPU")))
    assert engine._gameplay_reframe("m.mp4", clip, None, eafc, vertical=True,
                                    info=INFO).keyframes[0].cx == 0.55

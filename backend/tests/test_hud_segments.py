"""Tests for per-game segments (``app.providers.hud.segments``).

One recording can hold several games. These tests cover finding Valorant by
its top bar, turning per-sample flags into spans, running the kill tracker per
span, and the scoreboard result's view of "its own" part of a recording.

    cd backend && python -m pytest tests/test_hud_segments.py -q
"""
from __future__ import annotations

import os
import tempfile
import types

os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

import numpy as np  # noqa: E402

from app.providers.hud import detect as D  # noqa: E402
from app.providers.hud import killfeed as K  # noqa: E402
from app.providers.hud import segments as S  # noqa: E402
from app.providers.hud.layouts import LAYOUTS, layouts_for  # noqa: E402

W, H = 2560, 1440
INFO = types.SimpleNamespace(width=W, height=H, duration=1800.0)
REGION = S.topbar_region(W, H)


def topbar(kind: str) -> np.ndarray:
    """The top strip: 'valorant' (green band left, red band right) or another game."""
    _x, _y, w, h = REGION
    img = np.zeros((h, w, 3), np.uint8)
    if kind == "valorant":
        img[:] = (60, 70, 60)
        c = w // 2
        img[:, :c - 200] = (70, 120, 80)        # BGR: green-teal ally band
        img[:, c + 200:] = (60, 60, 150)        # red enemy band
    elif kind == "grass":
        img[:] = (40, 140, 60)
    elif kind == "sky":
        img[:] = (200, 150, 120)
    elif kind == "red":
        img[:] = (40, 40, 180)                  # red on both sides is not a top bar
    return img


def source_of(kind_at, seconds: float, step: float = 4.0):
    calls = []

    def source(region, **kw):
        calls.append(kw)
        assert region == REGION and kw["pix_fmt"] == "bgr24"
        t = 0.0
        while t < seconds:
            yield t, topbar(kind_at(t))
            t += step
    source.calls = calls
    return source


def test_topbar_region_is_even_and_centred():
    for w, h in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]:
        x, y, rw, rh = S.topbar_region(w, h)
        assert x % 2 == y % 2 == rw % 2 == rh % 2 == 0
        assert abs((x + rw / 2) - w / 2) <= 2 and y + rh <= h


def test_topbar_present_compares_the_two_sides():
    assert S.topbar_present(topbar("valorant"), H)
    for other in ("grass", "sky", "red", "black"):
        assert not S.topbar_present(topbar(other), H)
    assert not S.topbar_present(topbar("valorant")[:, ::-1], H)   # mirrored: red left


def test_spans_from_flags_needs_a_dense_stretch():
    flags = [(float(t), 300 <= t < 900) for t in range(0, 1800, 4)]
    flags += [(1500.0 + i * 40, True) for i in range(5)]           # stray hits elsewhere
    assert S.spans_from_flags(flags, 1800.0) == [(300.0 - S.PAD_S, 900.0 + S.PAD_S)]


def test_spans_from_flags_bridges_one_weak_minute_and_drops_short_runs():
    def on(t):
        return (120 <= t < 360 and not 240 <= t < 300) or 1200 <= t < 1260
    spans = S.spans_from_flags([(float(t), on(t)) for t in range(0, 1800, 4)], 1800.0)
    assert spans == [(120.0 - S.PAD_S, 360.0 + S.PAD_S)]


def test_valorant_spans_finds_the_valorant_part_of_a_mixed_recording():
    def kind(t):
        return "valorant" if 120 <= t < 900 else ("grass" if t < 1400 else "sky")
    src = source_of(kind, 1800.0)
    spans = S.valorant_spans("x.mp4", INFO, frame_source=src)
    assert spans == [(120.0 - S.PAD_S, 900.0 + S.PAD_S)]
    assert src.calls[0]["keyframes"] is True and len(src.calls) == 1


def test_valorant_spans_is_empty_for_another_game_and_none_without_a_size():
    assert S.valorant_spans("x.mp4", INFO, frame_source=source_of(lambda t: "grass", 600.0)) == []
    assert S.valorant_spans("x.mp4", types.SimpleNamespace(width=0, height=0, duration=10.0)) is None


def test_valorant_spans_falls_back_to_a_fixed_rate_with_sparse_keyframes():
    src = source_of(lambda t: "valorant", 1800.0, step=30.0)
    S.valorant_spans("x.mp4", INFO, frame_source=src)
    assert [c["keyframes"] for c in src.calls] == [True, False]
    assert src.calls[1]["fps"] == S.FALLBACK_FPS


def test_cluster_spans_joins_matches_and_pads():
    spans = [(100.0, 700.0), (900.0, 1500.0), (4000.0, 4600.0)]
    assert S.cluster_spans(spans, 5000.0) == [(0.0, 1680.0), (3820.0, 4780.0)]


def test_track_killfeed_spans_merges_and_renumbers(monkeypatch):
    def fake(src, info, *, start, duration, **kw):
        kills = [K.KillEvent(t=start + 10, kind="kill", confidence=0.9, t_last=start + 12),
                 K.KillEvent(t=start + 14, kind="kill", confidence=0.9, t_last=start + 16)]
        death = K.KillEvent(t=start + 30, kind="death", confidence=0.9, t_last=start + 34)
        return K.KillfeedResult(events=kills + [death], rounds=K.group_rounds(kills),
                                samples=int(duration * 2), sample_interval=0.5, active_s=40.0,
                                healthy=True, diagnostics={"round_starts": [start], "seconds": 1.0})

    monkeypatch.setattr(K, "track_killfeed", fake)
    res = S.track_killfeed_spans("x.mp4", INFO, [(100.0, 400.0), (1000.0, 1300.0)])
    assert [e.t for e in res.kills] == [110.0, 114.0, 1010.0, 1014.0]
    assert [r.index for r in res.rounds] == [1, 2]
    assert [e.round_index for e in res.kills] == [1, 1, 2, 2]
    assert res.samples == 1200 and res.active_s == 80.0 and res.healthy
    assert res.diagnostics["round_starts"] == [100.0, 1000.0]


def test_track_killfeed_spans_returns_none_when_nothing_reads(monkeypatch):
    monkeypatch.setattr(K, "track_killfeed", lambda *a, **kw: None)
    assert S.track_killfeed_spans("x.mp4", INFO, [(0.0, 60.0)]) is None


def _hud_result(**kw) -> D.HudResult:
    base = dict(layout="eafc_stacked", events=[], matches=[], in_play=[(1300.0, 2000.0)],
                coverage=700.0 / 5800.0, healthy=True, samples=100, sample_interval=4.0)
    base.update(kw)
    return D.HudResult(**base)


def test_scoreboard_gate_leaves_other_games_alone():
    mixed = _hud_result(game_spans=[(1100.0, 2200.0)])
    assert mixed.allows(1500.0)            # in the match
    assert not mixed.allows(1150.0)        # this game's menu
    assert mixed.allows(300.0) and mixed.allows(4000.0)     # another game: no say
    assert abs(mixed.game_coverage - 700.0 / 1100.0) < 1e-6
    whole = _hud_result()                  # no spans: the whole recording is this game
    assert not whole.allows(300.0) and whole.allows(1500.0)
    assert whole.game_coverage == whole.coverage


def test_every_profile_tries_the_scoreboard_layouts():
    assert layouts_for("eafc") == [LAYOUTS["eafc"]]
    for name in ("valorant", "fortnite", "auto", "generic", None):
        assert layouts_for(name) == list(LAYOUTS.values())

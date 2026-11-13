"""Tests for Valorant round results and spike plants (``app.providers.hud.valorant_rounds``).

The top bar shows both team scores and, while the spike is planted, a red
icon in place of the round timer. These tests cover reading one frame,
decoding noisy score reads into round results, splitting matches, and how
the results mark kill rounds and raise their clips' scores.

    cd backend && python -m pytest tests/test_hud_rounds.py -q
"""
from __future__ import annotations

import os
import tempfile
import types

os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.models import ImportSettings  # noqa: E402
from app.providers import detect_gameplay as DG  # noqa: E402
from app.providers.hud import killfeed as K  # noqa: E402
from app.providers.hud import valorant_rounds as V  # noqa: E402

W, H = 2560, 1440
REGION = V.bar_region(W, H)
INFO = types.SimpleNamespace(width=W, height=H, duration=3600.0)


def S(t, ally=None, enemy=None, *, spike=False) -> V.BarSample:
    return V.BarSample(t=float(t), ally=ally, enemy=enemy, spike=spike)


def scripted(changes, end, *, step=4.0, empty_every=0, misreads=None) -> list[V.BarSample]:
    """A sample every ``step`` s of a score that follows ``changes`` [(from_t, ally, enemy)].

    Every ``empty_every``-th sample reads nothing; ``misreads`` maps a sample
    index to the (ally, enemy) pair read there instead of the true one.
    """
    misreads = misreads or {}
    out = []
    for i in range(int(end / step)):
        t = i * step
        ally, enemy = [(a, e) for tc, a, e in changes if tc <= t][-1]
        if empty_every and i % empty_every == 0:
            ally = enemy = None
        ally, enemy = misreads.get(i, (ally, enemy))
        out.append(S(t, ally, enemy))
    return out


def bar_image(ally: str | None = "3", enemy: str | None = "7", *, spike=False) -> np.ndarray:
    """The top-bar strip: grey plates, white score digits, optional red spike icon."""
    _x, _y, w, h = REGION
    img = np.full((h, w, 3), 90, np.uint8)
    for dx, text in ((-V.SCORE_DX, ally), (V.SCORE_DX, enemy)):
        if text is not None:
            cx = w // 2 + int(dx * H)
            cv2.putText(img, text, (cx - 12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 255, 255), 3)
    if spike:
        cv2.circle(img, (w // 2, h // 2), 20, (30, 30, 220), -1)
    return img


class FakeReader:
    name = "fake"

    def __init__(self, text="4", conf=0.9):
        self.text, self.conf, self.calls = text, conf, 0

    def read_digits(self, img):
        self.calls += 1
        return self.text, self.conf


def _kill(t: float) -> K.KillEvent:
    return K.KillEvent(t=float(t), kind="kill", confidence=0.9, t_last=float(t) + 2.0)


def _result(kills, starts=None) -> K.KillfeedResult:
    return K.KillfeedResult(events=kills, rounds=K.group_rounds(kills, starts=starts),
                            samples=100, sample_interval=0.5, active_s=60.0, healthy=True,
                            diagnostics={"round_starts": list(starts or [])})


# --- one frame ----------------------------------------------------------------

def test_bar_region_is_even_and_centred():
    for w, h in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]:
        x, y, rw, rh = V.bar_region(w, h)
        assert x % 2 == y % 2 == rw % 2 == rh % 2 == 0
        assert abs((x + rw / 2) - w / 2) <= 2 and y + rh <= h


def test_spike_icon_needs_red_where_the_timer_is():
    assert V.spike_icon(bar_image(spike=True), H)
    assert not V.spike_icon(bar_image(), H)
    red_plates = bar_image()
    red_plates[:, :60] = (30, 30, 220)               # red at the edge is not the icon
    assert not V.spike_icon(red_plates, H)


def test_digit_image_needs_a_digit_but_not_a_white_wall():
    cols = V._cols(REGION[2], H, -V.SCORE_DX, V.SCORE_HALF)
    assert V.digit_image(bar_image()[:, cols]) is not None
    assert V.digit_image(bar_image(None, None)[:, cols]) is None
    assert V.digit_image(np.full((REGION[3], 52, 3), 250, np.uint8)) is None


def test_score_reads_need_confidence_and_a_plausible_value():
    img = bar_image()[:, V._cols(REGION[2], H, -V.SCORE_DX, V.SCORE_HALF)]
    assert V._read_score(FakeReader("4", 0.9), img) == 4
    assert V._read_score(FakeReader("4", 0.2), img) is None      # unsure
    assert V._read_score(FakeReader("40", 0.9), img) is None     # no match gets that far


def test_read_bar_reads_scores_and_the_spike_icon_inside_the_spans():
    frames = [(0.0, bar_image()), (4.0, bar_image(None, None, spike=True)),
              (8.0, bar_image()), (500.0, bar_image())]

    def source(region, **kw):
        assert region == REGION and kw == {"keyframes": True, "pix_fmt": "bgr24"}
        yield from frames

    reader = FakeReader("4")
    samples = V.read_bar("x.mp4", INFO, spans=[(0.0, 100.0)], reader=reader, frame_source=source)
    assert [(s.t, s.ally, s.enemy, s.spike) for s in samples] == [
        (0.0, 4, 4, False), (4.0, None, None, True), (8.0, 4, 4, False)]
    assert reader.calls == 4                          # no OCR without a digit on the plate


# --- decoding -----------------------------------------------------------------

def test_monotone_keeps_the_longest_series_that_never_falls():
    values = [0, 0, 0, 12, 1, 1, 13, 2, 2, 19, 3, 3]
    kept = V._monotone([(float(i), v) for i, v in enumerate(values)])
    assert [v for _t, v in kept] == [0, 0, 0, 1, 1, 2, 2, 3, 3]


def test_decode_rounds_from_noisy_reads():
    changes = [(0, 0, 0), (100, 0, 1), (200, 1, 1), (300, 1, 2), (400, 2, 2)]
    # Every third read is empty; three reads gain a digit (10 for 0, 11 for 1, 12 for 2).
    samples = scripted(changes, 500, empty_every=3,
                       misreads={10: (10, 0), 60: (1, 11), 110: (12, 2)})
    res = V.decode_rounds(samples)
    assert [(r.t, r.winner, r.score) for r in res] == [
        (100.0, "enemy", (0, 1)), (200.0, "ally", (1, 1)),
        (304.0, "enemy", (1, 2)), (400.0, "ally", (2, 2))]
    assert all(r.exact and r.t_low < r.t for r in res)
    assert res[0].t_low == 92.0                       # the last read that still showed 0


def test_a_single_read_counts_only_when_it_fills_a_plus_one_step():
    samples = ([S(t, 0, 0 if t == 0 else None) for t in range(0, 40, 4)]   # enemy 0 read once
               + [S(t, 0, 1) for t in range(40, 80, 4)]
               + [S(t, 0, 3) for t in range(80, 120, 4)]                   # 2 never read
               + [S(120, 1, 3)])                                           # final score, read once
    res = V.decode_rounds(samples)
    assert [(r.t, r.winner, r.score, r.exact) for r in res] == [
        (40.0, "enemy", (0, 1), True),
        (80.0, "enemy", (0, 2), False), (80.0, "enemy", (0, 3), False),
        (120.0, "ally", (1, 3), True)]
    # A lone read that does not fit the +1 progression is a misread.
    lone = [S(t, 0, 1) for t in range(0, 40, 4)] + [S(t, 0, 2) for t in range(40, 80, 4)] + [S(80, 0, 9)]
    assert [r.score for r in V.decode_rounds(lone)] == [(0, 2)]


def test_decode_rounds_splits_matches_at_a_reset_and_at_long_gaps():
    first = [S(0, 2, 2), S(4, 2, 2), S(8, 3, 2), S(12, 3, 2)]
    again = [S(16, 0, 0), S(20, 0, 0), S(24, 0, 1), S(28, 0, 1)]
    assert [(r.t, r.winner, r.score) for r in V.decode_rounds(first + again)] == [
        (8.0, "ally", (3, 2)), (24.0, "enemy", (0, 1))]
    later = [S(400, 1, 0), S(404, 1, 0), S(408, 1, 1), S(412, 1, 1)]
    assert [(r.t, r.score) for r in V.decode_rounds(first + later)] == [
        (8.0, (3, 2)), (408.0, (1, 1))]


def test_decode_rounds_needs_both_scores():
    assert V.decode_rounds([S(t, 1, None) for t in range(0, 40, 4)]) == []
    assert V.decode_rounds([]) == []


def test_spike_plants_need_two_sightings():
    flags = {8: True, 12: True, 16: True, 60: True}
    samples = [S(t, spike=flags.get(t, False)) for t in range(0, 100, 4)]
    assert V.spike_plants(samples) == [(8.0, 16.0)]


# --- using the results ----------------------------------------------------------

def test_annotate_gives_each_kill_round_its_own_result():
    starts = [60.0, 200.0, 350.0, 500.0]
    result = _result([_kill(100), _kill(104), _kill(250), _kill(400)], starts)
    rounds = [
        V.RoundResult(t=150.0, winner="ally", score=(1, 0), t_low=140.0),
        V.RoundResult(t=290.0, winner="enemy", score=(1, 1), t_low=270.0),
        # Only read after the next buy phase had begun: not this round's result.
        V.RoundResult(t=530.0, winner="ally", score=(2, 1), t_low=510.0),
    ]
    V.annotate(result, rounds, [(95.0, 110.0)])
    assert [r.won for r in result.rounds] == [True, False, None]
    assert [r.after_plant for r in result.rounds] == [True, False, False]
    assert result.diagnostics["round_results"][0] == {
        "t": 150.0, "t_low": 140.0, "winner": "ally", "score": [1, 0], "exact": True}
    assert result.diagnostics["spike_plants"] == [[95.0, 110.0]]


def test_annotate_does_not_take_the_previous_rounds_result():
    result = _result([_kill(240)], [100.0, 200.0])
    # The previous round's change, first read late: its old score was last read
    # at 150, before this round's buy phase.
    V.annotate(result, [V.RoundResult(t=235.0, winner="ally", score=(1, 0), t_low=150.0)], [])
    assert result.rounds[0].won is None


def test_won_rounds_and_planted_spikes_raise_the_clip_score():
    def clip(won, planted):
        result = _result([_kill(100), _kill(104)])
        result.rounds[0].won, result.rounds[0].after_plant = won, planted
        return DG._killfeed_clips(result, types.SimpleNamespace(duration=600.0),
                                  ImportSettings(game_profile="valorant"), lead=12.0, tail=4.0)[0]

    plain, won, both, lost = clip(None, False), clip(True, False), clip(True, True), clip(False, False)
    assert plain.score < won.score < both.score
    assert lost.score == plain.score
    assert {"Round won", "Kill with the spike planted"} <= {f.label for f in both.factors}


def test_enrich_never_raises(monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("no video")

    monkeypatch.setattr(V, "read_bar", broken)
    result = K.KillfeedResult(events=[], rounds=[], samples=0, sample_interval=0.5,
                              active_s=0.0, healthy=False)
    assert V.enrich("x.mp4", INFO, result) == ([], [])

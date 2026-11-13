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


STRIP = V.strip_region(W, H)


def draw_portrait(img: np.ndarray, sign: int, k: int) -> None:
    """A busy patch in portrait slot ``k`` (sign -1: own team, +1: enemies)."""
    c = img.shape[1] // 2
    cx = int(c + sign * (V.SLOT_DX + V.SLOT_PITCH * k) * H)
    half = int(V.SLOT_HALF * H)
    y0, y1 = int((V.SLOT_Y0 - V.STRIP_Y) * H), int((V.SLOT_Y1 - V.STRIP_Y) * H)
    stripes = (np.indices((y1 - y0, 2 * half)).sum(axis=0) // 4 % 2 * 255).astype(np.uint8)
    img[y0:y1, cx - half:cx + half] = stripes[..., None]


def strip_image(ally: str | None = "3", enemy: str | None = "7", *, spike=False,
                alive=(5, 5)) -> np.ndarray:
    """The whole top bar: team bands, ``alive`` portraits per side packed towards
    the centre, and :func:`bar_image` in the middle."""
    x, y, w, h = STRIP
    img = np.zeros((h, w, 3), np.uint8)
    c = w // 2
    img[:, :c] = (70, 120, 80)                       # green-teal own-team band
    img[:, c:] = (60, 60, 150)                       # red enemy band
    for sign, n in ((-1, alive[0]), (1, alive[1])):
        for k in range(n):
            draw_portrait(img, sign, k)
    bx, by, bw, bh = V.bar_region(W, H)
    img[by - y:by - y + bh, bx - x:bx - x + bw] = bar_image(ally, enemy, spike=spike)
    return img


def A(t, allies, enemies) -> V.BarSample:
    """A sample with alive counts only."""
    return V.BarSample(t=float(t), ally=None, enemy=None, spike=False,
                       alive_ally=allies, alive_enemy=enemies)


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


def test_alive_counts_are_the_filled_slots_from_the_centre():
    for alive in ((5, 5), (1, 3), (0, 2), (3, 0)):
        assert V.alive_counts(strip_image(alive=alive), H) == alive
    gap = strip_image(alive=(2, 5))
    draw_portrait(gap, -1, 4)                        # filled behind an empty slot: not the game
    assert V.alive_counts(gap, H) == (None, 5)
    menu = np.full_like(strip_image(), 40)           # no team bands on screen
    assert V.alive_counts(menu, H) == (None, None)


def test_read_bar_reads_scores_spike_and_alive_counts_inside_the_spans():
    frames = [(0.0, strip_image(alive=(5, 4))),
              (4.0, strip_image(None, None, spike=True, alive=(1, 2))),
              (8.0, strip_image()), (500.0, strip_image())]

    def source(region, **kw):
        assert region == STRIP and kw == {"keyframes": True, "pix_fmt": "bgr24"}
        yield from frames

    reader = FakeReader("4")
    samples = V.read_bar("x.mp4", INFO, spans=[(0.0, 100.0)], reader=reader, frame_source=source)
    assert [(s.t, s.ally, s.enemy, s.spike, s.alive_ally, s.alive_enemy) for s in samples] == [
        (0.0, 4, 4, False, 5, 4), (4.0, None, None, True, 1, 2), (8.0, 4, 4, False, 5, 5)]
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


def test_find_clutches_won_lost_and_a_teammates_clutch():
    samples = (
        [A(t, 5, 5) for t in (10, 20)] + [A(t, 1, 3) for t in (40, 44, 48)] + [A(52, 1, 1)]
        + [A(t, 5, 5) for t in (110, 120)] + [A(t, 1, 2) for t in (130, 134)]
        + [A(t, 5, 5) for t in (210, 220)] + [A(t, 1, 2) for t in (230, 234)]
        + [A(t, 5, 5) for t in (310, 320)] + [A(330, 1, 1), A(340, 2, 1)]
    )
    rounds = [
        V.RoundResult(t=60.0, winner="ally", score=(1, 0), t_low=56.0),
        V.RoundResult(t=150.0, winner="enemy", score=(1, 1), t_low=146.0),
        V.RoundResult(t=250.0, winner="ally", score=(2, 1), t_low=246.0),
        V.RoundResult(t=360.0, winner="ally", score=(3, 1), t_low=356.0),
    ]
    # The streamer died at 218, just before the round's last full-team sample:
    # the one left at 230 is a teammate. At 330 only one sample shows one left.
    clutches = V.find_clutches(samples, rounds, deaths=[218.0])
    assert [(c.t, c.enemies, c.t_end, c.won, c.t_live) for c in clutches] == [
        (40.0, 3, 60.0, True, 20.0), (130.0, 2, 150.0, False, 120.0)]


def test_find_clutches_ignores_being_left_alone_after_the_round_was_decided():
    samples = ([A(t, 5, 5) for t in (400, 404)] + [A(t, 2, 2) for t in (408, 412)]
               + [A(t, 1, 2) for t in (418, 422)])
    rounds = [V.RoundResult(t=414.0, winner="ally", score=(4, 1), t_low=410.0)]
    assert V.find_clutches(samples, rounds, deaths=[]) == []
    assert V.find_clutches([], rounds, deaths=[]) == []


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


def test_annotate_puts_a_won_clutch_on_its_kill_round():
    result = _result([_kill(45), _kill(50), _kill(250)], [0.0, 100.0, 200.0])
    clutches = [V.Clutch(t=40.0, enemies=2, t_end=60.0, won=True, t_live=20.0),
                V.Clutch(t=140.0, enemies=1, t_end=170.0, won=True, t_live=120.0),   # no kill
                V.Clutch(t=240.0, enemies=3, t_end=260.0, won=False, t_live=220.0)]
    rounds = [V.RoundResult(t=60.0, winner="ally", score=(1, 0), t_low=56.0)]
    V.annotate(result, rounds, [], clutches)
    assert result.rounds[0].clutch is clutches[0] and result.rounds[0].won is True
    assert result.rounds[1].clutch is None
    assert [c["round"] for c in result.diagnostics["clutches"]] == [1, None, None]


def test_a_clutch_leads_its_kill_clip_and_gets_its_own_clip_without_kills():
    result = _result([_kill(45)], [0.0, 100.0])
    clutch = V.Clutch(t=30.0, enemies=3, t_end=60.0, won=True, t_live=10.0)
    alone = V.Clutch(t=140.0, enemies=1, t_end=175.0, won=True, t_live=120.0)
    V.annotate(result, [V.RoundResult(t=60.0, winner="ally", score=(1, 0), t_low=56.0)], [],
               [clutch, alone])
    settings = ImportSettings(game_profile="valorant")
    clips = DG._killfeed_clips(result, types.SimpleNamespace(duration=600.0), settings,
                               lead=12.0, tail=4.0)
    assert [c.title for c in clips] == ["Clutch 1v3 — 0:45", "Clutch 1v1 — 2:20"]
    kill_clip, alone_clip = clips
    assert kill_clip.score == DG._CLUTCH_SCORES[3] and alone_clip.score == DG._CLUTCH_SCORES[1]
    assert kill_clip.start <= 27.0 and kill_clip.end >= 60.0     # left alone .. round over
    assert alone_clip.start <= 137.0 and alone_clip.end >= 175.0
    for c in clips:
        assert settings.min_len <= c.end - c.start <= settings.max_len


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

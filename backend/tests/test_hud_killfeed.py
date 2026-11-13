"""Tests for the Valorant kill tracker (``app.providers.hud.killfeed``).

The tracker reads pixels, so the tests draw the three things it looks at the
way the real HUD shows them at 1440p: kill-feed rows (teal = ally, red = enemy,
a yellow tab left of the streamer's own kills, a yellow outline on their
deaths), the kill banner (white skull in a ring) and the buy-phase plate. A
scripted timeline turns those into a fake frame source, so ``track_killfeed``
runs end to end with no ffmpeg.

    cd backend && python -m pytest tests/test_hud_killfeed.py -q
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

W, H = 2560, 1440
FEED = K.feed_region(W, H)          # (x, y, w, h)
BANNER = K.banner_region(W, H)
PLATE = K.plate_region(W, H)
ROW = int(K.ROW_H * H)              # 45 px
EDGE = FEED[2] - int(round(K.EDGE_MARGIN * H))


def _bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    px = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(px[0]), int(px[1]), int(px[2])


TEAL, RED, YELLOW, WHITE = _bgr(85, 150, 180), _bgr(0, 200, 200), _bgr(27, 120, 220), (240, 240, 240)


BOUNDARY = 480                      # where the victim's half of a test row starts


def draw_reticle(img, y0: int) -> None:
    """The headshot icon as the game draws it at 1440p: four bars around an
    empty centre and four short arcs on the diagonals, white, its centre 27 px
    left of the victim's half."""
    cx, cy = BOUNDARY - 27, y0 + ROW // 2
    for (x0, y0_, x1, y1_) in ((0, -16, 0, -5), (0, 5, 0, 16), (-15, 0, -6, 0), (6, 0, 15, 0)):
        cv2.line(img, (cx + x0, cy + y0_), (cx + x1, cy + y1_), WHITE, 3)
    for sx in (-1, 1):
        for sy in (-1, 1):
            mx, my = cx + sx * 9, cy + sy * 9
            cv2.line(img, (mx - sy * 3, my + sx * 3), (mx + sy * 3, my - sx * 3), WHITE, 3)


def draw_wallbang(img, y0: int) -> None:
    """The wall-penetration icon in the same spot: an arrow into a bar."""
    cx, cy = BOUNDARY - 27, y0 + ROW // 2
    for dy in (-6, 0, 6):
        cv2.line(img, (cx - 14, cy + dy), (cx + 4, cy + dy), WHITE, 2)
    cv2.line(img, (cx + 4, cy - 9), (cx + 10, cy), WHITE, 3)
    cv2.line(img, (cx + 4, cy + 9), (cx + 10, cy), WHITE, 3)
    cv2.line(img, (cx + 15, cy - 15), (cx + 15, cy + 15), WHITE, 3)


def draw_gun(img, y0: int) -> None:
    """A rifle whose barrel reaches almost to the victim's half (no icon)."""
    img[y0 + 13:y0 + 17, 300:BOUNDARY - 8] = WHITE       # barrel
    img[y0 + 13:y0 + 33, 310:330] = WHITE                # grip


def feed_image(rows: list[tuple[int, str]]) -> np.ndarray:
    """Feed region with ``rows`` of ``(slot, kind)``.

    ``kind`` is own / mate / death, optionally followed by what sits left of
    the victim's half: ``+hs`` (headshot icon), ``+wb`` (wall-penetration
    icon) or ``+gun`` (a long barrel and no icon), e.g. ``"own+hs"``.
    """
    img = np.full((FEED[3], FEED[2], 3), 25, np.uint8)
    for slot, spec in rows:
        kind, _, extra = spec.partition("+")
        y0 = 28 + 52 * slot
        y1 = y0 + ROW
        killer, victim = (RED, TEAL) if kind == "death" else (TEAL, RED)
        img[y0:y1, 200:BOUNDARY] = killer
        img[y0:y1, BOUNDARY:EDGE] = victim
        if kind == "own":
            img[y0:y1, 175:200] = YELLOW                 # arrow tab, ~0.55 row heights
        if kind == "death":
            img[y0 - 3:y0, BOUNDARY:EDGE] = YELLOW       # outline along the victim's end
        {"hs": draw_reticle, "wb": draw_wallbang, "gun": draw_gun}.get(extra, lambda *_: None)(img, y0)
    return img


def banner_image(on: bool = True) -> np.ndarray:
    size = BANNER[2]
    img = np.full((size, size, 3), 25, np.uint8)
    if on:
        c = size // 2
        cv2.circle(img, (c, c), int(round(K.BANNER_RING * H)), WHITE, 3)
        img[c - 48:c - 28, c - 26:c + 26] = WHITE        # cranium
        img[c + 26:c + 46, c - 16:c + 16] = WHITE        # jaw
    return img


def plate_image(on: bool = True) -> np.ndarray:
    img = np.full((PLATE[3], PLATE[2], 3), 25, np.uint8)
    if on:
        h, w = img.shape[:2]
        for x in range(int(0.08 * w), int(0.92 * w), 20):   # "letters": half the band light
            img[int(0.25 * h):int(0.63 * h), x:x + 10] = WHITE
    return img


def stacked(feed, banner, plate) -> np.ndarray:
    out = np.zeros((FEED[3] + BANNER[3] + PLATE[3], FEED[2], 3), np.uint8)
    out[:FEED[3]] = feed
    out[FEED[3]:FEED[3] + BANNER[3], :BANNER[2]] = banner
    out[FEED[3] + BANNER[3]:, :PLATE[2]] = plate
    return out


def source_from(script, seconds: float, fps: float = 2.0):
    """``script(t)`` -> (feed rows, banner on, plate on); returns a frame source."""
    def source(region, **kw):
        assert region == FEED and kw["pix_fmt"] == "bgr24" and kw["stack"] == [BANNER, PLATE]
        for i in range(int(seconds * fps)):
            t = i / fps
            rows, banner, plate = script(t)
            yield t, stacked(feed_image(rows), banner_image(banner), plate_image(plate))
    return source


INFO = types.SimpleNamespace(width=W, height=H, duration=60.0, has_video=True)


# --- single-frame readers ---------------------------------------------------

def test_supports_only_valorant():
    assert K.supports("valorant") and K.supports("Valorant")
    assert not K.supports("eafc") and not K.supports(None)


def test_regions_are_even_and_inside_the_frame():
    for w, h in [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]:
        for x, y, rw, rh in (K.feed_region(w, h), K.banner_region(w, h), K.plate_region(w, h)):
            assert x % 2 == 0 and y % 2 == 0 and rw % 2 == 0 and rh % 2 == 0
            assert 0 <= x and x + rw <= w and 0 <= y and y + rh <= h


def test_read_rows_tells_own_kill_from_teammates():
    rows = K.read_rows(feed_image([(0, "own"), (1, "mate"), (2, "death")]), float(ROW), EDGE)
    assert [(r.killer, r.victim, r.own) for r in rows] == [
        ("ally", "enemy", "kill"), ("ally", "enemy", None), ("enemy", "ally", "death")]


def test_read_rows_ignores_an_empty_feed():
    assert K.read_rows(feed_image([]), float(ROW), EDGE) == []


def test_wide_yellow_block_is_a_teammates_portrait_not_an_own_kill():
    img = feed_image([(0, "mate")])
    img[28:28 + ROW, 140:200] = YELLOW                   # 1.3 row heights wide
    assert K.read_rows(img, float(ROW), EDGE)[0].own is None


def test_read_rows_scores_the_headshot_icon():
    rows = K.read_rows(feed_image([(0, "own+hs"), (1, "own+gun"), (2, "mate+hs"),
                                   (3, "death+wb"), (4, "own")]), float(ROW), EDGE)
    scores = [r.headshot for r in rows]
    assert len(scores) == 5
    assert scores[0] >= K.HEADSHOT_MIN and scores[2] >= K.HEADSHOT_MIN       # the icon
    assert scores[1] < K.HEADSHOT_MIN                                        # a barrel is not one
    assert scores[3] < K.HEADSHOT_MIN                                        # nor is the wallbang icon
    assert scores[4] == 0.0                                                  # nothing there at all


def test_headshot_score_handles_a_row_at_the_frame_edge():
    white = np.zeros((40, 30), bool)
    assert K.headshot_score(white, 0, 40, 10, 45.0) == 0.0                   # window too small
    assert K.headshot_score(np.zeros((60, 200), bool), 5, 50, 150, 45.0) == 0.0   # blank


def test_tracker_calls_a_headshot_by_majority():
    def row(hs: float) -> K.FeedRow:
        return K.FeedRow(y0=0, y1=45, killer="ally", victim="enemy", own="kill", strength=0.5,
                         sig=np.zeros((6, 48), bool), headshot=hs)

    for scores, want in (([0.7, 0.7, 0.1, 0.7], True),     # icon covered once
                         ([0.1, 0.7, 0.1, 0.1], False),    # one stray match
                         ([0.7, 0.1], True)):              # a tie keeps the icon
        tracker = K.FeedTracker()
        for i, s in enumerate(scores):
            tracker.push(i * 0.5, [row(s)])
        (event,) = tracker.finish()
        assert event.headshot is want, scores


def test_banner_kill_takes_the_headshot_from_its_feed_row():
    feed = [K.KillEvent(t=20.0, kind="kill", confidence=0.8, t_last=24.0, headshot=True)]
    out = K.merge_banner_kills(feed, [[20.5, 22.0, 0.9], [90.0, 91.5, 0.9]], interval=0.5)
    assert [(e.sources, e.headshot) for e in out] == [(["banner", "feed"], True), (["banner"], False)]
    assert "(headshot)" in out[0].detail and "(headshot)" not in out[1].detail


def test_banner_present_needs_skull_and_ring():
    assert K.banner_present(banner_image(True), H) >= 0.6
    assert K.banner_present(banner_image(False), H) == 0.0
    flash = np.full_like(banner_image(), 250)            # a bright wall fills the gap too
    assert K.banner_present(flash, H) == 0.0


def test_plate_present_rejects_blank_and_bright_frames():
    assert K.plate_present(plate_image(True))
    assert not K.plate_present(plate_image(False))
    assert not K.plate_present(np.full_like(plate_image(), 250))


# --- event logic ------------------------------------------------------------

def _kill(t, sources=("banner", "feed")):
    return K.KillEvent(t=t, kind="kill", confidence=0.9, t_last=t + 2, sources=list(sources))


def _death(t):
    return K.KillEvent(t=t, kind="death", confidence=0.9, t_last=t + 4)


def test_round_starts_drops_single_sightings():
    times = [10.0, 10.5, 11.0, 11.5, 60.0, 120.0, 120.5, 121.0]
    assert K.round_starts(times) == [10.0, 120.0]


def test_feed_kill_without_banner_is_not_a_kill():
    feed_only = [K.KillEvent(t=5.0, kind="kill", confidence=0.8, t_last=8.0)]
    assert K.merge_banner_kills(feed_only, [], interval=0.5) == []


def test_banner_is_confirmed_by_a_nearby_feed_kill():
    feed = [K.KillEvent(t=20.0, kind="kill", confidence=0.8, t_last=24.0), _death(40.0)]
    out = K.merge_banner_kills(feed, [[20.5, 22.0, 0.9], [90.0, 91.5, 0.9]], interval=0.5)
    kills = [e for e in out if e.kind == "kill"]
    assert [e.sources for e in kills] == [["banner", "feed"], ["banner"]]
    assert kills[0].confidence > kills[1].confidence
    assert len([e for e in out if e.kind == "death"]) == 1


def test_a_banner_hidden_for_one_sample_is_still_one_kill():
    # Seen on a real recording: banner at 1697.5 s, hidden by a screen flash at
    # 1698.0 s, back at 1698.5 s, with a single own-kill row in the feed.
    feed = [K.KillEvent(t=1697.0, kind="kill", confidence=0.8, t_last=1701.0)]
    out = K.merge_banner_kills(feed, [[1697.5, 1697.5, 0.93], [1698.5, 1698.5, 0.96]], interval=0.5)
    assert [(e.t, e.t_last, e.sources) for e in out] == [(1697.25, 1698.5, ["banner", "feed"])]
    # A real second kill a second later has its own feed row and stays.
    feed = [K.KillEvent(t=10.0, kind="kill", confidence=0.8), K.KillEvent(t=11.0, kind="kill", confidence=0.8)]
    out = K.merge_banner_kills(feed, [[10.5, 10.5, 0.9], [11.5, 11.5, 0.9]], interval=0.5)
    assert [e.sources for e in out] == [["banner", "feed"], ["banner", "feed"]]
    # Banners further apart than one missed sample are separate kills.
    out = K.merge_banner_kills([], [[10.5, 10.5, 0.9], [12.0, 12.0, 0.9]], interval=0.5)
    assert len(out) == 2


def test_spectated_banner_after_own_death_is_dropped():
    events = [_death(30.0), _kill(40.0, ["banner"])]
    assert [e.kind for e in K.drop_spectated(events, starts=[0.0])] == ["death"]
    # A new round started in between: the streamer is alive again.
    assert len(K.drop_spectated(events, starts=[0.0, 35.0])) == 2
    # Confirmed by the feed's own-kill tab: kept even without a round start.
    assert len(K.drop_spectated([_death(30.0), _kill(40.0)], starts=[0.0])) == 2
    # Deathmatch (no buy phase): only a window after the death is suspect.
    assert len(K.drop_spectated([_death(30.0), _kill(100.0, ["banner"])], starts=[])) == 2


def test_group_rounds_by_buy_phase_and_by_gap():
    kills = [_kill(10.0), _kill(14.0), _kill(30.0), _kill(100.0)]
    by_plate = K.group_rounds(kills, starts=[0.0, 20.0, 90.0])
    assert [r.count for r in by_plate] == [2, 1, 1]
    assert by_plate[0].label == "double_kill" and kills[1].streak == 2
    by_gap = K.group_rounds([_kill(10.0), _kill(14.0), _kill(30.0), _kill(100.0)])
    assert [r.count for r in by_gap] == [3, 1]
    assert by_gap[0].label == "triple_kill"


def test_five_kills_in_a_round_is_an_ace():
    rounds = K.group_rounds([_kill(float(t)) for t in (10, 13, 16, 19, 22)], starts=[0.0])
    assert rounds[0].label == "ace" and rounds[0].span == 12.0


# --- end to end -------------------------------------------------------------

def _script(t):
    rows = [(3, "mate")] if t < 30 else []               # keeps the feed "in use"
    if 40 <= t < 47:
        rows.append((0, "own+gun"))
    if 44 <= t < 48:
        rows.append((1, "own+hs"))
    if 48 <= t < 52:
        rows.append((2, "death"))
    banner = 40 <= t < 42 or 44 <= t < 46 or 54 <= t < 56   # the last one: spectating
    return rows, banner, t < 4


def test_track_killfeed_end_to_end():
    res = K.track_killfeed("x.mp4", INFO, frame_source=source_from(_script, 60))
    assert res is not None and res.healthy and res.samples == 120
    assert [round(e.t) for e in res.kills] == [40, 44]
    assert all(e.sources == ["banner", "feed"] for e in res.kills)
    assert [e.headshot for e in res.kills] == [False, True]
    assert len(res.deaths) == 1 and round(res.deaths[0].t) == 48
    assert [(r.label, r.count, r.headshots) for r in res.rounds] == [("double_kill", 2, 1)]
    assert res.diagnostics["round_starts"] == [0.0]
    bands = res.diagnostics["headshot_icon_rows"]
    assert bands["yes"] >= 6 and bands["unsure"] == 0


def test_track_killfeed_on_other_footage_is_unhealthy_and_empty():
    res = K.track_killfeed("x.mp4", INFO, frame_source=source_from(lambda t: ([], False, False), 30))
    assert res is not None and not res.healthy and res.events == []


def test_track_killfeed_without_a_frame_size_gives_up():
    assert K.track_killfeed("x.mp4", types.SimpleNamespace(width=0, height=0, duration=1.0)) is None


def test_killfeed_clips_score_multi_kills_higher():
    kills = [_kill(100.0), _kill(104.0), _kill(300.0)]
    rounds = K.group_rounds(kills, starts=[90.0, 250.0])
    res = K.KillfeedResult(events=kills, rounds=rounds, samples=100, sample_interval=0.5,
                           active_s=60.0, healthy=True)
    info = types.SimpleNamespace(duration=600.0)
    settings = ImportSettings(game_profile="valorant")
    clips = DG._killfeed_clips(res, info, settings, lead=12.0, tail=4.0)
    assert [c.title.split(" — ")[0] for c in clips] == ["Double kill", "Kill"]
    assert clips[0].score > clips[1].score >= 70
    for c in clips:
        assert settings.min_len <= c.end - c.start <= settings.max_len
    assert clips[0].start <= 100.0 and clips[0].end >= 104.0


def test_killfeed_clips_reward_and_name_headshots():
    def kill(t, hs):
        k = _kill(t)
        k.headshot = hs
        return k

    kills = [kill(100.0, True), kill(300.0, True), kill(302.0, True), kill(500.0, False),
             kill(700.0, True), kill(703.0, False)]
    rounds = K.group_rounds(kills, starts=[90.0, 250.0, 450.0, 650.0])
    res = K.KillfeedResult(events=kills, rounds=rounds, samples=100, sample_interval=0.5,
                           active_s=60.0, healthy=True)
    clips = DG._killfeed_clips(res, types.SimpleNamespace(duration=900.0),
                               ImportSettings(game_profile="valorant"), lead=12.0, tail=4.0)
    by_start = sorted(clips, key=lambda c: c.start)
    assert [c.title.split(" — ")[0] for c in by_start] == [
        "Headshot", "Double kill, all headshots", "Kill", "Double kill"]
    single_hs, double_hs, single, double_one = by_start
    assert single_hs.score == single.score + 2
    assert double_hs.score == double_one.score + 2       # two headshots (+4) against one (+2)
    assert any(f.label == "2 headshots" for f in double_hs.factors)
    assert not any("headshot" in f.label for f in single.factors)


# --- parallel scan ----------------------------------------------------------

def test_plan_workers_mixes_gpu_and_cpu_decoders():
    from app.providers.hud import scan
    assert scan.plan_workers(True, cpus=12) == ["gpu", "cpu", "cpu"]
    assert scan.plan_workers(True, cpus=4) == ["gpu"]
    assert scan.plan_workers(False, cpus=12) == ["cpu", "cpu", "cpu"]
    assert scan.plan_workers(False, cpus=2) == ["cpu"]


def test_map_frames_returns_every_sample_once_in_order(monkeypatch):
    from app.providers.hud import scan
    calls = []

    def fake_iter(src, region, *, start, duration, fps, hwaccel, skip_noref, **kw):
        calls.append((start, duration, hwaccel, skip_noref))
        if hwaccel and start == 20.0:
            raise scan.HudScanError("gpu said no")
        n = int(round(duration * fps))
        for i in range(n + 1):                     # one frame too many at the chunk end
            yield start + i / fps, start + i / fps

    monkeypatch.setattr(scan, "iter_frames", fake_iter)
    out = scan.map_frames("x.mp4", (0, 0, 2, 2), lambda f: f * 10, fps=2.0, start=0.0,
                          duration=45.0, chunk_s=10.0, workers=["gpu", "cpu"])
    assert [t for t, _ in out] == [i / 2 for i in range(90)]
    assert all(v == t * 10 for t, v in out)
    assert sorted({c[0] for c in calls}) == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert (20.0, 10.0, False, True) in calls       # the failed GPU chunk was re-read in software


def test_map_frames_raises_when_nothing_can_be_read(monkeypatch):
    import pytest
    from app.providers.hud import scan

    def broken(*a, **kw):
        raise scan.HudScanError("no video")
        yield  # pragma: no cover

    monkeypatch.setattr(scan, "iter_frames", broken)
    with pytest.raises(scan.HudScanError):
        scan.map_frames("x.mp4", (0, 0, 2, 2), lambda f: f, fps=2.0, duration=30.0, workers=["cpu"])


def test_gpu_scan_keeps_frames_on_the_gpu_until_after_fps(monkeypatch):
    from app.providers.hud import scan
    monkeypatch.setattr(scan, "_hw_args", lambda src: ["-hwaccel", "cuda", "-c:v", "h264_cuvid"])
    cmd = scan.build_command("ffmpeg", "x.mp4", (0, 0, 4, 4), keyframes=False, fps=2.0, start=None,
                             duration=None, hwaccel=True, passthrough=("-fps_mode", "passthrough"))
    assert "-hwaccel_output_format" in cmd
    assert cmd[cmd.index("-vf") + 1].startswith("fps=2,hwdownload,format=nv12,crop=")
    soft = scan.build_command("ffmpeg", "x.mp4", (0, 0, 4, 4), keyframes=False, fps=2.0, start=None,
                              duration=None, hwaccel=False, passthrough=("-fps_mode", "passthrough"),
                              skip_noref=True)
    assert soft[soft.index("-skip_frame") + 1] == "noref" and "hwdownload" not in " ".join(soft)

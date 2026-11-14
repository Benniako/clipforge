"""Death screens, round-state text and map callouts in the keyword OCR, and
keyword clips beside the Valorant kill tracker.

RapidOCR reads Valorant's death recap ("GETÖTET VON …", "KAMPFBERICHT") and
"RUNDE LÄUFT" cleanly, at ~0.96 confidence. Counted as kills or clutches they
outscored the kills read off the kill banner. Map callouts such as "C Lobby"
read as the lobby menu and muted the loud moments around them. The texts
below are real reads from a German Valorant recording.

    cd backend && python -m pytest tests/test_ocr_death_screen.py -q
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

import pytest  # noqa: E402

from app.media.ffmpeg import MediaInfo  # noqa: E402
from app.models import ImportSettings  # noqa: E402
from app.providers import detect_gameplay as DG  # noqa: E402
from app.providers import detect_ocr as O  # noqa: E402
from app.providers.audio_events import AudioEvent  # noqa: E402
from app.providers.detect_cues import CueEvent  # noqa: E402
from app.providers.detect_ocr import OcrEvent  # noqa: E402
from app.providers.hud import killfeed as K  # noqa: E402

DUR = 600.0
DEATH_RECAP = ("Mid Courtyard 0:51 GETÖTET VON CLOVE 60 mindestzaune Serviert Tee "
               "VERURSACHT KAMPFBERICHT ERLTEN 0 167")
ROUND_LOST = ("0:02 mindestzaune VERLOREN TEAM ELIMINIERT GETÖTET VON JETT 06 "
              "Oneshot3308 Eisblock VERURSACHT KAMPFBERICHT")


def _labels(text: str, profile: str = "valorant") -> set[str]:
    return {label for label, _ in O.match_keywords(text, profile)}


# --------------------------------------------------------------------------- #
# The lexicon
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    DEATH_RECAP,
    ROUND_LOST,
    "GETÖTET VON JETT KOPFSCHUSS",
    "KILLED BY JETT HEADSHOT COMBAT REPORT",
])
@pytest.mark.parametrize("profile", ["valorant", "generic"])
def test_a_death_screen_is_a_death_not_a_kill(text, profile):
    labels = _labels(text, profile)
    assert "death" in labels
    assert not labels & {"kill", "ace", "clutch", "eliminated"}


@pytest.mark.parametrize("text", ["RUNDE LÄUFT", "LETZTE RUNDE", "LAST ROUND"])
@pytest.mark.parametrize("profile", ["valorant", "generic"])
def test_round_state_text_is_not_a_clutch(text, profile):
    assert O.match_keywords(text, profile) == []


def test_the_combat_report_without_its_title_is_a_death_in_valorant():
    # A crop where "TEAM" and "KAMPFBERICHT" came back garbled.
    garbled = "ELIMINIERT GET VERURSACHT KAI GEDRUCH"
    assert _labels(garbled) == {"death"}
    # Elsewhere these are everyday German words.
    assert "death" not in _labels("100 SCHADEN VERURSACHT", "generic")


def test_a_round_result_is_not_a_kill():
    # A crop of the round-end banner: "TEAM ELIMINIERT" holds "eliminiert".
    assert O.match_keywords("ERLORE TEAM ELIMINIERT", "valorant") == [
        ("eliminated", "team eliminiert")]
    # On its own the word still counts both ways, as before.
    assert _labels("ELIMINIERT") == {"eliminated", "kill"}


@pytest.mark.parametrize("text", ["KRASSER KOPFSCHUSS", "DOPPELKILL", "GEGNER ÜBRIG",
                                  "GEGNER GETÖTET", "HEADSHOT", "ENEMY DOUBLE KILL"])
def test_kill_words_still_count(text):
    assert "kill" in _labels(text)


# --------------------------------------------------------------------------- #
# Map callouts are places, not menus
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "C Lobby 1:33 100 3090 550 :13.09,00.5804228",          # mid-round, by the minimap
    "A Lobby GETÖTET VON JETT 06 Oneshot3308 Eisblock VERURSACHT KAMPFBERICHT",
    "Mid Shops 0:41", "B LOBBY", "ALobby 1:02",
])
def test_a_map_callout_is_not_a_menu(text):
    assert "menu" not in _labels(text)
    event = OcrEvent(t=48.0, label="death", text=f"kampfbericht | {text}", confidence=0.96)
    assert not DG._is_menu_ocr_event(event)
    assert DG._menu_events([event], None) == []


@pytest.mark.parametrize("text", [
    "LOBBY", "PARTY LOBBY", "MAIN MENU SETTINGS",
    "0:22 SHOP R1 SEKUNDÄRWAFFEN MASCHINENPISTOLEN STURMGEWEHRE",   # the buy menu
])
def test_real_menus_still_count(text):
    assert "menu" in _labels(text)
    assert DG._is_menu_ocr_event(OcrEvent(t=1.0, label="kill", text=text, confidence=0.9))


@pytest.mark.parametrize("text", [
    "0:11 RUNDE4 KAUFPHASE HALTE ZUM KAUFEN GETÖTET VON JETT 06 Onesho",
    "Attacker Side Spawn 0:26 KAUFPHASE HALTE ZUM KAUFEN WLZ0",
    "BUY PHASE 0:25",
])
def test_the_buy_phase_is_read_in_valorant(text):
    assert "buy_phase" in _labels(text)
    assert "buy_phase" not in _labels(text, "generic")


def test_the_buy_phase_makes_no_clip_and_mutes_the_sounds_beside_it():
    buy = OcrEvent(t=33.5, label="buy_phase", text="kaufphase | C Lobby 0:07 KAUFPHASE",
                   confidence=0.95)
    assert DG._is_menu_ocr_event(buy)
    assert DG._ocr_clips([buy], _info(), _settings()) == []
    audio = [AudioEvent(t=t, label="explosive_action", confidence=0.8, detail="CLAP")
             for t in (30.5, 34.5, 60.0)]
    assert [e.t for e in DG._suppress_audio_events_near_menu(audio, [buy])] == [60.0]


def test_a_callout_does_not_mute_the_moment_beside_it():
    callout = OcrEvent(t=48.0, label="death", text="kampfbericht | C Lobby GETÖTET VON CLOVE",
                       confidence=0.96)
    menu = OcrEvent(t=48.0, label="menu", text="lobby | PARTY LOBBY", confidence=0.96)
    audio = [AudioEvent(t=50.0, label="explosive_action", confidence=0.8, detail="CLAP")]
    assert DG._suppress_audio_events_near_menu(audio, [callout]) == audio
    assert DG._suppress_audio_events_near_menu(audio, [menu]) == []


# --------------------------------------------------------------------------- #
# Death events: recorded, never a clip, never a confirmation
# --------------------------------------------------------------------------- #
def _info(**kw) -> MediaInfo:
    base = dict(duration=DUR, width=1920, height=1080, fps=30,
                has_audio=True, has_video=True, codec=None)
    base.update(kw)
    return MediaInfo(**base)


def _settings() -> ImportSettings:
    return ImportSettings(min_len=8, max_len=20, target_clips=8, game_profile="valorant")


def test_a_death_event_makes_no_clip():
    death = OcrEvent(t=100.0, label="death", text="kampfbericht", confidence=0.96)
    banner = OcrEvent(t=300.0, label="victory", text="victory", confidence=0.96)
    clips = DG._ocr_clips([death, banner], _info(), _settings())
    assert [c.title for c in clips] == ["Victory — 5:00"]


def _clip(t: float) -> DG.GameplayClip:
    return DG.GameplayClip(start=t - 10.0, end=t + 10.0, score=70, peak_t=t,
                           title="Highlight", factors=[],
                           features={"intensity": 0.5, "climax_t": t})


def test_a_death_event_never_confirms_a_clip():
    audio = [AudioEvent(t=101.0, label="crowd_cheer", confidence=0.8, detail="crowd")]
    death, real = _clip(100.0), _clip(100.0)
    DG.apply_corroboration(
        [death], [], [OcrEvent(t=100.5, label="death", text="kampfbericht", confidence=0.96)], audio)
    DG.apply_corroboration(
        [real], [], [OcrEvent(t=100.5, label="victory", text="victory", confidence=0.96)], audio)
    assert death.score == 70 and "supporting_audio" not in death.features
    assert real.score == 73 and real.features["supporting_audio"] == 1.0


# --------------------------------------------------------------------------- #
# Beside a readable kill feed
# --------------------------------------------------------------------------- #
def test_the_kill_feed_cap_sits_below_every_kill():
    assert DG.KILLFEED_UNVERIFIED_CAP < min(DG._KILL_SCORES.values())
    assert DG._unverified_cap(None, None) is None
    assert DG._unverified_cap(object(), None) == DG.HUD_UNVERIFIED_CAP
    assert DG._unverified_cap(None, object()) == DG.KILLFEED_UNVERIFIED_CAP
    assert DG._unverified_cap(object(), object()) == DG.KILLFEED_UNVERIFIED_CAP


def _kills() -> K.KillfeedResult:
    kills = [K.KillEvent(t=t, kind="kill", confidence=0.9, t_last=t + 2.0,
                         sources=["banner", "feed"]) for t in (100.0, 300.0)]
    death = K.KillEvent(t=320.0, kind="death", confidence=0.9, t_last=324.0)
    # The round start at 400 s ends the streamer's death at 320 s.
    diagnostics = {"round_starts": [90.0, 250.0, 400.0],
                   "round_results": [{"t": 110.0, "winner": "ally", "score": [1, 0]}],
                   "spike_plants": [(150.0, 190.0)],
                   "clutches": [{"t": 400.0, "t_end": 420.0, "enemies": 2, "won": False,
                                 "round": None}]}
    return K.KillfeedResult(events=kills + [death],
                            rounds=K.group_rounds(kills, starts=[90.0, 250.0]),
                            samples=1200, sample_interval=0.5, active_s=600.0,
                            healthy=True, diagnostics=diagnostics)


# What the keyword sweep reads: the streamer's death recap right after the
# first kill, a kill word nobody can check, and a banner.
OCR = [OcrEvent(t=104.0, label="death", text="kampfbericht", confidence=0.96),
       OcrEvent(t=200.0, label="kill", text="kopfschuss", confidence=0.96),
       OcrEvent(t=470.0, label="victory", text="victory", confidence=0.96)]


BURSTS = ((100.0, 0.8), (200.0, 0.6), (470.0, 0.7))


def _envelope(bursts=BURSTS):
    """10 Hz loudness with a burst at each (time, height) (exact arithmetic)."""
    import numpy as np

    idx = np.arange(int(DUR * 10))
    times = idx * 0.1
    rms = 0.01 + ((idx * 7919) % 6007) * 1e-8
    for centre, amp in bursts:
        rms = rms + amp * np.clip(1.0 - np.abs(times - centre) / 3.0, 0.0, None)
    return times, rms.astype(np.float32)


def _detect(monkeypatch, *, mode: str = "normal", kills=None, ocr=OCR, audio=(), cues=(),
            bursts=BURSTS, peaks=None):
    """detect_gameplay with the kill tracker and every detector faked.

    ``peaks`` pins the loud moments (seconds) instead of picking them.
    """
    if peaks is not None:
        idxs = [int(round(t * 10)) for t in peaks]
        monkeypatch.setattr(DG, "_pick_peaks", lambda *a, **k: (list(idxs), 0.1, 0.01))
    kills = _kills() if kills is None else kills
    monkeypatch.setattr(DG, "_track_hud", lambda *a, **k: None)
    monkeypatch.setattr(DG, "_track_killfeed", lambda *a, **k: kills)
    monkeypatch.setattr(DG, "_find_ocr_events", lambda *a, **k: list(ocr))
    monkeypatch.setattr(DG, "_cue_events", lambda wav, st: list(cues))
    monkeypatch.setattr(DG, "_audio_events", lambda *a, **k: list(audio))
    monkeypatch.setattr(DG, "_load_rms",
                        (lambda wav, **k: (None, None)) if mode == "no_rms"
                        else (lambda wav, **k: _envelope(bursts)))
    if mode == "no_peaks":
        monkeypatch.setattr(DG, "_pick_peaks", lambda *a, **k: ([], 0.0, 0.0))
    events: list = []
    clips = DG.detect_gameplay("unused.mp4", _info(has_audio=mode != "no_audio"), _settings(),
                               wav_path="unused.wav", events_out=events, warnings_out=[])
    return clips, events


@pytest.mark.parametrize("mode", ["normal", "no_peaks", "no_rms", "no_audio"])
def test_keyword_clips_rank_below_the_kill_feed(monkeypatch, mode):
    clips, events = _detect(monkeypatch, mode=mode)

    kills = [c for c in clips if c.features.get("hud")]
    others = [c for c in clips if not c.features.get("hud")]
    assert sorted(c.title for c in kills) == ["Kill — 1:40", "Kill — 5:00"], mode
    assert min(c.score for c in kills) >= DG._KILL_SCORES[1]
    assert sorted(c.title for c in others) == ["Kill — 3:20", "Victory — 7:50"], mode
    assert max(c.score for c in others) == DG.KILLFEED_UNVERIFIED_CAP, mode

    # The trackers' events are recorded with plain values (lists used to make
    # the event model reject them, failing the whole detection).
    tracked = [e for e in events if e.data.get("detector") == "killfeed"]
    assert [(e.label, e.data["sources"]) for e in tracked] == [
        ("kill", "banner+feed"), ("kill", "banner+feed"), ("death", "feed")]
    rounds = [e for e in events if e.data.get("detector") == "valorant_rounds"]
    assert [(e.label, e.data.get("score")) for e in rounds] == [
        ("round_win", "1-0"), ("spike_plant", None), ("clutch_lost", None)]
    # The death recap is kept as an event.
    assert [(e.t, e.label) for e in events if e.source == "ocr" and not e.data] == [
        (104.0, "death"), (200.0, "kill"), (470.0, "victory")]


# --------------------------------------------------------------------------- #
# While the streamer is dead
# --------------------------------------------------------------------------- #
def _dead(deaths, *, starts=(), results=()) -> K.KillfeedResult:
    events = [K.KillEvent(t=t, kind="death", confidence=0.9, t_last=t + 4.0) for t in deaths]
    diagnostics = {"round_starts": list(starts),
                   "round_results": [{"t": t, "winner": "enemy", "score": [0, 1]}
                                     for t in results]}
    return K.KillfeedResult(events=events, rounds=[], samples=1200, sample_interval=0.5,
                            active_s=600.0, healthy=True, diagnostics=diagnostics)


def test_spectating_runs_from_the_death_to_the_round_end():
    # The Valorant cut: died at 66.2 s, the top bar showed the round lost at
    # 110.9 s, and the buy-phase plate lit up early at 97.5 s (the scoreboard).
    (span,) = DG._spectating_spans(
        _dead([66.2], starts=[3.5, 97.5], results=[110.9]), 120.0)
    assert span == pytest.approx((69.2, 115.9))
    # A round end the top bar missed: the next buy phase is over 30 s after
    # the plate, so the next round's play is never muted.
    assert DG._spectating_spans(_dead([300.0], starts=[0.0, 340.0]), 600.0) == [(303.0, 370.0)]
    # No later round marker at all: until the end of the footage at most.
    assert DG._spectating_spans(_dead([500.0], starts=[0.0]), 600.0) == [(503.0, 600.0)]
    # Deathmatch (no round starts or results) respawns at once: nothing is muted.
    assert DG._spectating_spans(_dead([100.0]), 600.0) == []
    assert DG._spectating_spans(None, 600.0) == []


def test_nothing_from_a_teammate_while_the_streamer_is_dead(monkeypatch):
    # _kills(): died at 320 s, the next round starts at 400 s (muted to 430 s).
    ocr = OCR + [OcrEvent(t=350.0, label="defeat", text="verloren", confidence=0.96),
                 OcrEvent(t=352.0, label="death", text="kampfbericht", confidence=0.96)]
    audio = [AudioEvent(t=360.0, label="explosive_action", confidence=0.8, detail="CLAP")]
    cues = [CueEvent(t=365.0, label="kill", similarity=0.95)]
    for mode in ("normal", "no_peaks", "no_rms"):
        clips, events = _detect(monkeypatch, mode=mode, ocr=ocr, audio=audio, cues=cues,
                                bursts=BURSTS + ((350.0, 0.9),))
        assert not any(323.0 <= c.peak_t <= 430.0 for c in clips), mode
        assert sorted(c.title for c in clips) == [
            "Kill — 1:40", "Kill — 3:20", "Kill — 5:00", "Victory — 7:50"], mode
        seen = [(e.t, e.source, e.label) for e in events if 323.0 <= e.t <= 430.0]
        # Only the death recap is left from those 107 s (plus the tracker's own
        # clutch event at 400 s, read off the top bar).
        assert seen == [(352.0, "ocr", "death"), (400.0, "ocr", "clutch_lost")], mode


def test_loud_moments_while_dead_are_muted_unless_nothing_else_is_left(monkeypatch):
    # No kills: died at 320 s, next round at 400 s, the loudest burst at 350 s.
    clips, _ = _detect(monkeypatch, kills=_dead([320.0], starts=[0.0, 400.0]), ocr=[],
                       bursts=BURSTS + ((350.0, 0.95),))
    assert clips and not any(323.0 <= c.peak_t <= 400.0 for c in clips)
    # Every loud moment while dead: keep them rather than return no clips at all.
    clips, _ = _detect(monkeypatch, kills=_dead([90.0], starts=[0.0]), ocr=[],
                       bursts=((100.0, 0.8), (200.0, 0.6)), peaks=[100.0, 200.0])
    assert clips and all(93.0 <= c.peak_t <= 240.0 for c in clips)

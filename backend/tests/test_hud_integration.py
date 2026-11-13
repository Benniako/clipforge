"""The scoreboard tracker wired into gameplay detection.

No ffmpeg, no OCR and no real video: the tracker's result is a small fake
``HudResult`` and the audio / keyword detectors are replaced by fixed lists, so
these tests pin down what ``detect_gameplay`` does *with* a scoreboard result
(goals lead, other detectors are held to match time) and that it does exactly
what it always did without one.

    cd backend && python -m pytest tests/test_hud_integration.py -q
"""
from __future__ import annotations

import os
import re
import tempfile
import types

# Keep the learning DB and cue store away from the real data dir; this has to
# happen before anything imports app.config.
os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

import pytest

from app.media.ffmpeg import MediaInfo
from app.models import (Clip, DetectedEvent, GameProfileConfig, ImportSettings,
                        Project)
from app.providers import detect_gameplay as DG
from app.providers.audio_events import AudioEvent
from app.providers.detect_cues import CueEvent
from app.providers.detect_ocr import OcrEvent
from app.providers.hud import HudEvent, HudResult, MatchInfo

DUR = 600.0
REAL_TRACK_HUD = "real"     # _run(hud=REAL_TRACK_HUD): leave _track_hud unpatched

# What the detectors "find" in the synthetic footage.
CUES = [CueEvent(t=101.0, label="goal", similarity=0.93),
        CueEvent(t=330.0, label="kill", similarity=0.95)]
AUDIO = [AudioEvent(t=252.0, label="crowd_cheer", confidence=0.8, detail="crowd"),
         AudioEvent(t=45.0, label="excited_shouting", confidence=0.7, detail="shout")]
OCR = [OcrEvent(t=104.0, label="goal", text="goal", confidence=0.9),
       OcrEvent(t=405.0, label="scoreboard", text="2 - 1", confidence=0.8),
       OcrEvent(t=48.0, label="menu", text="main menu settings", confidence=0.9),
       OcrEvent(t=470.0, label="victory", text="victory", confidence=0.85)]

# detect_gameplay's output for the lists above before the scoreboard tracker
# existed, as (start, end, score, title) per return path. Recorded from the
# unmodified detector; "without a scoreboard nothing changes" is checked
# against these, not just against another run of the new code.
LEGACY_CLIPS = {
    "normal": [(88.58, 102.58, 98, "Goal — 1:44"),
               (241.08, 255.08, 81, "Crowd Cheer - 4:12"),
               (319.08, 333.08, 71, "Kill — 5:30"),
               (389.58, 403.58, 83, "Scoreboard — 6:45"),
               (459.08, 473.08, 97, "Victory — 7:50")],
    "no_audio": [(89.0, 103.0, 98, "Goal — 1:44"),
                 (459.5, 473.5, 97, "Victory — 7:50"),
                 (390.0, 404.0, 83, "Scoreboard — 6:45")],
    "no_rms": [(89.0, 103.0, 98, "Goal — 1:44"),
               (459.5, 473.5, 97, "Victory — 7:50"),
               (390.0, 404.0, 83, "Scoreboard — 6:45"),
               (241.5, 255.5, 81, "Crowd Cheer - 4:12")],
    "no_peaks": [(88.58, 102.58, 98, "Goal — 1:44"),
                 (459.08, 473.08, 97, "Victory — 7:50"),
                 (389.58, 403.58, 83, "Scoreboard — 6:45"),
                 (241.08, 255.08, 81, "Crowd Cheer - 4:12"),
                 (319.08, 333.08, 71, "Kill — 5:30")],
}
LEGACY_EVENTS = {
    "normal": [(48.0, "ocr", "menu"), (101.0, "cue", "goal"), (104.0, "ocr", "goal"),
               (252.0, "audio", "crowd_cheer"), (330.0, "cue", "kill"),
               (405.0, "ocr", "scoreboard"), (470.0, "ocr", "victory")],
    "no_audio": [(48.0, "ocr", "menu"), (104.0, "ocr", "goal"),
                 (405.0, "ocr", "scoreboard"), (470.0, "ocr", "victory")],
}
LEGACY_EVENTS["no_rms"] = LEGACY_EVENTS["no_peaks"] = LEGACY_EVENTS["normal"]

# Every way detect_gameplay can return.
PATHS = ["normal", "no_peaks", "no_rms", "no_audio", "no_extract"]


# --------------------------------------------------------------------------- #
# Fixtures built by hand
# --------------------------------------------------------------------------- #
def _info(**kw) -> MediaInfo:
    base = dict(duration=DUR, width=1920, height=1080, fps=30,
                has_audio=True, has_video=True, codec=None)
    base.update(kw)
    return MediaInfo(**base)


def _settings(**kw) -> ImportSettings:
    base = dict(min_len=8, max_len=20, target_clips=8, game_profile="eafc")
    base.update(kw)
    return ImportSettings(**base)


def _legacy_on(**kw) -> ImportSettings:
    return _settings(game_config=GameProfileConfig(legacy_ocr_with_hud=True), **kw)


def _goal(t: float, before: tuple[int, int], after: tuple[int, int], *,
          team: str = "a", **kw) -> HudEvent:
    return HudEvent(t=t, team=team, score_before=before, score_after=after,
                    t_change=t + 1.8, window=(t - 2.0, t + 4.0), **kw)


def _two_goals() -> list[HudEvent]:
    return [_goal(100.0, (0, 0), (1, 0), minute=13, clock="12:29", importance=9.0,
                  streamer=True, reasons=["Go-ahead goal", "Streamer's team scores"]),
            _goal(400.0, (1, 0), (1, 1), team="b", importance=2.0, streamer=False,
                  reasons=["Equaliser", "Opponent scores"])]


def _hud(events=None, *, in_play=((60.0, 440.0),), coverage: float | None = None,
         healthy: bool = True) -> HudResult:
    events = _two_goals() if events is None else list(events)
    spans = [(float(a), float(b)) for a, b in in_play]
    if coverage is None:
        coverage = sum(b - a for a, b in spans) / DUR
    final = events[-1].score_after if events else (0, 0)
    return HudResult(layout="eafc_stacked", events=events,
                     matches=[MatchInfo(t_start=spans[0][0], t_end=spans[-1][1],
                                        start_score=(0, 0), final_score=final,
                                        goals=len(events))],
                     in_play=spans, coverage=coverage, healthy=healthy,
                     samples=300, sample_interval=2.0)


def _envelope():
    """A 10 Hz loudness envelope with four bursts (100 s, 250 s, 400 s, 520 s).

    Built from exact arithmetic only (no sin / exp / random) and with no two
    equal values, so peak picking gives the same answer on every platform.
    """
    import numpy as np

    idx = np.arange(6000)
    times = idx * 0.1
    rms = 0.01 + ((idx * 7919) % 6007) * 1e-8
    for centre, amp in ((100.0, 0.8), (250.0, 0.6), (400.0, 0.7), (520.0, 0.5)):
        rms = rms + amp * np.clip(1.0 - np.abs(times - centre) / 3.0, 0.0, None)
    return times, rms.astype(np.float32)


def _run(monkeypatch, *, hud=None, mode: str = "normal", settings=None,
         cues=CUES, audio=AUDIO, ocr=OCR):
    """Run detect_gameplay on the synthetic footage down one return path."""
    seen = types.SimpleNamespace(ocr_calls=[], hud_calls=0, corroboration=[])

    def fake_ocr(src_path, info, st, *, focus_times=None, warnings_out=None):
        seen.ocr_calls.append(None if focus_times is None else list(focus_times))
        return list(ocr)

    def fake_track(*a, **k):
        seen.hud_calls += 1
        return hud

    def boom(*a, **k):
        raise RuntimeError("no audio stream")

    real_corroboration = DG.apply_corroboration

    def spy_corroboration(clips, cue_events, ocr_events, audio_events=None):
        seen.corroboration.append(list(ocr_events))
        return real_corroboration(clips, cue_events, ocr_events, audio_events)

    monkeypatch.setattr(DG, "_find_ocr_events", fake_ocr)
    monkeypatch.setattr(DG, "_cue_events", lambda wav, st: list(cues))
    monkeypatch.setattr(DG, "_audio_events", lambda *a, **k: list(audio))
    monkeypatch.setattr(DG, "apply_corroboration", spy_corroboration)
    monkeypatch.setattr(DG, "_load_rms",
                        (lambda wav, **k: (None, None)) if mode == "no_rms"
                        else (lambda wav, **k: _envelope()))
    if mode == "no_peaks":
        monkeypatch.setattr(DG, "_pick_peaks", lambda *a, **k: ([], 0.0, 0.0))
    if mode == "no_extract":
        monkeypatch.setattr(DG.ffmpeg, "extract_audio_wav", boom)
    if hud != REAL_TRACK_HUD:
        monkeypatch.setattr(DG, "_track_hud", fake_track)

    events: list = []
    warnings: list = []
    clips = DG.detect_gameplay(
        "unused.mp4", _info(has_audio=(mode != "no_audio")), settings or _settings(),
        wav_path=None if mode == "no_extract" else "unused.wav",
        events_out=events, warnings_out=warnings)
    seen.clips, seen.events, seen.warnings = clips, events, warnings
    return seen


def _sig(clips) -> list[tuple]:
    return [(c.start, c.end, c.score, c.title) for c in clips]


def _is_goal(c) -> bool:
    return float(c.features.get("hud", 0.0)) > 0.0


def _hud_events(events) -> list[DetectedEvent]:
    return [e for e in events if e.data.get("detector") == "hud"]


# --------------------------------------------------------------------------- #
# _hud_goal_clips
# --------------------------------------------------------------------------- #
def test_goal_clip_window_is_centred_on_the_goal_moment():
    hud = _hud([_goal(100.0, (0, 0), (1, 0))])
    (c,) = DG._hud_goal_clips(hud, _info(), _settings(), lead=10.0, tail=4.0)
    # No offset: unlike a keyword banner, ev.t already is the goal.
    assert (c.start, c.end) == (90.0, 104.0)
    assert c.peak_t == 100.0
    assert c.cue_ts == []


def test_goal_clip_window_respects_video_bounds_and_length_limits():
    st = _settings()                                   # min_len 8, max_len 20

    def window(t, lead, tail):
        (c,) = DG._hud_goal_clips(_hud([_goal(t, (0, 0), (1, 0))]), _info(), st,
                                  lead=lead, tail=tail)
        return c.start, c.end

    assert window(3.0, 10.0, 4.0) == (0.0, 7.0)        # clamped at the start
    assert window(598.0, 10.0, 4.0) == (588.0, 600.0)  # clamped at the end
    assert window(100.0, 2.0, 2.0) == (94.0, 102.0)    # stretched back to min_len
    assert window(100.0, 30.0, 5.0) == (85.0, 105.0)   # trimmed at the front to max_len


def test_goal_clip_score_arithmetic():
    def score(**kw):
        (c,) = DG._hud_goal_clips(_hud([_goal(100.0, (0, 0), (1, 0), **kw)]),
                                  _info(), _settings(), lead=10.0, tail=4.0)
        assert type(c.score) is int
        return c.score

    assert score() == 84                               # importance 0, confidence 0.95
    assert score(importance=9.0) == 93
    assert score(importance=-3.0) == 81
    assert score(importance=2.4) == 86                 # rounded, not truncated
    assert score(importance=9.0, confidence=0.6) == 85     # low confidence: -8
    assert score(importance=9.0, confidence=0.7) == 93     # 0.7 itself is not low
    assert score(importance=9.0, confidence=0.69) == 85
    assert score(importance=40.0) == 99                # upper clamp
    # Lower clamp: the weakest verified goal still outranks any unverified clip.
    assert score(importance=-40.0, confidence=0.5) == DG.HUD_GOAL_FLOOR == 80
    assert score(importance=-20.0) == 80
    assert DG.HUD_GOAL_FLOOR > DG.HUD_UNVERIFIED_CAP


def test_goal_clip_title_shows_the_new_score_and_the_time():
    hud = _hud([_goal(100.9, (1, 1), (2, 1)), _goal(3725.0, (2, 1), (2, 2), team="b")])
    a, b = DG._hud_goal_clips(hud, _info(duration=4000.0), _settings(), lead=10.0, tail=4.0)
    assert a.title == "Goal 2-1 — 1:40"
    assert b.title == "Goal 2-2 — 62:05"
    # Not a "Highlight" title, so the fallback renamer leaves it alone.
    DG.apply_title_fallbacks([a, b], profile="eafc")
    assert a.title == "Goal 2-1 — 1:40"


def test_goal_clip_features_are_all_floats():
    ev = _goal(100.0, (0, 0), (1, 0), importance=9, confidence=0.85)   # int importance
    (c,) = DG._hud_goal_clips(_hud([ev]), _info(), _settings(), lead=10.0, tail=4.0)
    assert c.features == {"ocr": 0.85, "hud": 1.0, "hud_importance": 9.0,
                          "intensity": 0.0, "sustain": 0.0, "transient": 0.0,
                          "spikes": 0.0, "cue": 0.0, "climax_t": 100.0}
    assert all(type(v) is float for v in c.features.values())
    # Clip.features is dict[str, float]: the clip model must take them as they are.
    assert Clip(start=c.start, end=c.end, features=c.features).features == c.features


def test_goal_clip_factors_with_minute_clock_and_reasons():
    ev = _goal(100.0, (1, 0), (1, 1), team="b", minute=88, clock="87:12", importance=12.0,
               confidence=0.9,
               reasons=["Equaliser", "Late goal (88')", "Comeback from 2 down", "Opponent scores"])
    (c,) = DG._hud_goal_clips(_hud([ev]), _info(), _settings(), lead=10.0, tail=4.0)
    assert len(c.factors) == 2
    board, why = c.factors
    assert board.label == "Scoreboard 1-0 → 1-1 (88')"
    assert board.weight == 18.0
    assert board.detail == "Read off the on-screen scoreboard at clock 87:12"
    assert why.label == "Equaliser; Late goal (88'); Comeback from 2 down"   # first three
    assert why.weight == 12.0
    assert why.detail == "What this goal did to the match"


def test_goal_clip_factors_without_minute_clock_or_reasons():
    ev = _goal(100.0, (0, 0), (1, 0), confidence=0.6)
    (c,) = DG._hud_goal_clips(_hud([ev]), _info(), _settings(), lead=10.0, tail=4.0)
    assert len(c.factors) == 1
    assert c.factors[0].label == "Scoreboard 0-0 → 1-0"
    assert c.factors[0].weight == 12.0
    assert c.factors[0].detail == "Read off the on-screen scoreboard"


def test_no_goals_means_no_goal_clips():
    assert DG._hud_goal_clips(_hud([]), _info(), _settings(), lead=10.0, tail=4.0) == []


# --------------------------------------------------------------------------- #
# detect_gameplay with a healthy scoreboard result
# --------------------------------------------------------------------------- #
def test_goals_are_returned_and_rank_first(monkeypatch):
    out = _run(monkeypatch, hud=_hud())
    goals = [c for c in out.clips if _is_goal(c)]
    others = [c for c in out.clips if not _is_goal(c)]
    assert sorted(c.title for c in goals) == ["Goal 1-0 — 1:40", "Goal 1-1 — 6:40"]
    assert others, "the scenario should also produce unverified clips"
    assert min(c.score for c in goals) > max(c.score for c in others)
    ranked = sorted(out.clips, key=lambda c: c.score, reverse=True)
    assert all(_is_goal(c) for c in ranked[:len(goals)])
    assert out.hud_calls == 1                          # tracked once, not per path


def test_legacy_keyword_ocr_is_skipped_with_a_scoreboard(monkeypatch):
    for mode in ("normal", "no_audio"):                # deferred and immediate sweep
        out = _run(monkeypatch, hud=_hud(), mode=mode)
        assert out.ocr_calls == [], mode
        assert [e for e in out.events if e.source == "ocr"] == _hud_events(out.events)
        assert not any(c.title.startswith(("Scoreboard", "Victory")) for c in out.clips)


def test_legacy_keyword_ocr_runs_when_asked_and_loses_its_goal_events(monkeypatch):
    ocr = OCR + [OcrEvent(t=300.0, label="victory", text="victory", confidence=0.85)]
    for mode in ("normal", "no_audio"):
        out = _run(monkeypatch, hud=_hud(), mode=mode, settings=_legacy_on(), ocr=ocr)
        assert len(out.ocr_calls) == 1, mode
        keyword = [(e.t, e.label) for e in out.events
                   if e.source == "ocr" and e.data.get("detector") != "hud"]
        # goal@104 and scoreboard@405 are superseded by the tracker; menu@48 and
        # victory@470 were read while no scoreboard was on screen (coverage 63%).
        assert keyword == [(300.0, "victory")], mode
        assert not any(c.title.startswith("Scoreboard") for c in out.clips)
        assert sorted(c.title for c in out.clips if _is_goal(c)) == [
            "Goal 1-0 — 1:40", "Goal 1-1 — 6:40"]
        assert any(c.title.startswith("Victory") for c in out.clips)


def test_legacy_keyword_ocr_keeps_out_of_play_events_below_the_coverage_gate(monkeypatch):
    out = _run(monkeypatch, hud=_hud(coverage=0.2), settings=_legacy_on())
    keyword = [(e.t, e.label) for e in out.events
               if e.source == "ocr" and e.data.get("detector") != "hud"]
    assert keyword == [(48.0, "menu"), (470.0, "victory")]   # goal / scoreboard still dropped


def test_coverage_gate_drops_cue_and_audio_events_outside_match_time(monkeypatch):
    cues = CUES + [CueEvent(t=560.0, label="kill", similarity=0.95)]

    gated = _run(monkeypatch, hud=_hud(), cues=cues)             # coverage 0.63
    assert [e.t for e in gated.events if e.source == "cue"] == [101.0, 330.0]
    assert [e.t for e in gated.events if e.source == "audio"] == [252.0]
    assert not any(c.peak_t in (45.0, 560.0) for c in gated.clips)

    loose = _run(monkeypatch, hud=_hud(coverage=0.2), cues=cues)
    assert [e.t for e in loose.events if e.source == "cue"] == [101.0, 330.0, 560.0]
    assert [e.t for e in loose.events if e.source == "audio"] == [45.0, 252.0]
    assert {45.0, 560.0} <= {c.peak_t for c in loose.clips}


def test_coverage_gate_threshold_is_inclusive(monkeypatch):
    assert DG.HUD_GATE_MIN_COVERAGE == 0.30
    at = _run(monkeypatch, hud=_hud(coverage=0.30))
    below = _run(monkeypatch, hud=_hud(coverage=0.29))
    assert [e.t for e in at.events if e.source == "audio"] == [252.0]
    assert [e.t for e in below.events if e.source == "audio"] == [45.0, 252.0]


def test_coverage_gate_drops_loudness_peaks_outside_match_time(monkeypatch):
    # A goalless match and no other detector: only loudness clips remain, and
    # the best-scoring burst (400 s) lies outside the 60-300 s match.
    span = ((60.0, 300.0),)
    loose = _run(monkeypatch, hud=_hud([], in_play=span, coverage=0.2), cues=[], audio=[])
    assert loose.clips
    assert all(c.title.startswith("Highlight") for c in loose.clips)
    assert max(loose.clips, key=lambda c: c.score).peak_t == pytest.approx(400.0)

    gated = _run(monkeypatch, hud=_hud([], in_play=span), cues=[], audio=[])   # coverage 0.4
    assert gated.clips
    assert all(c.title.startswith("Highlight") for c in gated.clips)
    assert all(58.0 <= c.peak_t <= 302.0 for c in gated.clips)


def test_gated_out_moments_are_not_ocr_focus_times(monkeypatch):
    cues = CUES + [CueEvent(t=560.0, label="kill", similarity=0.95)]
    hud = _hud()

    gated = _run(monkeypatch, hud=hud, cues=cues, settings=_legacy_on())
    (focus,) = gated.ocr_calls
    assert focus and all(hud.is_in_play(t) for t in focus)
    assert 101.0 in focus and 252.0 in focus

    loose = _run(monkeypatch, hud=_hud(coverage=0.2), cues=cues, settings=_legacy_on())
    (focus,) = loose.ocr_calls
    assert {45.0, 560.0} <= set(focus)
    assert any(abs(t - 520.0) < 1.0 for t in focus)


def test_menu_text_read_during_match_time_is_not_menu_evidence(monkeypatch):
    # "CHAMPIONS" on a stadium board at 250 s, a real menu at 48 s.
    ocr = [OcrEvent(t=250.0, label="menu", text="champions", confidence=0.9),
           OcrEvent(t=48.0, label="menu", text="main menu settings", confidence=0.9)]
    hud = _hud(coverage=0.2)                           # below the gate: nothing is pre-filtered
    assert [e.t for e in DG._menu_events(ocr, hud)] == [48.0]
    assert [e.t for e in DG._menu_events(ocr, None)] == [250.0, 48.0]

    out = _run(monkeypatch, hud=hud, ocr=ocr, settings=_legacy_on())
    # The cheer at 252 s survives the board; the shout at 45 s is still a menu sound.
    assert [e.t for e in out.events if e.source == "audio"] == [252.0]

    legacy = _run(monkeypatch, hud=None, ocr=ocr)
    assert [e.t for e in legacy.events if e.source == "audio"] == []


def test_menu_text_does_not_suppress_loudness_peaks_during_match_time(monkeypatch):
    ocr = [OcrEvent(t=250.0, label="menu", text="champions", confidence=0.9)]

    # Match time 200-300 s holds one burst, at 250 s, right under the "menu" text.
    span = ((200.0, 300.0),)
    tracked = _run(monkeypatch, hud=_hud([], in_play=span, coverage=0.5), ocr=ocr,
                   cues=[], audio=[], settings=_legacy_on())
    assert [c.peak_t for c in tracked.clips] == [pytest.approx(250.0)]

    # Without a scoreboard the same text still silences the burst.
    legacy = _run(monkeypatch, hud=None, ocr=ocr, cues=[], audio=[])
    assert legacy.clips
    assert not any(abs(c.peak_t - 250.0) < 10.0 for c in legacy.clips)


def test_unverified_priority_clips_are_capped_below_the_goals(monkeypatch):
    assert DG.HUD_UNVERIFIED_CAP == 79
    # A kill sound and a kill-feed line at the same instant corroborate each
    # other; even that bonus must not lift the clip past the cap.
    ocr = [OcrEvent(t=300.0, label="victory", text="victory", confidence=0.85),
           OcrEvent(t=331.0, label="kill", text="eliminated", confidence=0.9)]
    for mode in ("normal", "no_peaks", "no_rms", "no_audio"):
        out = _run(monkeypatch, hud=_hud(), mode=mode, settings=_legacy_on(), ocr=ocr)
        others = [c for c in out.clips if not _is_goal(c) and not c.title.startswith("Segment")]
        assert others, mode
        assert max(c.score for c in others) == 79, (mode, _sig(others))
        assert min(c.score for c in out.clips if _is_goal(c)) > 79, mode

    # The same input without a scoreboard keeps its scores.
    legacy = _run(monkeypatch, hud=None, ocr=ocr)
    assert max(c.score for c in legacy.clips) > 79


def test_corroboration_sees_scoreboard_goals_as_on_screen_evidence(monkeypatch):
    hud = _hud()
    out = _run(monkeypatch, hud=hud)
    (ocr_seen,) = out.corroboration
    assert ocr_seen == [OcrEvent(t=100.0, label="goal", text=hud.events[0].detail, confidence=0.95),
                        OcrEvent(t=400.0, label="goal", text=hud.events[1].detail, confidence=0.95)]
    # The goal sound at 101 s and the scoreboard goal at 100 s confirm each other.
    first = next(c for c in out.clips if c.title == "Goal 1-0 — 1:40")
    assert first.features.get("corroborated") == 1.0
    assert first.factors[0].label == "Multimodal confirm"
    assert first.score == 99                           # 93 + 6
    assert all(type(v) is float for v in first.features.values())


def test_corroboration_gets_keyword_events_plus_goals_when_legacy_ocr_runs(monkeypatch):
    ocr = [OcrEvent(t=300.0, label="victory", text="victory", confidence=0.85)]
    out = _run(monkeypatch, hud=_hud(), settings=_legacy_on(), ocr=ocr)
    (ocr_seen,) = out.corroboration
    assert [(e.t, e.label) for e in ocr_seen] == [(300.0, "victory"), (100.0, "goal"), (400.0, "goal")]


@pytest.mark.parametrize("mode", PATHS)
def test_every_return_path_includes_goal_clips_and_records_goal_events(monkeypatch, mode):
    out = _run(monkeypatch, hud=_hud(), mode=mode)
    assert sorted(c.title for c in out.clips if _is_goal(c)) == [
        "Goal 1-0 — 1:40", "Goal 1-1 — 6:40"]
    recorded = _hud_events(out.events)
    assert [(e.t, e.source, e.label) for e in recorded] == [
        (100.0, "ocr", "goal"), (400.0, "ocr", "goal")]
    assert [e.data["score_after"] for e in recorded] == ["1-0", "1-1"]
    assert out.events == sorted(out.events, key=lambda e: e.t)
    assert out.hud_calls == 1
    assert out.ocr_calls == []


@pytest.mark.parametrize("mode", PATHS)
def test_goalless_scoreboard_result_still_returns_clips(monkeypatch, mode):
    out = _run(monkeypatch, hud=_hud([]), mode=mode)
    assert out.clips
    assert not any(_is_goal(c) for c in out.clips)
    assert _hud_events(out.events) == []


def test_goal_events_survive_the_accepted_events_filter(monkeypatch):
    out = _run(monkeypatch, hud=_hud())
    accepted = DG.accepted_events_for_clips(out.events, out.clips)
    assert [(e.t, e.data.get("detector")) for e in accepted if e.source == "ocr"] == [
        (100.0, "hud"), (400.0, "hud")]


# --------------------------------------------------------------------------- #
# detect_gameplay without a scoreboard result
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["normal", "no_peaks", "no_rms", "no_audio"])
def test_without_a_scoreboard_result_detection_is_unchanged(monkeypatch, mode):
    def must_not_scan(*a, **k):
        raise AssertionError("track_scoreboard called although CLIPFORGE_HUD=0")

    # The feature switched off: the real _track_hud, which must not even scan.
    monkeypatch.setenv("CLIPFORGE_HUD", "0")
    monkeypatch.setattr(DG.hud_mod, "track_scoreboard", must_not_scan)
    disabled = _run(monkeypatch, hud=REAL_TRACK_HUD, mode=mode)
    # The tracker ran and found nothing usable.
    no_result = _run(monkeypatch, hud=None, mode=mode)

    assert no_result.clips == disabled.clips           # factors and features included
    assert no_result.events == disabled.events
    assert _sig(no_result.clips) == LEGACY_CLIPS[mode]
    assert [(e.t, e.source, e.label) for e in no_result.events] == LEGACY_EVENTS[mode]
    assert all(e.data == {} for e in no_result.events)
    assert len(no_result.ocr_calls) == len(disabled.ocr_calls) == 1
    assert no_result.warnings == disabled.warnings == []


def test_audio_extract_failure_without_a_scoreboard_is_unchanged(monkeypatch):
    out = _run(monkeypatch, hud=None, mode="no_extract")
    assert out.clips == DG._uniform_fallback(_info(), _settings())
    assert out.events == []


# --------------------------------------------------------------------------- #
# _track_hud
# --------------------------------------------------------------------------- #
def _spy_tracker(monkeypatch, result):
    """Replace hud.track_scoreboard; returns the list its calls are logged to."""
    calls: list = []

    def fake(src_path, info, settings=None, **kw):
        calls.append((src_path, info, settings, kw))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.delenv("CLIPFORGE_HUD", raising=False)
    monkeypatch.setattr(DG.hud_mod, "track_scoreboard", fake)
    return calls


def test_track_hud_returns_a_healthy_result(monkeypatch):
    hud = _hud()
    calls = _spy_tracker(monkeypatch, hud)
    info, st, warnings = _info(), _settings(target_clips=4), []
    assert DG._track_hud("match.mp4", info, st, warnings) is hud
    assert warnings == []
    ((src, got_info, got_settings, kw),) = calls
    assert (src, got_info, got_settings) == ("match.mp4", info, st)
    assert kw == {"refine_limit": 30}


def test_track_hud_refine_limit_scales_with_the_wanted_clips(monkeypatch):
    calls = _spy_tracker(monkeypatch, _hud())
    for target, expected in ((1, 30), (10, 30), (11, 33), (20, 60)):
        DG._track_hud("match.mp4", _info(), _settings(target_clips=target), [])
        assert calls[-1][3] == {"refine_limit": expected}


def test_track_hud_skips_the_scan_when_it_cannot_apply(monkeypatch):
    calls = _spy_tracker(monkeypatch, _hud())
    warnings: list = []
    assert DG._track_hud("m.mp4", _info(), _settings(use_ocr=False), warnings) is None
    assert DG._track_hud("m.mp4", _info(has_video=False), _settings(), warnings) is None
    assert calls == []
    # Another game's profile still looks for a scoreboard (a recording can hold
    # several games), and a clean read with goals is used.
    assert DG._track_hud("m.mp4", _info(), _settings(game_profile="valorant"), warnings) is not None
    assert len(calls) == 1
    for value in ("0", "false", "OFF", "No", " off "):
        monkeypatch.setenv("CLIPFORGE_HUD", value)
        assert DG._track_hud("m.mp4", _info(), _settings(), warnings) is None, value
    assert len(calls) == 1
    assert warnings == []

    # Anything else leaves it on; "auto" tries every known layout.
    monkeypatch.setenv("CLIPFORGE_HUD", "1")
    assert DG._track_hud("m.mp4", _info(), _settings(), warnings) is not None
    monkeypatch.delenv("CLIPFORGE_HUD")
    assert DG._track_hud("m.mp4", _info(), _settings(game_profile="auto"), warnings) is not None
    assert len(calls) == 3


def test_track_hud_unhealthy_result_warns_only_for_a_named_scoreboard_game(monkeypatch):
    not_found = ("No scoreboard was found in this footage, so goals fall back to audio "
                 "and on-screen keywords. Scoreboard tracking cannot read a scaled or "
                 "moved game HUD yet.")
    unreadable = ("Scoreboard tracking could not read this video, so goals fall back "
                  "to audio and on-screen keywords.")
    # Two different causes, two different messages: an unhealthy result means
    # the scan ran and saw no scoreboard; None means the scan itself failed.
    for result, message in ((_hud(healthy=False), not_found), (None, unreadable)):
        calls = _spy_tracker(monkeypatch, result)
        for profile in ("eafc", "fifa", "EA FC"):
            warnings: list = []
            assert DG._track_hud("m.mp4", _info(), _settings(game_profile=profile), warnings) is None
            assert warnings == [message], profile
        for profile in ("auto", "generic"):
            warnings = []
            assert DG._track_hud("m.mp4", _info(), _settings(game_profile=profile), warnings) is None
            assert warnings == [], profile
        assert len(calls) == 5


def test_track_hud_crash_warns_and_returns_none(monkeypatch):
    calls = _spy_tracker(monkeypatch, RuntimeError("decoder exploded"))
    warnings: list = []
    assert DG._track_hud("m.mp4", _info(), _settings(game_profile="auto"), warnings) is None
    assert warnings == ["Scoreboard tracking crashed, so goals fall back to audio "
                        "and on-screen keywords."]
    DG._track_hud("m.mp4", _info(), _settings(), warnings)
    assert len(warnings) == 1                          # deduped
    assert len(calls) == 2
    assert DG._track_hud("m.mp4", _info(), _settings(), None) is None   # no warning list given


def test_track_hud_crash_falls_back_to_legacy_detection(monkeypatch):
    _spy_tracker(monkeypatch, RuntimeError("decoder exploded"))
    out = _run(monkeypatch, hud=REAL_TRACK_HUD)
    assert _sig(out.clips) == LEGACY_CLIPS["normal"]
    assert len(out.ocr_calls) == 1
    assert out.warnings == ["Scoreboard tracking crashed, so goals fall back to audio "
                            "and on-screen keywords."]


# --------------------------------------------------------------------------- #
# Recorded events and the models that carry them
# --------------------------------------------------------------------------- #
def test_record_events_writes_goal_events_with_the_full_payload():
    ev = _goal(100.0004, (1, 0), (2, 0), minute=13, clock="12:29", importance=9,
               streamer=True, confidence=0.94999)
    events: list = []
    DG._record_events(events, [CueEvent(t=50.0, label="kill", similarity=0.9)],
                      [OcrEvent(t=200.0, label="victory", text="victory", confidence=0.8)],
                      [AudioEvent(t=150.0, label="crowd_cheer", confidence=0.7, detail="crowd")],
                      hud=_hud([ev]))
    assert [(e.t, e.source) for e in events] == [
        (50.0, "cue"), (100.0, "ocr"), (150.0, "audio"), (200.0, "ocr")]
    goal = events[1]
    assert goal == DetectedEvent(
        t=100.0, source="ocr", label="goal", detail="Scoreboard 1-0 -> 2-0 (13')",
        confidence=0.95,
        data={"detector": "hud", "team": "a", "score_before": "1-0", "score_after": "2-0",
              "minute": 13, "clock": "12:29", "t_change": 101.8, "streamer": True,
              "importance": 9.0})
    assert type(goal.data["minute"]) is int
    assert type(goal.data["streamer"]) is bool
    assert type(goal.data["importance"]) is float
    assert events[3].data == {}                        # keyword hits carry no extras


def test_record_events_keeps_unknown_goal_details_as_none():
    events: list = []
    DG._record_events(events, [], [], [], hud=_hud([_goal(100.0, (0, 0), (0, 1), team="b")]))
    (goal,) = events
    assert goal.data["minute"] is None
    assert goal.data["streamer"] is None
    assert goal.data["clock"] == ""
    assert goal.data["team"] == "b"
    assert goal.detail == "Scoreboard 0-0 -> 0-1"


def test_record_events_without_a_scoreboard_is_unchanged():
    with_kw: list = []
    without: list = []
    cue = [CueEvent(t=50.0, label="kill", similarity=0.9)]
    DG._record_events(with_kw, cue, [], [], hud=None)
    DG._record_events(without, cue, [], [])
    assert with_kw == without == [DetectedEvent(t=50.0, source="cue", label="kill",
                                                detail="kill sound", confidence=0.9)]
    DG._record_events(None, cue, [], [], hud=_hud())   # no sink: nothing to do


def test_detected_event_round_trips_with_and_without_data():
    plain = DetectedEvent(t=1.5, source="cue", label="kill", detail="kill sound", confidence=0.9)
    assert plain.data == {}
    assert DetectedEvent.model_validate(plain.model_dump()) == plain
    assert DetectedEvent.model_validate_json(plain.model_dump_json()) == plain

    rich = DetectedEvent(t=100.0, source="ocr", label="goal", confidence=0.95,
                         data={"detector": "hud", "team": "a", "score_after": "2-0",
                               "minute": 13, "clock": "", "t_change": 101.8,
                               "streamer": False, "importance": 9.0, "unknown": None})
    for back in (DetectedEvent.model_validate(rich.model_dump()),
                 DetectedEvent.model_validate_json(rich.model_dump_json())):
        assert back == rich
        assert back.data["streamer"] is False
        assert back.data["unknown"] is None
        assert type(back.data["minute"]) is int
        assert type(back.data["importance"]) is float


def test_projects_stored_before_these_fields_existed_still_load():
    old = {"name": "Old project",
           "settings": {"game_profile": "eafc",
                        "game_config": {"detection_mode": "hybrid",
                                        "audio_prompts": ["crowd roar"]}},
           "events": [{"t": 12.0, "source": "ocr", "label": "goal",
                       "detail": "GOAL", "confidence": 0.9}]}
    p = Project.model_validate(old)
    assert p.events[0].data == {}
    assert p.settings.game_config.legacy_ocr_with_hud is False
    assert p.settings.game_config.detection_mode == "hybrid"
    assert GameProfileConfig().legacy_ocr_with_hud is False

    # And a project carrying scoreboard goals survives storage.
    p.events.append(DetectedEvent(t=30.0, source="ocr", label="goal",
                                  data={"detector": "hud", "minute": 13, "streamer": True}))
    again = Project.model_validate_json(p.model_dump_json())
    assert again.events == p.events


# --------------------------------------------------------------------------- #
# Evidence caps
# --------------------------------------------------------------------------- #
def test_evidence_caps_leave_scoreboard_goals_alone():
    (goal,) = DG._hud_goal_clips(_hud([_goal(100.0, (0, 0), (1, 0), importance=13.0)]),
                                 _info(), _settings(), lead=10.0, tail=4.0)
    assert goal.score == 97
    loud = DG.GameplayClip(start=0, end=10, score=97, features={"intensity": 1.0})
    cue = DG.GameplayClip(start=20, end=30, score=97, features={"cue": 0.95})
    audio = DG.GameplayClip(start=40, end=50, score=97, features={"audio_event": 0.9})
    # A goal that also matched a sound cue must not fall under the cue-only cap.
    goal_with_cue = DG.GameplayClip(start=60, end=70, score=97,
                                    features={"hud": 1.0, "ocr": 0.95, "cue": 0.9})
    DG.apply_evidence_caps([goal, loud, cue, audio, goal_with_cue])
    assert goal.score == 97
    assert goal_with_cue.score == 97
    assert (loud.score, cue.score, audio.score) == (89, 88, 86)


# --------------------------------------------------------------------------- #
# Orchestrator: what cue learning is allowed to see
# --------------------------------------------------------------------------- #
def test_cue_learning_never_sees_scoreboard_goals():
    import inspect

    from app.pipeline import orchestrator as O

    events: list = []
    DG._record_events(events, [CueEvent(t=50.0, label="kill", similarity=0.9)],
                      [OcrEvent(t=200.0, label="victory", text="victory", confidence=0.8),
                       OcrEvent(t=210.0, label="goal", text="goal", confidence=0.8)],
                      [AudioEvent(t=150.0, label="crowd_cheer", confidence=0.7)],
                      hud=_hud())
    assert len(events) == 6
    kept = O._cue_learning_events(events)
    assert [(e.t, e.label) for e in kept] == [(200.0, "victory"), (210.0, "goal")]
    assert all(e.source == "ocr" and e.data.get("detector") != "hud" for e in kept)

    # Objects without a data attribute (older event shapes) count as keyword hits.
    bare = types.SimpleNamespace(t=1.0, source="ocr", label="ace")
    assert O._cue_learning_events([bare]) == [bare]
    assert O._cue_learning_events([]) == []

    # The detection stage hands exactly this list to cue learning.
    src = inspect.getsource(O.Engine._stage_detect_and_score)
    assert "ocr_evs = _cue_learning_events(detected)" in src
    assert re.search(r"save_audio_cues_from_ocr\(\s*src_path,\s*ocr_evs,", src)

# --------------------------------------------------------------------------- #
# Ranking and safety nets added on top of the wiring
# --------------------------------------------------------------------------- #
def test_guessing_the_game_needs_clean_goals(monkeypatch):
    # With "auto" every known layout is tried on footage of any game, so a
    # layout only counts when it produced goals and no contradictions.
    _spy_tracker(monkeypatch, _hud(events=[]))
    assert DG._track_hud("m.mp4", _info(), _settings(game_profile="auto"), []) is None
    assert DG._track_hud("m.mp4", _info(), _settings(game_profile="eafc"), []) is not None
    doubtful = _hud()
    doubtful.diagnostics["violations"] = 2
    _spy_tracker(monkeypatch, doubtful)
    assert DG._track_hud("m.mp4", _info(), _settings(game_profile="generic"), []) is None
    assert DG._track_hud("m.mp4", _info(), _settings(game_profile="eafc"), []) is doubtful


def test_loudness_breaks_ties_between_goals_and_nothing_more():
    times, rms = _envelope()                           # bursts at 100, 250, 400 and 520 s
    loud, quiet = DG._hud_goal_clips(
        _hud([_goal(100.0, (0, 0), (1, 0), importance=5.0), _goal(180.0, (1, 0), (2, 0), importance=5.0)]),
        _info(), _settings(), lead=10.0, tail=4.0)
    assert loud.score == quiet.score == 89
    DG._add_goal_loudness([loud, quiet], times, rms)
    assert loud.score == 89 + DG.HUD_LOUDNESS_BONUS and quiet.score == 89
    assert loud.factors[-1].label == "Loud moment" and loud.features["intensity"] > 0.95
    assert all(f.label != "Loud moment" for f in quiet.factors)
    assert all(type(v) is float for v in loud.features.values())
    top = DG._hud_goal_clips(_hud([_goal(100.0, (0, 0), (1, 0), importance=14.0)]),
                             _info(), _settings(), lead=10.0, tail=4.0)
    DG._add_goal_loudness(top, times, rms)
    assert top[0].score == 99                          # never above the scale
    DG._add_goal_loudness(top, None, None)             # no envelope: nothing to do
    DG._add_goal_loudness([], times, rms)


def test_gate_stays_open_when_it_would_leave_no_clip_at_all(monkeypatch):
    # A tracked scoreboard, no goals, and every loud moment outside match time.
    away = _hud(events=[], in_play=((560.0, 590.0),), coverage=0.5)
    seen = _run(monkeypatch, hud=away, cues=[], audio=[], ocr=[])
    assert seen.clips, "an empty result fails the whole project"
    # With goals to show, the same loud moments stay out.
    goals = _hud(_two_goals(), in_play=((560.0, 590.0),), coverage=0.5)
    seen = _run(monkeypatch, hud=goals, cues=[], audio=[], ocr=[])
    assert seen.clips and all(_is_goal(c) for c in seen.clips)


def test_hashtag_texts_leave_out_scoreboard_goal_details():
    import inspect

    from app.pipeline import orchestrator

    src = inspect.getsource(orchestrator.Engine._stage_detect_and_score)
    assert re.search(r"ocr_texts\s*=\s*\[e\.detail for e in _cue_learning_events\(detected\)", src)

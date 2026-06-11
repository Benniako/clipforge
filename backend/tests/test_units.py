"""Fast, dependency-light unit tests for the pure pipeline logic.

These don't need ffmpeg, Whisper, or OpenCV — they lock the behaviour of the
parts most prone to silent regressions (caption ASS formatting, the language
lexicons, detection ranking, scoring range, crop geometry).

    cd backend && python -m pytest tests/test_units.py        # or:
    cd backend && python -m tests.test_units                  # no pytest needed
"""
from __future__ import annotations

import os
import tempfile

# Isolate the learning DB before anything imports settings.
os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

from app.models import (Clip, ImportSettings, Platform, Reframe,
                        ReframeKeyframe, Transcript, Word)
from app.pipeline import captionize, render
from app.pipeline import captions as C
from app.providers import detect, score, signals
from app.styles import get_style


def _words(text: str, per: float = 0.4, d: float = 0.34) -> list[Word]:
    out, t = [], 0.0
    for tok in text.split():
        out.append(Word(t=round(t, 3), d=d, text=tok))
        t += per
    return out


# --------------------------------------------------------------------------- #
# Captions — the ASS field-count bug must never come back.
# --------------------------------------------------------------------------- #
def test_dialogue_has_exactly_eight_commas_before_text():
    cs = captionize.build_caption_set(Transcript(words=_words("alpha beta gamma")),
                                      0.0, 2.0, "bold-pop")
    ass = C.build_ass(cs, get_style("bold-pop"), 1080, 1920)
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert dialogues
    for line in dialogues:
        prefix = line[len("Dialogue: "):]
        # Text is field 9; the 8 preceding commas delimit fields 1..8.
        head = prefix.split(",", 8)
        assert len(head) == 9
        text = head[8]
        assert not text.lstrip().startswith(","), f"leading comma in text: {text!r}"


def test_caption_cleaning_strips_leading_punct_and_drops_pure_punct():
    tr = Transcript(words=[Word(t=0, d=0.3, text=", hello"),
                           Word(t=0.4, d=0.3, text=","),
                           Word(t=0.8, d=0.3, text="world.")])
    cs = captionize.build_caption_set(tr, 0.0, 2.0, "bold-pop")
    assert [w.text for w in cs.words] == ["hello", "world."]


def test_ass_timestamp_never_emits_60_seconds():
    from app.pipeline.captions import _ts
    assert _ts(59.999) == "0:01:00.00"   # float edge must roll over, not ":60.00"
    assert _ts(0.0) == "0:00:00.00"
    assert _ts(3661.5) == "1:01:01.50"
    assert _ts(-1.0) == "0:00:00.00"     # clamped


def test_srt_output_format():
    from app.models import CaptionSet, CaptionWord
    from app.pipeline.captions import build_srt
    cs = CaptionSet(words=[CaptionWord(t=0.0, d=0.4, text="Hallo"),
                           CaptionWord(t=0.5, d=0.4, text="Welt")],
                    max_words_per_line=2)
    srt = build_srt(cs)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,900\nHallo Welt")


def test_spa_fallback_serves_index_for_client_routes(tmp_path=None):
    import tempfile
    from pathlib import Path
    from starlette.applications import Starlette
    from starlette.testclient import TestClient
    from app.main import SPAStaticFiles

    d = Path(tempfile.mkdtemp())
    (d / "index.html").write_text("<html>app</html>")
    app = Starlette()
    app.mount("/", SPAStaticFiles(directory=str(d), html=True), name="spa")
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/").status_code == 200
    assert c.get("/p/proj_abc").status_code == 200          # client-side route
    assert "app" in c.get("/p/proj_abc/clip/c1").text       # nested route too


def test_caption_uppercase_and_highlight_present():
    cs = captionize.build_caption_set(Transcript(words=_words("one two three")),
                                      0.0, 2.0, "bold-pop")
    ass = C.build_ass(cs, get_style("bold-pop"), 1080, 1920)
    assert "ONE" in ass  # bold-pop is uppercase
    assert "\\fscx112" in ass  # active-word pop


# --------------------------------------------------------------------------- #
# Language-aware signals
# --------------------------------------------------------------------------- #
def test_german_lexicon_beats_english_on_german_text():
    w = _words("Warum macht das niemand? Das ist unglaublich und verrückt.")
    h_de, _ = signals.hook_strength(w, signals.get_lexicon("de"))
    h_en, _ = signals.hook_strength(w, signals.get_lexicon("en"))
    e_de, _ = signals.emotional_payoff(w, signals.get_lexicon("de"))
    assert h_de > h_en
    assert e_de > 0.0


def test_unknown_language_falls_back_to_english():
    assert signals.get_lexicon("xx") is signals.get_lexicon("en")
    assert signals.get_lexicon(None) is signals.get_lexicon("en")


# --------------------------------------------------------------------------- #
# Detection + scoring
# --------------------------------------------------------------------------- #
def test_detection_returns_within_length_and_count():
    st = ImportSettings(platform=Platform.tiktok, min_len=5, max_len=15, target_clips=3)
    text = ("So here is the biggest secret nobody tells you about this. "
            "What would change if you tried it every single day for a year? "
            "The truth is that small habits compound into something incredible. "
            "I learned this the hard way and it honestly changed everything.")
    cands = detect.detect_moments(_words(text), st, lang="en")
    assert 1 <= len(cands) <= 3
    for c in cands:
        assert 5 <= c.duration <= 15.01


def test_window_fallback_terminates_on_overlong_word():
    # A single word spanning the whole target window (bad ASR timestamp) used
    # to leave the fallback's index stuck — the loop never terminated.
    st = ImportSettings(platform=Platform.tiktok, min_len=5, max_len=15, target_clips=3)
    words = [Word(t=0.0, d=30.0, text="stuck"), Word(t=30.0, d=0.4, text="end")]
    cands = detect._window_fallback(words, st, signals.get_lexicon("en"))
    assert 1 <= len(cands) <= 2  # returning at all means it terminated


def test_synthetic_transcript_stays_within_duration():
    from app.providers import transcribe
    # probe fails on a missing file -> the documented 60s fallback duration
    tr = transcribe.synthetic_transcript("/nonexistent.mp4", lang="en")
    assert tr.words
    assert all(w.t + w.d <= 60.0 for w in tr.words)


def test_score_in_range_and_has_factors():
    st = ImportSettings(platform=Platform.tiktok)
    w = _words("What would change if you showed up every single day? It is incredible.")
    s, factors, feats = score.score_clip(w, 18.0, st, lang="en")
    assert 1 <= s <= 99
    assert len(factors) >= 2
    assert all(f.label for f in factors)
    assert "hook" in feats and 0.0 <= feats["hook"] <= 1.0  # features returned for learning


def test_platform_weights_shift_score():
    st_tt = ImportSettings(platform=Platform.tiktok)
    st_sh = ImportSettings(platform=Platform.shorts)
    w = _words("Why does nobody talk about this? It is the biggest secret ever.")
    s_tt, _, _ = score.score_clip(w, 18.0, st_tt, lang="en")
    s_sh, _, _ = score.score_clip(w, 18.0, st_sh, lang="en")
    assert s_tt != s_sh  # hook-heavy clip scores differently per platform


# --------------------------------------------------------------------------- #
# Reframe / crop geometry (no ffmpeg needed)
# --------------------------------------------------------------------------- #
def test_build_crop_static_when_subject_steady():
    clip = Clip(start=0, end=5, reframe=Reframe(keyframes=[ReframeKeyframe(t=0, cx=0.5)]))
    cw, ch, x = render.build_crop(clip, 1920, 1080)
    assert cw == 606 and ch == 1080  # 9:16 of 1080 height (even)
    assert x.lstrip("-").isdigit()  # static => plain number


def test_build_crop_dynamic_when_subject_moves():
    clip = Clip(start=0, end=6, reframe=Reframe(keyframes=[
        ReframeKeyframe(t=0, cx=0.2), ReframeKeyframe(t=6, cx=0.8)]))
    _, _, x = render.build_crop(clip, 1920, 1080)
    assert "if(" in x and "t" in x  # time-varying expression


def test_reframe_skips_already_vertical_source():
    from app.pipeline import reframe as RF
    rf = RF.compute_reframe("/nonexistent.mp4", 0.0, 5.0, src_aspect=9 / 16)
    assert rf.tracked is False
    assert rf.keyframes[0].cx == 0.5


# --------------------------------------------------------------------------- #
# Facecam — clustering + layout geometry (no cv2/ffmpeg needed)
# --------------------------------------------------------------------------- #
def test_facecam_cluster_finds_stable_corner_face():
    from app.pipeline import facecam as FC
    # A small face pinned bottom-left in every sample + random in-game faces.
    cam = (0.05, 0.70, 0.08, 0.14)
    noise = [(0.4, 0.3, 0.05, 0.09), (0.7, 0.5, 0.04, 0.07), (0.55, 0.2, 0.06, 0.1)]
    samples = [[cam, noise[i % 3]] for i in range(12)]
    rect = FC.stable_face_cluster(samples)
    assert rect is not None
    assert abs((rect.x + rect.w / 2) - 0.09) < 0.02   # centred on the cam face
    overlay = FC.expand_to_overlay(rect)
    assert overlay.w > rect.w and overlay.h > rect.h
    assert 0.0 <= overlay.x and overlay.x + overlay.w <= 1.0
    assert 0.0 <= overlay.y and overlay.y + overlay.h <= 1.0


def test_facecam_cluster_rejects_moving_and_big_faces():
    from app.pipeline import facecam as FC
    # Moving face (a talking head pan) — positions drift too much.
    moving = [[(0.1 + i * 0.03, 0.4, 0.1, 0.16)] for i in range(12)]
    assert FC.stable_face_cluster(moving) is None
    # Stable but huge face — that's talking content, not an overlay.
    big = [[(0.3, 0.2, 0.35, 0.5)] for _ in range(12)]
    assert FC.stable_face_cluster(big) is None
    # Too few samples to trust.
    assert FC.stable_face_cluster([[(0.05, 0.7, 0.08, 0.14)]] * 3) is None


def test_rect_crop_grows_to_aspect_and_clamps():
    from app.models import Rect
    cam = Rect(x=0.02, y=0.72, w=0.18, h=0.24)   # bottom-left cam on 1920x1080
    w, h, x, y = render.rect_crop(cam, 1920, 1080, aspect=1080 / 576)
    assert abs(w / h - 1080 / 576) < 0.02        # grown to the pane aspect
    assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080


def test_game_pane_crop_dodges_facecam():
    from app.models import Rect
    aspect = 1080 / 1344                          # split-layout gameplay pane
    cam = Rect(x=0.0, y=0.6, w=0.30, h=0.4)       # wide cam on the left
    w, h, x, y = render.game_pane_crop(0.25, cam, 1920, 1080, aspect)
    assert x >= int(0.30 * 1920) - 1              # shifted right of the cam
    # No cam -> stays where the action centroid put it.
    w2, h2, x2, _ = render.game_pane_crop(0.5, None, 1920, 1080, aspect)
    assert abs((x2 + w2 / 2) - 960) <= 1


def test_reaction_weights_make_room_and_renormalise():
    from app.providers import detect_gameplay as G
    w = G.with_reaction(G.audio_weights("valorant"))
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["reaction"] == G.REACTION_WEIGHT
    s_calm, _ = G.score_audio({"intensity": 0.8, "reaction": 0.0}, w)
    s_hype, f_hype = G.score_audio({"intensity": 0.8, "reaction": 1.0}, w)
    assert s_hype > s_calm                        # a big reaction raises the score
    assert any("react" in f.label.lower() for f in f_hype)


def test_composed_graph_split_and_framed():
    from app.models import Rect
    cam = Rect(x=0.02, y=0.70, w=0.20, h=0.26)
    clip = Clip(start=0, end=10, reframe=Reframe(
        layout="split", facecam=cam, keyframes=[ReframeKeyframe(t=0, cx=0.5)]))
    info = type("I", (), {"width": 1920, "height": 1080})()
    g = "\n".join(render._composed_graph(clip, cam, info, 1080, 1920, "ass=f=cap.ass"))
    assert "vstack=inputs=2" in g and "[vo]" in g and "ass=f=cap.ass" in g
    clip.reframe.layout = "framed"
    g2 = "\n".join(render._composed_graph(clip, cam, info, 1080, 1920, None))
    assert "overlay=" in g2 and "vstack" not in g2


# --------------------------------------------------------------------------- #
# Transcription engine selection (whisperX > faster-whisper > synthetic)
# --------------------------------------------------------------------------- #
def _settings(**kw):
    from pathlib import Path
    from app.config import Settings
    base = dict(data_dir=Path("/tmp"), db_path=Path("/tmp/x.db"), media_dir=Path("/tmp"),
                ffmpeg="ff", ffprobe="fp", has_whisper=True, has_whisperx=False,
                has_opencv=True, has_ytdlp=True, has_cuda=False, has_nvenc=False,
                has_nvidia=False)
    base.update(kw)
    return Settings(**base)


def test_transcription_engine_prefers_whisperx_then_faster_then_synthetic():
    assert _settings(has_whisperx=True).transcription_engine == "whisperx"
    assert _settings(has_whisperx=False, has_whisper=True).transcription_engine == "whisper"
    assert _settings(has_whisperx=False, has_whisper=False).transcription_engine == "synthetic"


def test_transcription_engine_respects_explicit_preference():
    # force faster-whisper even when whisperX is available
    assert _settings(has_whisperx=True, transcriber="faster").transcription_engine == "whisper"
    # force synthetic
    assert _settings(has_whisperx=True, transcriber="synthetic").transcription_engine == "synthetic"
    # ask for whisperX but it's not installed -> falls back, never crashes
    assert _settings(has_whisperx=False, transcriber="whisperx").transcription_engine == "whisper"


def test_diarization_capability_requires_token():
    assert _settings(has_whisperx=True, hf_token="tok").capability_report()["diarization"] is True
    assert _settings(has_whisperx=True, hf_token=None).capability_report()["diarization"] is False
    assert _settings(has_whisperx=False, hf_token="tok").capability_report()["diarization"] is False


# --------------------------------------------------------------------------- #
# GPU encoding gating — never try NVENC without a real GPU
# --------------------------------------------------------------------------- #
def test_auto_whisper_model_picks_for_hardware():
    from app.config import _auto_whisper_model
    assert _auto_whisper_model(True, 16000, 12) == "large-v3"   # big GPU
    assert _auto_whisper_model(True, 4000, 8) == "medium"       # small GPU
    assert _auto_whisper_model(False, 0, 12) == "small"         # strong CPU
    assert _auto_whisper_model(False, 0, 6) == "base"
    assert _auto_whisper_model(False, 0, 2) == "tiny"           # weak CPU


def test_nvenc_requires_a_real_gpu():
    assert _settings(has_nvenc=True, has_nvidia=False, has_cuda=False).use_nvenc is False
    assert _settings(has_nvenc=True, has_nvidia=True).use_nvenc is True
    assert _settings(has_nvenc=False, has_nvidia=True).use_nvenc is False  # no encoder


def test_encoder_args_switch():
    cpu = _settings(has_nvenc=False).video_encoder_args()
    gpu = _settings(has_nvenc=True, has_nvidia=True).video_encoder_args()
    assert "libx264" in cpu
    assert "h264_nvenc" in gpu


# --------------------------------------------------------------------------- #
# Aspect ratios + hashtags
# --------------------------------------------------------------------------- #
def test_aspect_dims():
    from app.models import ImportSettings
    assert ImportSettings(aspect="9:16").dims() == (1080, 1920)
    assert ImportSettings(aspect="1:1").dims() == (1080, 1080)
    assert ImportSettings(aspect="4:5").dims() == (1080, 1350)
    assert ImportSettings(aspect="bogus").dims() == (1080, 1920)  # safe default


def test_hashtags_talking_vs_gameplay():
    from app.providers import hashtags
    talk = hashtags.suggest_hashtags("consistency discipline mindset success habits",
                                     content_type="talking", platform="reels")
    assert any(t.startswith("#") for t in talk)
    assert "#reels" in talk
    game = hashtags.suggest_hashtags("insane valorant ace clutch round",
                                     content_type="gameplay", platform="tiktok")
    assert "#gaming" in game and "#valorant" in game and "#tiktok" in game


def test_speech_coverage_drives_classification_inputs():
    from app.pipeline.classify import speech_coverage
    from app.models import Transcript, Word
    tr = Transcript(words=[Word(t=0, d=0.5, text="a"), Word(t=1, d=0.5, text="b")])
    assert abs(speech_coverage(tr, 10.0) - 0.1) < 1e-6
    assert speech_coverage(None, 10.0) == 0.0


# --------------------------------------------------------------------------- #
# Aspect 16:9, game profiles, montage scoring
# --------------------------------------------------------------------------- #
def test_horizontal_aspect():
    from app.models import ImportSettings
    assert ImportSettings(aspect="16:9").dims() == (1920, 1080)


def test_game_profiles_bias_signals():
    from app.providers.detect_gameplay import get_profile
    assert get_profile("auto") is get_profile("generic")
    assert get_profile("fifa") is get_profile("eafc")
    # horror rewards sudden spikes; EA FC rewards sustained loudness
    assert get_profile("horror")["w_transient"] > get_profile("eafc")["w_transient"]
    assert get_profile("eafc")["w_sustain"] > get_profile("horror")["w_sustain"]
    assert get_profile("nonsense") is get_profile("generic")  # safe default


def test_cue_template_matching_finds_inserted_sound():
    import numpy as np
    from app.providers.detect_cues import match_template, SR
    t = np.linspace(0, 0.4, int(0.4 * SR), endpoint=False)
    chirp = (0.8 * np.sin(2 * np.pi * (800 + 1500 * t) * t)).astype(np.float32)
    rng = np.random.default_rng(0)
    sig = (0.02 * rng.standard_normal(15 * SR)).astype(np.float32)
    for at in (4.0, 10.0):
        i = int(at * SR)
        sig[i:i + len(chirp)] += chirp
    hits = sorted(round(t0) for t0, _ in match_template(sig, chirp, threshold=0.5, min_gap=2.0))
    assert any(abs(h - 4) <= 1 for h in hits)
    assert any(abs(h - 10) <= 1 for h in hits)


def test_montage_score_weights_opening():
    from app.pipeline.montage import score_montage
    strong_open = score_montage([Clip(start=0, end=5, score=90),
                                 Clip(start=0, end=5, score=50)])[0]
    weak_open = score_montage([Clip(start=0, end=5, score=50),
                               Clip(start=0, end=5, score=90)])[0]
    assert 1 <= strong_open <= 99 and 1 <= weak_open <= 99
    assert strong_open > weak_open  # same clips, better hook first -> higher score
    score, factors = score_montage([Clip(start=0, end=5, score=80),
                                    Clip(start=0, end=5, score=78)])
    assert len(factors) >= 1


# --------------------------------------------------------------------------- #
# Local learning loop (feedback.py)
# --------------------------------------------------------------------------- #
_BASE = {"hook": 0.30, "emotion": 0.20, "clarity": 0.20, "quote": 0.10,
         "pace": 0.10, "length": 0.05, "list": 0.05}


def _feats(**over):
    f = {k: 0.3 for k in _BASE}
    f.update(over)
    return f


def test_learning_cold_start_returns_base():
    from app import feedback
    feedback.init_db()
    feedback.reset("score:cold")
    w = feedback.learned_weights("score:cold", _BASE)
    assert w == _BASE  # no feedback -> defaults unchanged


def test_learning_shifts_weights_toward_liked_features():
    from app import feedback
    feedback.init_db()
    sc = "score:shift"
    feedback.reset(sc)
    # You keep emotional clips and reject hooky ones.
    for i in range(12):
        feedback.record_rating(f"up{i}", sc, 1.0, _feats(emotion=0.9, hook=0.1))
        feedback.record_rating(f"dn{i}", sc, 0.0, _feats(emotion=0.1, hook=0.9))
    w = feedback.learned_weights(sc, _BASE)
    assert w["emotion"] > _BASE["emotion"]      # learned you like payoff
    assert w["hook"] < _BASE["hook"]            # and care less about hooks
    assert abs(sum(w.values()) - sum(_BASE.values())) < 1e-6  # total preserved


def test_learning_confidence_grows_with_data():
    from app import feedback
    feedback.init_db()
    few, many = "score:few", "score:many"
    feedback.reset(few)
    feedback.reset(many)
    feedback.record_rating("f1", few, 1.0, _feats(emotion=0.9, hook=0.1))
    feedback.record_rating("f2", few, 0.0, _feats(emotion=0.1, hook=0.9))
    for i in range(15):
        feedback.record_rating(f"m_up{i}", many, 1.0, _feats(emotion=0.9, hook=0.1))
        feedback.record_rating(f"m_dn{i}", many, 0.0, _feats(emotion=0.1, hook=0.9))
    shift_few = feedback.learned_weights(few, _BASE)["emotion"] - _BASE["emotion"]
    shift_many = feedback.learned_weights(many, _BASE)["emotion"] - _BASE["emotion"]
    assert shift_few > 0                  # even a little feedback nudges
    assert shift_many > shift_few         # more feedback -> stronger personalization


def test_gameplay_scoring_respects_profile():
    from app.providers.detect_gameplay import audio_weights, score_audio
    transient = {"intensity": 0.5, "sustain": 0.1, "transient": 0.9, "spikes": 0.1, "cue": 0.0}
    sustained = {"intensity": 0.5, "sustain": 0.9, "transient": 0.1, "spikes": 0.1, "cue": 0.0}
    assert score_audio(transient, audio_weights("horror"))[0] > \
        score_audio(sustained, audio_weights("horror"))[0]
    assert score_audio(sustained, audio_weights("eafc"))[0] > \
        score_audio(transient, audio_weights("eafc"))[0]


def test_gameplay_learns_from_feedback():
    from app import feedback
    from app.providers.detect_gameplay import audio_weights
    feedback.init_db()
    sc = feedback.score_scope("gameplay", "eafc")
    feedback.reset(sc)
    base = audio_weights("eafc")
    transient = {"intensity": 0.5, "sustain": 0.1, "transient": 0.9, "spikes": 0.1, "cue": 0.0}
    sustained = {"intensity": 0.5, "sustain": 0.9, "transient": 0.1, "spikes": 0.1, "cue": 0.0}
    for i in range(12):  # you actually prefer sudden plays, even in EA FC
        feedback.record_rating(f"u{i}", sc, 1.0, transient)
        feedback.record_rating(f"d{i}", sc, 0.0, sustained)
    w = feedback.learned_weights(sc, base)
    assert w["transient"] > base["transient"]


def test_boundary_correction_converges_and_gates():
    from app import feedback
    feedback.init_db()
    sc = "bound:test"
    feedback.reset(sc)
    feedback.record_trim(sc, 1.0, -0.5)
    feedback.record_trim(sc, 1.0, -0.5)
    assert feedback.boundary_correction(sc) == (0.0, 0.0)  # < min samples
    feedback.record_trim(sc, 1.0, -0.5)
    cs, ce = feedback.boundary_correction(sc)
    assert 0.7 <= cs <= 0.9 and -0.5 <= ce <= -0.3        # damped median, clamped
    feedback.reset(sc)
    assert feedback.boundary_correction(sc) == (0.0, 0.0)  # reset works


# --------------------------------------------------------------------------- #
# Silence tightening (jump cuts)
# --------------------------------------------------------------------------- #
def _tight_transcript():
    # two sentences separated by 3s of dead air
    words = []
    t = 0.0
    for tok in "this is the first sentence".split():
        words.append(Word(t=t, d=0.3, text=tok)); t += 0.35
    t += 3.0
    for tok in "and here comes the payoff".split():
        words.append(Word(t=t, d=0.3, text=tok)); t += 0.35
    return Transcript(words=words, provider="whisper")


def test_tight_segments_cut_dead_air():
    from app.pipeline.captionize import compute_tight_segments
    tr = _tight_transcript()
    end = tr.words[-1].end + 0.2
    segs = compute_tight_segments(tr, 0.0, end)
    assert len(segs) == 2
    kept = sum(b - a for a, b in segs)
    assert kept < end - 2.0                       # the 3s gap mostly removed
    # and a clip with no long gaps stays whole
    segs1 = compute_tight_segments(tr, 0.0, tr.words[4].end + 0.1)
    assert len(segs1) == 1


def test_map_to_tight_and_caption_retime():
    from app.pipeline.captionize import (build_tight_caption_set,
                                         compute_tight_segments, map_to_tight)
    tr = _tight_transcript()
    end = tr.words[-1].end + 0.2
    segs = compute_tight_segments(tr, 0.0, end)
    (a1, b1), (a2, b2) = segs
    assert map_to_tight(a1, segs) == 0.0
    assert abs(map_to_tight(a2, segs) - (b1 - a1)) < 1e-6   # 2nd seg starts where 1st ends
    cs = build_tight_caption_set(tr, segs, "bold-pop")
    total = sum(b - a for a, b in segs)
    assert all(w.t < total + 0.01 for w in cs.words)        # all words inside tight timeline
    assert len(cs.words) == len(tr.words)


def test_logistic_learner_kicks_in_with_data():
    from app import feedback
    feedback.init_db()
    sc = "score:logreg"
    feedback.reset(sc)
    for i in range(20):  # 40 weighted samples -> logistic path
        feedback.record_rating(f"u{i}", sc, 1.0, _feats(emotion=0.9, hook=0.15))
        feedback.record_rating(f"d{i}", sc, 0.0, _feats(emotion=0.1, hook=0.85))
    w = feedback.learned_weights(sc, _BASE)
    assert w["emotion"] == max(w.values())                  # right feature on top
    assert abs(sum(w.values()) - sum(_BASE.values())) < 1e-6


def test_llm_titles_safe_when_unavailable():
    from app.providers import llm
    # no Ollama in CI -> must return {} fast instead of raising/hanging
    assert llm.suggest_titles(["some transcript"], lang="en", budget=2.0) == {}


if __name__ == "__main__":
    import sys
    # Windows consoles default to a legacy code page that can't print "✓".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} unit tests passed")

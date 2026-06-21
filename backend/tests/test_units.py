"""Fast, dependency-light unit tests for the pure pipeline logic.

These don't need ffmpeg, Whisper, or OpenCV — they lock the behaviour of the
parts most prone to silent regressions (caption ASS formatting, the language
lexicons, detection ranking, scoring range, crop geometry).

    cd backend && python -m pytest tests/test_units.py        # or:
    cd backend && python -m tests.test_units                  # no pytest needed
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

# Isolate the learning DB before anything imports settings.
os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

from app.models import (Clip, ClipStatus, DetectedEvent, GameProfileConfig,
                        ImportSettings, Platform, Project, ProjectStatus, Reframe,
                        ReframeKeyframe, SourceMedia, Transcript, Word)
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
    from app.pipeline.captions import _srt_ts, build_srt
    cs = CaptionSet(words=[CaptionWord(t=0.0, d=0.4, text="Hallo"),
                           CaptionWord(t=0.5, d=0.4, text="Welt")],
                    max_words_per_line=2)
    srt = build_srt(cs)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,900\nHallo Welt")
    assert _srt_ts(1.9996) == "00:00:02,000"
    assert _srt_ts(59.9996) == "00:01:00,000"


def test_premiere_edl_uses_source_timecodes():
    from app.pipeline.nle_export import build_cmx3600, timecode

    p = Project(
        name="EDL Test",
        source=SourceMedia(filename="source.mp4", path="proj/source.mp4", fps=30),
        clips=[
            Clip(start=10.0, end=15.0, title="First", score=91, export_url="/media/a.mp4"),
            Clip(start=30.0, end=33.0, title="Second", score=72, export_url="/media/b.mp4"),
        ],
    )
    edl = build_cmx3600(p, source_file="D:/clips/source.mp4")
    assert timecode(10.0, 30) == "00:00:10:00"
    assert "00:00:10:00 00:00:15:00 00:00:00:00 00:00:05:00" in edl
    assert "* SOURCE FILE: D:/clips/source.mp4" in edl


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


def test_generated_media_urls_are_browser_paths():
    from app.config import get_settings
    from app.pipeline.orchestrator import _media_url

    p = get_settings().media_dir / "proj_x" / "clips" / "clip_1.mp4"
    url = _media_url(p)
    assert url == "/media/proj_x/clips/clip_1.mp4"
    assert "\\" not in url


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
    assert "instant_hook" in feats and "swipe" in feats


def test_first_two_seconds_reduce_swipe_risk():
    strong = _words("Why does nobody show you this trick? It changed everything")
    soft = _words("Um okay so this is just a normal update")
    lex = signals.get_lexicon("en")
    ih_strong, _ = signals.instant_hook(strong, lex)
    ih_soft, _ = signals.instant_hook(soft, lex)
    sw_strong, _ = signals.swipe_resistance(strong, 12.0, lex)
    sw_soft, _ = signals.swipe_resistance(soft, 12.0, lex)
    assert ih_strong > ih_soft
    assert sw_strong > sw_soft


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


def test_facecam_person_cutout_fallback_for_background_removed_cam():
    from app.pipeline import facecam as FC
    cutout = (0.04, 0.48, 0.16, 0.42)
    samples = [[cutout] for _ in range(12)]
    rect = FC.stable_person_cluster(samples)
    assert rect is not None
    overlay = FC.expand_person_to_overlay(rect)
    assert overlay.w > rect.w and overlay.h > rect.h
    moving = [[(0.05 + i * 0.04, 0.48, 0.16, 0.42)] for i in range(12)]
    assert FC.stable_person_cluster(moving) is None
    huge = [[(0.2, 0.05, 0.55, 0.88)] for _ in range(12)]
    assert FC.stable_person_cluster(huge) is None


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
    assert _settings(has_whisperx=True, hf_token="tok").capability_report()["diarization_model"] == "pyannote/speaker-diarization-community-1"
    assert _settings(has_whisperx=True, hf_token=None).capability_report()["diarization_model"] is None


def test_whisperx_diarization_constructor_supports_new_and_old_token_api():
    from app.providers import transcribe as T

    old_get_settings, old_pipe = T.get_settings, T._wx_diarize
    old_parent = sys.modules.get("whisperx")
    old_child = sys.modules.get("whisperx.diarize")
    calls = []

    parent = types.ModuleType("whisperx")
    parent.__path__ = []
    child = types.ModuleType("whisperx.diarize")

    class NewPipeline:
        def __init__(self, **kw):
            calls.append(kw)

    child.DiarizationPipeline = NewPipeline
    sys.modules["whisperx"] = parent
    sys.modules["whisperx.diarize"] = child
    T.get_settings = lambda: _settings(
        has_whisperx=True, hf_token="tok", device="cuda",
        diarization_model="pyannote/test-model")
    T._wx_diarize = None
    try:
        assert isinstance(T._diarization_pipeline(), NewPipeline)
        assert calls[-1] == {
            "model_name": "pyannote/test-model",
            "token": "tok",
            "device": "cuda",
        }

        class OldPipeline:
            def __init__(self, **kw):
                if "token" in kw:
                    raise TypeError("old whisperX")
                calls.append(kw)

        child.DiarizationPipeline = OldPipeline
        T._wx_diarize = None
        assert isinstance(T._diarization_pipeline(), OldPipeline)
        assert calls[-1] == {"use_auth_token": "tok", "device": "cuda"}
    finally:
        T.get_settings, T._wx_diarize = old_get_settings, old_pipe
        if old_parent is None:
            sys.modules.pop("whisperx", None)
        else:
            sys.modules["whisperx"] = old_parent
        if old_child is None:
            sys.modules.pop("whisperx.diarize", None)
        else:
            sys.modules["whisperx.diarize"] = old_child


def test_german_transcription_prompt_is_optional_and_backward_compatible():
    from app.providers import transcribe as T

    old_get = T.get_settings
    T.get_settings = lambda: _settings(german_gaming_prompt="Deutsch Gaming Prompt")

    class OldEngine:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, **kwargs):
            self.calls.append(kwargs)
            if "initial_prompt" in kwargs:
                raise TypeError("unexpected keyword initial_prompt")
            return "segments", "info"

    try:
        assert T._initial_prompt("de") == "Deutsch Gaming Prompt"
        assert T._initial_prompt("en") is None
        engine = OldEngine()
        assert T._transcribe_with_prompt(engine, "audio.wav", language="de",
                                         initial_prompt="Prompt") == ("segments", "info")
        assert "initial_prompt" in engine.calls[0]
        assert "initial_prompt" not in engine.calls[-1]
    finally:
        T.get_settings = old_get


# --------------------------------------------------------------------------- #
# GPU encoding gating — never try NVENC without a real GPU
# --------------------------------------------------------------------------- #
def test_auto_whisper_model_picks_for_hardware():
    from app.config import _auto_whisper_model
    assert _auto_whisper_model(True, 16000, 12) == "large-v3-turbo"  # big GPU: turbo
    assert _auto_whisper_model(True, 4000, 8) == "medium"       # small GPU
    assert _auto_whisper_model(False, 0, 12) == "small"         # strong CPU
    assert _auto_whisper_model(False, 0, 6) == "base"
    assert _auto_whisper_model(False, 0, 2) == "tiny"           # weak CPU


def test_nvenc_requires_a_real_gpu():
    assert _settings(has_nvenc=True, has_nvidia=False, has_cuda=False).use_nvenc is False
    assert _settings(has_nvenc=True, has_nvidia=True).use_nvenc is True
    assert _settings(has_nvenc=False, has_nvidia=True).use_nvenc is False  # no encoder


def test_upload_cap_unlimited_by_default():
    assert _settings(max_upload_mb=0).upload_cap_bytes is None      # 0 = no cap
    assert _settings(max_upload_mb=10).upload_cap_bytes == 10 * 1024 * 1024


def test_encoder_args_switch():
    cpu = _settings(has_nvenc=False).video_encoder_args()
    gpu = _settings(has_nvenc=True, has_nvidia=True).video_encoder_args()
    assert "libx264" in cpu
    assert "h264_nvenc" in gpu


def test_power_mode_scales_local_engine():
    s = _settings(has_cuda=True, vram_mb=16000, device="cuda",
                  whisper_batch_size=8, render_workers=2)
    assert s.whisper_batch_for("max_gpu") >= 16
    assert s.render_workers_for("max_gpu") >= s.render_workers
    assert s.vlm_options_for("quality")["n_frames"] > s.vlm_options_for("balanced")["n_frames"]
    assert s.capability_report()["recommended_power_mode"] == "max_gpu"


def test_av1_codec_opt_in_with_safe_fallbacks():
    # av1 requested + encoder present -> av1_nvenc
    av1 = _settings(has_nvenc=True, has_nvidia=True, has_av1_nvenc=True,
                    codec="av1").video_encoder_args()
    assert "av1_nvenc" in av1
    # av1 requested but the ffmpeg build lacks the encoder -> h264_nvenc
    no_enc = _settings(has_nvenc=True, has_nvidia=True, has_av1_nvenc=False,
                       codec="av1").video_encoder_args()
    assert "h264_nvenc" in no_enc
    # av1 requested with no GPU at all -> x264
    no_gpu = _settings(has_nvenc=False, codec="av1").video_encoder_args()
    assert "libx264" in no_gpu
    # default codec ignores the AV1 encoder even when present
    default = _settings(has_nvenc=True, has_nvidia=True,
                        has_av1_nvenc=True).video_encoder_args()
    assert "h264_nvenc" in default


# --------------------------------------------------------------------------- #
# Aspect ratios + hashtags
# --------------------------------------------------------------------------- #
def test_aspect_dims():
    from app.models import ImportSettings
    assert ImportSettings(aspect="9:16").dims() == (1080, 1920)
    assert ImportSettings(aspect="1:1").dims() == (1080, 1080)
    assert ImportSettings(aspect="4:5").dims() == (1080, 1350)
    assert ImportSettings(aspect="bogus").dims() == (1080, 1920)  # safe default


def test_game_profile_config_defaults_and_roi_clamp():
    cfg = GameProfileConfig(visual_rois=[{"x": -1, "y": 0.9, "w": 2, "h": 0.5}])
    roi = cfg.visual_rois[0].clamped()
    assert cfg.audio_negative_prompts
    assert "kill feed" in cfg.vlm_visual_prompts
    assert roi.x == 0.0 and roi.w == 1.0
    assert 0.0 <= roi.y <= 1.0 and roi.y + roi.h <= 1.0


def test_detection_mode_manual_skips_zero_shot_audio_events():
    """detection_mode='manual' must disable the zero-shot CLAP sweep, while
    'zero_shot'/'hybrid' still run it. Guards against the field being dead code."""
    from app.providers import detect_gameplay as G
    from app.models import ImportSettings, GameProfileConfig

    calls = []
    orig = G.audio_events_mod.find_events
    G.audio_events_mod.find_events = lambda *a, **k: calls.append(1) or []
    try:
        manual = ImportSettings(use_audio_events=True,
                                game_config=GameProfileConfig(detection_mode="manual"))
        assert G._audio_events("a.wav", 30.0, manual) == []
        assert calls == []  # zero-shot sweep was skipped

        auto = ImportSettings(use_audio_events=True,
                              game_config=GameProfileConfig(detection_mode="zero_shot"))
        G._audio_events("a.wav", 30.0, auto)
        assert calls == [1]  # zero-shot sweep ran
    finally:
        G.audio_events_mod.find_events = orig


def test_detector_failures_surface_warnings():
    """A crashing CLAP/OCR detector must record a UI warning instead of silently
    returning [] with no explanation."""
    from app.providers import detect_gameplay as G
    from app.models import ImportSettings, GameProfileConfig

    def _boom(*a, **k):
        raise RuntimeError("model missing")

    warnings: list[str] = []
    orig = G.audio_events_mod.find_events
    G.audio_events_mod.find_events = _boom
    try:
        st = ImportSettings(use_audio_events=True,
                            game_config=GameProfileConfig(detection_mode="zero_shot"))
        assert G._audio_events("a.wav", 30.0, st, warnings_out=warnings) == []
        assert any("CLAP" in w for w in warnings)
        # deduped: a second failure does not append a duplicate
        G._audio_events("a.wav", 30.0, st, warnings_out=warnings)
        assert len(warnings) == 1
    finally:
        G.audio_events_mod.find_events = orig


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


def test_gameplay_detection_from_prepared_wav():
    """The pipeline hands the detector its already-extracted wav — this path
    must work without ffmpeg and find the loud moment."""
    import math
    import struct
    import wave as wave_mod

    from app.media.ffmpeg import MediaInfo
    from app.providers.detect_gameplay import detect_gameplay

    sr, dur = 16000, 60
    fd, wav_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave_mod.open(wav_name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(sr * dur):
            t = i / sr
            amp = 0.7 if 30.0 <= t < 31.0 else 0.01   # one loud second at t=30
            frames += struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * 220 * t)))
        wf.writeframes(bytes(frames))
    try:
        info = MediaInfo(duration=dur, width=1920, height=1080, fps=30,
                         has_audio=True, has_video=True, codec=None)
        st = ImportSettings(min_len=8, max_len=20, target_clips=3)
        clips = detect_gameplay("unused.mp4", info, st, wav_path=wav_name)
    finally:
        os.unlink(wav_name)
    assert clips
    assert any(c.start <= 30.0 <= c.end for c in clips), \
        f"no clip covers the burst: {[(c.start, c.end) for c in clips]}"


def test_speech_aware_reframe_helpers():
    from app.pipeline.orchestrator import _speech_intervals
    from app.pipeline.reframe import _speech_active

    # No intervals known -> always "active" (legacy behaviour).
    assert _speech_active(1.0, None) is True
    assert _speech_active(1.0, []) is True
    assert _speech_active(1.0, [(0.5, 2.0)]) is True
    assert _speech_active(3.0, [(0.5, 2.0)]) is False

    # Synthetic transcripts opt out — their filler timing isn't real speech.
    assert _speech_intervals(Transcript(provider="synthetic"), 0, 10) is None
    assert _speech_intervals(None, 0, 10) is None
    tr = _tight_transcript()  # two sentences around a 3s gap
    iv = _speech_intervals(tr, 0.0, tr.words[-1].end + 0.2)
    assert iv and len(iv) == 2
    assert iv[0][0] >= 0.0 and iv[1][0] > iv[0][1]   # clip-relative, disjoint


def test_scene_showinfo_parse_and_snap():
    from app.providers import scenes
    err = ("[Parsed_showinfo_1 @ 0x1] n:   0 pts:  12345 pts_time:1.04  fmt:yuv420p\n"
           "[Parsed_showinfo_1 @ 0x1] n:   1 pts:  98765 pts_time:7.5 fmt:yuv420p\n"
           "frame=  2 fps=0.0 q=-0.0\n")
    assert scenes.parse_showinfo_times(err) == [1.04, 7.5]
    cuts = [10.0, 14.2]
    assert scenes.snap(11.0, cuts, window=1.5) == 10.0   # nearest within window
    assert scenes.snap(13.0, cuts, window=1.5) == 14.2   # 3.0 vs 1.2 -> closer cut
    assert scenes.snap(12.0, cuts, window=1.0) == 12.0   # nothing in range
    assert scenes.snap(5.0, [], window=2.0) == 5.0       # no cuts at all


def test_status_ws_reports_missing_project():
    from starlette.testclient import TestClient
    from app import store
    from app.main import create_app

    store.init_db()  # TestClient without a context manager skips lifespan
    c = TestClient(create_app(), raise_server_exceptions=False)
    with c.websocket_connect("/api/projects/proj_missing/ws") as ws:
        assert ws.receive_json() == {"error": "project not found"}


def test_pause_resume_project_endpoint():
    from starlette.testclient import TestClient
    from app import store
    from app.main import create_app

    store.init_db()
    p = Project(
        id="proj_pause_resume",
        status=ProjectStatus.processing,
        source=SourceMedia(filename="source.mp4", path="proj_pause_resume/source.mp4"),
    )
    store.save(p)
    c = TestClient(create_app(), raise_server_exceptions=False)

    paused = c.post(f"/api/projects/{p.id}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    assert "Paused" in paused.json()["progress"]["message"]

    resumed = c.post(f"/api/projects/{p.id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["status"] in {"queued", "processing"}

    missing = c.post("/api/projects/proj_missing/pause")
    assert missing.status_code == 404


def test_render_finish_fails_project_when_no_clips_ready():
    from app import store
    from app.pipeline.orchestrator import Engine

    store.init_db()
    p = Project(
        id="proj_render_finish_none_ready",
        status=ProjectStatus.processing,
        clips=[
            Clip(start=0, end=5, status=ClipStatus.failed, error="boom"),
            Clip(start=5, end=10, status=ClipStatus.failed, error="boom"),
        ],
    )
    store.save(p)
    Engine()._finish_render_progress(p.id)
    out = store.get(p.id)
    assert out is not None
    assert out.status == ProjectStatus.failed
    assert "every clip" in (out.error or "").lower()
    assert out.progress.stage == "render"


def test_multikill_streaks_chain_and_outscore_single_kills():
    from app.media.ffmpeg import MediaInfo
    from app.providers.detect_cues import CueEvent
    from app.providers.detect_gameplay import _cue_clips, group_streaks

    evs = [CueEvent(t=100.0, label="kill", similarity=0.8),
           CueEvent(t=104.0, label="kill", similarity=0.9),
           CueEvent(t=109.0, label="kill", similarity=0.85),
           CueEvent(t=300.0, label="kill", similarity=0.9)]   # far away: own clip
    groups = group_streaks(evs)
    assert [len(g) for g in groups] == [3, 1]

    info = MediaInfo(duration=600, width=1920, height=1080, fps=30,
                     has_audio=True, has_video=True, codec=None)
    st = ImportSettings(min_len=15, max_len=45)
    clips = _cue_clips(evs, info, lead=10.0, tail=8.0, settings=st)
    triple, single = clips[0], clips[1]
    assert triple.score > single.score            # more kills = more viral
    assert "Triple Kill" in triple.title
    assert triple.features["streak"] > single.features["streak"]
    assert triple.start <= 100.0 and triple.end >= 109.0   # covers all kills
    assert len(triple.cue_ts) == 3
    assert any("streak" in f.label for f in triple.factors)


def test_cue_streaks_do_not_chain_unrelated_valorant_events():
    from app.media.ffmpeg import MediaInfo
    from app.providers.detect_cues import CueEvent
    from app.providers.detect_gameplay import (_cue_clips, apply_evidence_caps,
                                               group_streaks, GameplayClip)

    evs = [CueEvent(t=10.0, label="spike_plant", similarity=0.95),
           CueEvent(t=13.0, label="spike_defuse", similarity=0.96),
           CueEvent(t=16.0, label="kill", similarity=0.94)]
    groups = group_streaks(evs)
    assert [[e.label for e in g] for g in groups] == [
        ["spike_plant"], ["spike_defuse"], ["kill"]]

    info = MediaInfo(duration=120, width=1920, height=1080, fps=30,
                     has_audio=True, has_video=True, codec=None)
    clips = _cue_clips(evs, info, lead=5.0, tail=5.0,
                       settings=ImportSettings(min_len=8, max_len=20))
    assert all(c.score <= 88 for c in clips)
    assert not any("Streak:" in c.title for c in clips)

    c = GameplayClip(start=0, end=10, score=99, features={"cue": 0.99})
    apply_evidence_caps([c])
    assert c.score == 88
    confirmed = GameplayClip(start=0, end=10, score=99,
                             features={"cue": 0.99, "ocr": 0.9})
    apply_evidence_caps([confirmed])
    assert confirmed.score == 99


def test_custom_sound_cues_can_be_disabled():
    from app.providers import detect_gameplay as DG

    calls = []
    old_find = DG.detect_cues.find_events
    DG.detect_cues.find_events = lambda wav, path: calls.append((wav, path)) or ["hit"]
    try:
        disabled = DG._cue_events("source.wav", ImportSettings(
            game_profile="valorant", use_cues=False))
        assert disabled == []
        assert calls == []

        enabled = DG._cue_events("source.wav", ImportSettings(
            game_profile="valorant", use_cues=True))
        assert enabled == ["hit", "hit"]      # active game + common cues
        assert len(calls) == 2
    finally:
        DG.detect_cues.find_events = old_find


def test_corroboration_requires_same_moment_cluster():
    from app.providers.detect_cues import CueEvent
    from app.providers.detect_ocr import OcrEvent
    from app.providers.audio_events import AudioEvent
    from app.providers.detect_gameplay import GameplayClip, apply_corroboration

    far = GameplayClip(start=90, end=130, score=80, peak_t=100.0,
                       features={})
    apply_corroboration(
        [far],
        [CueEvent(t=100.0, label="kill", similarity=0.95)],
        [OcrEvent(t=120.0, label="kill", text="headshot", confidence=0.9)],
    )
    assert far.score == 80
    assert "corroborated" not in far.features

    near = GameplayClip(start=90, end=130, score=80, peak_t=100.0,
                        features={})
    apply_corroboration(
        [near],
        [CueEvent(t=100.0, label="kill", similarity=0.95)],
        [OcrEvent(t=102.0, label="kill", text="headshot", confidence=0.9)],
    )
    assert near.score > 80
    assert near.features["corroborated"] == 1.0
    assert near.features["cue"] == 0.95

    mismatch = GameplayClip(start=90, end=130, score=80, peak_t=100.0,
                            features={})
    apply_corroboration(
        [mismatch],
        [CueEvent(t=100.0, label="spike_plant", similarity=0.95)],
        [OcrEvent(t=101.0, label="kill", text="headshot", confidence=0.9)],
    )
    assert mismatch.score == 80
    assert "corroborated" not in mismatch.features

    soft = GameplayClip(start=90, end=130, score=80, peak_t=100.0,
                        features={})
    apply_corroboration(
        [soft],
        [],
        [OcrEvent(t=100.0, label="kill", text="headshot", confidence=0.9)],
        [AudioEvent(t=101.0, label="explosive_action", confidence=0.9)],
    )
    assert soft.score == 83
    assert "corroborated" not in soft.features
    assert soft.features["supporting_audio"] == 1.0


def test_only_accepted_events_are_shown_for_final_clips():
    from app.providers.detect_gameplay import GameplayClip, accepted_events_for_clips

    clip = GameplayClip(start=90, end=130, score=80, peak_t=100.0,
                        cue_ts=[104.0])
    events = [
        DetectedEvent(t=100.0, source="cue", label="kill", confidence=0.95),
        DetectedEvent(t=104.3, source="cue", label="kill", confidence=0.92),
        DetectedEvent(t=120.0, source="ocr", label="kill", confidence=0.9),
        DetectedEvent(t=300.0, source="cue", label="kill", confidence=0.99),
    ]
    kept = accepted_events_for_clips(events, [clip])
    assert [(e.source, e.label, round(e.t, 1)) for e in kept] == [
        ("cue", "kill", 100.0),
        ("cue", "kill", 104.3),
    ]

    shifted_ocr_clip = GameplayClip(start=90, end=130, score=88, peak_t=90.0,
                                    features={"ocr": 0.9, "evidence_t": 120.0})
    kept = accepted_events_for_clips(events, [shifted_ocr_clip])
    assert [(e.source, e.label, round(e.t, 1)) for e in kept] == [
        ("ocr", "kill", 120.0),
    ]


def test_single_audio_or_model_signal_cannot_score_99():
    from app.providers.detect_gameplay import GameplayClip, apply_evidence_caps

    audio_only = GameplayClip(start=0, end=10, score=99,
                              features={"audio_event": 0.95})
    apply_evidence_caps([audio_only])
    assert audio_only.score == 86

    model_only = GameplayClip(start=0, end=10, score=99,
                              features={"vlm_viral": 0.95})
    apply_evidence_caps([model_only])
    assert model_only.score == 89

    two_signals = GameplayClip(start=0, end=10, score=99,
                               features={"reaction": 0.8, "excitement": 0.8})
    apply_evidence_caps([two_signals])
    assert two_signals.score == 99


def test_caption_exclude_mutes_cue_windows():
    tr = Transcript(words=_words("streamer talking double kill more talk"),
                    provider="whisper")
    # words at t=0,0.4,0.8,1.2,1.6,2.0 (d=0.34) — mute the span covering
    # words 3+4 only; any overlap with the span mutes a word.
    cs = captionize.build_caption_set(tr, 0.0, 3.0, "bold-pop",
                                      exclude=[(0.75, 1.55)])
    assert [w.text for w in cs.words] == ["streamer", "talking", "more", "talk"]


def test_game_noise_phrases_removed_from_captions():
    from app.models import CaptionWord
    words = [CaptionWord(t=i * 0.4, d=0.3, text=t) for i, t in
             enumerate("nice one Double Kill! let's go".split())]
    out = captionize.remove_phrases(words, captionize.game_noise("valorant"))
    assert [w.text for w in out] == ["nice", "one", "let's", "go"]
    # aliases + unknown profiles
    assert captionize.game_noise("fifa") == captionize.game_noise("eafc")
    assert captionize.game_noise("unknown") == frozenset()


def test_gameplay_hashtags_lead_with_the_game():
    from app.providers import hashtags
    tags = hashtags.suggest_hashtags("crazy round", content_type="gameplay",
                                     platform="tiktok", game="valorant")
    assert tags[0] == "#valorant" and tags[1] == "#valorantclips"
    assert "#tiktok" in tags and len(tags) <= 7


def test_ollama_model_autopick_prefers_strongest():
    from app.providers import llm
    assert llm._resolve_model(["llama3.2:latest", "qwen3:8b"]) == "qwen3:8b"
    assert llm._resolve_model(["gemma4:12b", "llama3.3:70b"]) == "gemma4:12b"
    assert llm._resolve_model(["qwen3:8b", "qwen3:14b"]) == "qwen3:14b"
    assert llm._resolve_model(["qwen2.5vl:7b", "qwen3:8b"]) == "qwen3:8b"
    assert llm._resolve_model(["llama3.2:latest"]) == "llama3.2:latest"
    assert llm._resolve_model(["some-custom:7b"]) == "some-custom:7b"  # anything > nothing
    assert llm._resolve_model([]) is None


def test_llm_title_strips_reasoning_blocks():
    from app.providers.llm import _clean_title
    assert _clean_title("<think>\nhmm what title\n</think>\nHe aced the round") \
        == "He aced the round"
    assert _clean_title("<think>truncated reasoning with no close tag") == ""
    assert _clean_title('"Title: My hook"') == "My hook"


def test_audio_url_extracted_from_soundboard_page():
    from app.game_packs import audio_url_from_html
    base = "https://www.myinstants.com/en/instant/valorant-kill/"
    # MyInstants-style relative path in an onclick handler
    html = """<html><button onclick="play('/media/sounds/valorant-kill.mp3', ...)">▶</button></html>"""
    assert (audio_url_from_html(html, base)
            == "https://www.myinstants.com/media/sounds/valorant-kill.mp3")
    # absolute URL wins
    html2 = '<a href="https://cdn.example.com/sfx/goal.mp3">download</a>'
    assert audio_url_from_html(html2, base) == "https://cdn.example.com/sfx/goal.mp3"
    # a page with no audio reference
    assert audio_url_from_html("<html><p>nothing here</p></html>", base) is None


def test_set_aspect_endpoint_validation():
    from starlette.testclient import TestClient
    from app import store
    from app.main import create_app

    store.init_db()
    c = TestClient(create_app(), raise_server_exceptions=False)
    r = c.post("/api/projects/proj_missing/aspect", json={"aspect": "21:9"})
    assert r.status_code == 400          # unknown aspect rejected first
    r = c.post("/api/projects/proj_missing/aspect", json={"aspect": "16:9"})
    assert r.status_code == 404          # then the project must exist


def test_ready_endpoint_is_lightweight():
    from starlette.testclient import TestClient
    from app.main import create_app

    c = TestClient(create_app(), raise_server_exceptions=False)
    r = c.get("/api/ready")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_clip_aspect_override_falls_back_to_project_dims():
    from app.models import ASPECTS
    st = ImportSettings(aspect="9:16")
    clip = Clip(start=0, end=5, aspect="16:9")
    assert (ASPECTS.get(clip.aspect or "") or st.dims()) == (1920, 1080)
    clip.aspect = None
    assert (ASPECTS.get(clip.aspect or "") or st.dims()) == (1080, 1920)


def test_llm_titles_safe_when_unavailable():
    from app.providers import llm
    old_available = llm.available
    try:
        llm.available = lambda: False
        assert llm.suggest_titles(["some transcript"], lang="en", budget=2.0) == {}
    finally:
        llm.available = old_available


# --------------------------------------------------------------------------- #
# OCR on-screen detection (pure helpers — no OCR backend needed)
# --------------------------------------------------------------------------- #
def test_ocr_keyword_matching_finds_viral_markers():
    from app.providers import detect_ocr as O
    # noisy OCR text still matches the marker as a normalized substring
    assert ("victory", "victory") in O.match_keywords("|| VICT0RY ||".replace("0", "o"), "valorant")
    assert any(l == "victory" for l, _ in O.match_keywords("SIEG - RUNDE GEWONNEN", "generic"))
    assert any(l == "kill" for l, _ in O.match_keywords("KRASSER KOPFSCHUSS", "generic"))
    assert any(l == "kill" for l, _ in O.match_keywords("ENEMY DOUBLE KILL", "cs2"))
    assert any(l == "goal" for l, _ in O.match_keywords("GOOOAL!! what a save", "rocketleague"))
    # word-boundary: "ko" must not fire inside "took"
    assert all(l != "eliminated" for l, _ in O.match_keywords("he took the lead", "generic"))
    # nothing viral -> no events
    assert O.match_keywords("loading please wait", "generic") == []
    # exact matching must never fire on clean unrelated text (fuzzy off here)
    assert O.match_keywords("the quick brown fox", "generic", fuzzy=False) == []


def test_ocr_menu_context_is_detected_but_not_highlighted():
    from app.media.ffmpeg import MediaInfo
    from app.providers import detect_gameplay as DG
    from app.providers import detect_ocr as O
    from app.providers.detect_ocr import OcrEvent

    assert any(l == "menu" for l, _ in O.match_keywords("MAIN MENU SETTINGS", "generic"))
    menu = OcrEvent(t=10.0, label="menu", text="main menu settings", confidence=0.9)
    audio = [types.SimpleNamespace(t=12.0, label="excited_shouting", confidence=0.8, detail="CLAP")]
    assert DG._suppress_audio_events_near_menu(audio, [menu]) == []

    info = MediaInfo(duration=120, width=1920, height=1080, fps=30,
                     has_audio=True, has_video=True, codec=None)
    assert DG._ocr_clips([menu], info, ImportSettings(min_len=8, max_len=20)) == []


def test_saved_visual_cues_extend_ocr_lexicon():
    from app import visual_cues
    from app.providers import detect_ocr as O

    visual_cues.add_visual_cue("valorant", "custom_killfeed", "One enemy remaining")
    assert ("custom_killfeed", "one enemy remaining") in O.match_keywords(
        "HUD: ONE ENEMY REMAINING", "valorant", fuzzy=False)
    assert not O.match_keywords("HUD: ONE ENEMY REMAINING", "cs2", fuzzy=False)


def test_visual_cue_regions_and_false_hits_are_persisted():
    from app import visual_cues
    from app.providers import detect_ocr as O

    game = "unit_visual_calibration"
    visual_cues.add_visual_cue(game, "killfeed", "Fake headshot")
    visual_cues.add_visual_region(game, "killfeed", {"x": 0.55, "y": 0.02, "w": 0.35, "h": 0.22})
    visual_cues.add_false_visual_cue(game, "killfeed", "Fake headshot")

    meta = visual_cues.list_visual_meta()[game]
    assert meta["phrases"]["killfeed"] == ["Fake headshot"]
    assert meta["regions"]["killfeed"][0]["x"] == 0.55
    assert visual_cues.is_false_positive(game, "killfeed", "HUD: FAKE HEADSHOT")
    assert O.match_keywords("HUD: FAKE HEADSHOT", game, fuzzy=False) == []


def test_saved_visual_regions_are_sampled_by_ocr_cropper():
    try:
        from PIL import Image
    except ImportError:
        return  # Pillow is optional (OCR ROI cropping); skip when absent in CI
    from app import visual_cues
    from app.providers import detect_ocr as O

    game = "unit_region_scan"
    visual_cues.add_visual_region(game, "scoreboard", {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.2})
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        frame = tmpd / "frame.png"
        Image.new("RGB", (640, 360), "black").save(frame)
        crops = O._ocr_frame_images(frame, tmpd, idx=1, profile=game)
    assert any(name.startswith("saved_scoreboard") for name, _ in crops)


def test_ocr_valorant_german_markers_and_easyocr_shapes():
    from app.providers import detect_ocr as O

    assert any(l == "kill" for l, _ in O.match_keywords("GEGNER ÜBRIG", "valorant"))
    assert any(l == "spike" for l, _ in O.match_keywords("SPIKE ENTSCHÄRFT", "valorant"))

    assert any(l == "bomb" for l, _ in O.match_keywords("BOMBE WURDE GELEGT", "cs2"))
    assert any(l == "win" for l, _ in O.match_keywords("TERRORISTEN GEWINNEN", "cs2"))
    assert any(l == "goal" for l, _ in O.match_keywords("TOR! ELFMETER VERWANDELT", "eafc"))

    class Reader:
        def readtext(self, _path, detail=1, paragraph=False):
            assert paragraph is False
            if detail == 1:
                return [([[0, 0]], "Spike platziert", 0.98),
                        ([[0, 1]], ("Headshot", 0.92))]
            return ["fallback"]

    text = O._easyocr_text(Reader(), "frame.png")
    assert "Spike platziert" in text and "Headshot" in text


def test_paddle_text_accepts_ocr_result_objects():
    from app.providers import detect_ocr as O

    class ResultObject:
        rec_texts = ["Sieg", "Kopfschuss"]

    class Reader:
        def predict(self, _path):
            return [ResultObject(), {"rec_texts": ["Bombe gelegt"]}]

    text = O._paddle_text(Reader(), "frame.png")
    assert "Sieg" in text and "Kopfschuss" in text and "Bombe gelegt" in text


def test_ocr_reader_falls_back_to_easyocr_when_paddle_fails():
    from app.providers import detect_ocr as O

    old_reader = O._reader
    old_make_paddle = O._make_paddle
    old_make_easyocr = O._make_easyocr

    class EasyReader:
        pass

    try:
        O._reader = None

        def fail_paddle(_gpu):
            raise RuntimeError("broken paddle runtime")

        O._make_paddle = fail_paddle
        O._make_easyocr = lambda _gpu: EasyReader()
        kind, reader = O._get_reader("paddleocr")
        assert kind == "easyocr"
        assert isinstance(reader, EasyReader)
    finally:
        O._reader = old_reader
        O._make_paddle = old_make_paddle
        O._make_easyocr = old_make_easyocr


def test_ocr_fuzzy_matches_garbled_text():
    from app.providers import detect_ocr as O
    try:
        import rapidfuzz  # noqa: F401
    except Exception:
        return  # graceful: fuzzy path is a no-op without rapidfuzz
    # OCR mangles stylized game fonts; fuzzy still resolves the marker.
    assert any(l == "eliminated" for l, _ in O.match_keywords("YOU WERE ELiMlNATED", "generic"))
    assert any(l == "kill" for l, _ in O.match_keywords("DOUBLE KlLL", "cs2"))
    assert any(l == "kill" for l, _ in O.match_keywords("HEADSH0T", "cs2"))
    # but a high threshold must not invent markers out of unrelated prose
    assert O.match_keywords("the quick brown fox jumps", "generic", threshold=95) == []


def test_ocr_frame_sampling_is_bounded_and_spaced():
    from app.providers import detect_ocr as O
    assert O.sample_frame_times(0) == []
    ts = O.sample_frame_times(20, every=2.0)
    assert ts and all(0 < t < 20 for t in ts)
    assert all(b - a >= 1.9 for a, b in zip(ts, ts[1:]))
    # a long VOD never exceeds the frame cap
    assert len(O.sample_frame_times(100000, every=2.0, max_frames=400)) <= 400
    focused = O.focused_frame_times(3600, [100.0, 103.0], max_frames=12)
    assert any(99.0 <= t <= 104.5 for t in focused)
    assert len(focused) <= 12


def test_ocr_dedupes_persisting_banner():
    from app.providers.detect_ocr import OcrEvent, dedupe_events
    evs = [OcrEvent(t=10.0, label="victory", text="victory", confidence=0.8),
           OcrEvent(t=11.0, label="victory", text="victory", confidence=0.8),  # same banner
           OcrEvent(t=12.0, label="victory", text="victory", confidence=0.8),
           OcrEvent(t=40.0, label="victory", text="victory", confidence=0.8)]  # later round
    out = dedupe_events(evs, min_gap=4.0)
    assert [round(e.t) for e in out] == [10, 40]


def test_ocr_clips_open_before_the_banner():
    from app.media.ffmpeg import MediaInfo
    from app.providers.detect_ocr import OcrEvent
    from app.providers.detect_gameplay import _ocr_clips
    info = MediaInfo(duration=600, width=1920, height=1080, fps=30,
                     has_audio=True, has_video=True, codec=None)
    st = ImportSettings(min_len=15, max_len=45)
    clips = _ocr_clips([OcrEvent(t=120.0, label="victory", text="victory", confidence=0.9)],
                       info, st)
    assert len(clips) == 1
    c = clips[0]
    assert c.start < 120.0 <= c.end           # the banner moment is inside the clip
    assert st.min_len <= c.duration <= st.max_len + 0.01
    assert c.features["ocr"] == 0.9
    assert any("on-screen" in f.detail.lower() for f in c.factors)


# --------------------------------------------------------------------------- #
# LLM virality re-rank (pure parse + boost — no Ollama needed)
# --------------------------------------------------------------------------- #
def test_llm_viral_parse():
    from app.providers.llm import _parse_viral
    assert _parse_viral("SCORE: 82 | REASON: strong hook")[0] == 0.82
    assert _parse_viral("SCORE: 82 | REASON: strong hook")[1] == "strong hook"
    assert _parse_viral("score: 5") == (0.05, "AI virality read")
    assert _parse_viral("no number here") is None
    assert _parse_viral("SCORE: 250")[0] == 1.0     # clamped


def test_viral_boost_is_bounded_and_explainable():
    from app.models import ScoreFactor
    from app.providers.score import apply_viral_boost
    base = [ScoreFactor(label="hook", weight=20.0)]
    hi, fh = apply_viral_boost(50, base, 1.0, "great hook", max_swing=12)
    lo, fl = apply_viral_boost(50, base, 0.0, "weak", max_swing=12)
    mid, fm = apply_viral_boost(50, base, 0.5, "meh", max_swing=12)
    assert hi == 62 and lo == 38 and mid == 50      # ±max_swing, centred at 0.5
    assert fh[0].label.startswith("AI:") and fh[0].weight == 12.0
    assert fm == base                                # no swing -> no added factor
    # never escapes 1..99
    assert apply_viral_boost(95, base, 1.0, "x", max_swing=12)[0] == 99


def test_ocr_capability_detection_default_off():
    # CI has no OCR backend installed -> capability is off, detection no-ops.
    assert _settings().has_ocr is False
    assert _settings().capability_report()["ocr"] is False


# --------------------------------------------------------------------------- #
# Per-speaker captions + the no-linger caption fix
# --------------------------------------------------------------------------- #
def _multi_speaker_transcript():
    # speaker 0 then speaker 1, alternating
    ws = [Word(t=0.0, d=0.3, text="hello", speaker=0),
          Word(t=0.4, d=0.3, text="there", speaker=0),
          Word(t=0.8, d=0.3, text="hi", speaker=1),
          Word(t=1.2, d=0.3, text="back", speaker=1)]
    return Transcript(words=ws, provider="whisper", speakers=2)


def test_caption_speaker_filter_keeps_only_chosen():
    tr = _multi_speaker_transcript()
    both = captionize.build_caption_set(tr, 0.0, 2.0, "bold-pop")
    assert [w.text for w in both.words] == ["hello", "there", "hi", "back"]
    assert [w.speaker for w in both.words] == [0, 0, 1, 1]
    only0 = captionize.build_caption_set(tr, 0.0, 2.0, "bold-pop", speakers={0})
    assert [w.text for w in only0.words] == ["hello", "there"]
    only1 = captionize.build_caption_set(tr, 0.0, 2.0, "bold-pop", speakers={1})
    assert [w.text for w in only1.words] == ["hi", "back"]


def test_speakers_in_lists_present_speakers():
    tr = _multi_speaker_transcript()
    assert captionize.speakers_in(tr, 0.0, 2.0) == [0, 1]
    assert captionize.speakers_in(tr, 0.0, 0.6) == [0]   # only speaker 0 talks early


def test_caption_does_not_linger_through_silence():
    from app.models import CaptionSet, CaptionWord
    # word at t=0, then a 3s gap, then a word — within one line the first word
    # must NOT hold for 3s; it clears shortly after it's spoken.
    cs = CaptionSet(words=[CaptionWord(t=0.0, d=0.3, text="alpha"),
                           CaptionWord(t=3.3, d=0.3, text="beta")],
                    max_words_per_line=3)
    ass = C.build_ass(cs, get_style("bold-pop"), 1080, 1920)
    first = next(l for l in ass.splitlines() if l.startswith("Dialogue:") and "ALPHA" in l)
    # End timestamp (field 3) of the alpha event should be ~0.7s, not ~3.3s.
    end_ts = first.split(",", 3)[2]
    assert end_ts < "0:00:01.50", f"caption lingered through silence: {end_ts}"


def test_common_cue_pack_is_available_for_all_games():
    from app import game_packs
    status = game_packs.pack_status()
    assert "common" in status
    assert status["common"]["label"].lower().startswith(("common", "allgemein"))
    assert {e["name"] for e in status["common"]["events"]} >= {"airhorn", "hype", "laugh"}


# --------------------------------------------------------------------------- #
# Caption precision, hook front-loading, multimodal corroboration, cue learning
# --------------------------------------------------------------------------- #
def test_caption_line_breaks_on_speech_pause():
    from app.models import CaptionWord
    from app.pipeline.captions import _group_lines
    # a big gap before the third word forces a new line even under the count cap
    ws = [CaptionWord(t=0.0, d=0.3, text="a"), CaptionWord(t=0.4, d=0.3, text="b"),
          CaptionWord(t=3.0, d=0.3, text="c")]
    assert [len(l) for l in _group_lines(ws, 5)] == [2, 1]
    # tight speech (no gaps) stays one line up to the count cap
    ws2 = [CaptionWord(t=i * 0.4, d=0.3, text=str(i)) for i in range(3)]
    assert len(_group_lines(ws2, 5)) == 1


def test_hook_rewards_front_loaded_curiosity():
    early = _words("Why does nobody do this one secret")        # hook word at t=0
    late = [Word(t=0.0, d=0.3, text="um"), Word(t=0.6, d=0.3, text="okay"),
            Word(t=1.2, d=0.3, text="anyway")]                  # no hook up front
    he, _ = signals.hook_strength(early, signals.get_lexicon("en"))
    hl, _ = signals.hook_strength(late, signals.get_lexicon("en"))
    assert he > hl


def test_corroboration_boosts_multi_signal_clips():
    from app.providers.detect_cues import CueEvent
    from app.providers.detect_gameplay import GameplayClip, apply_corroboration
    from app.providers.detect_ocr import OcrEvent
    c = GameplayClip(start=10, end=30, score=70, peak_t=20)
    apply_corroboration([c], [CueEvent(t=20, label="kill", similarity=0.9)],
                        [OcrEvent(t=21, label="kill", text="kill", confidence=0.8)])
    assert c.score > 70 and c.features.get("corroborated") == 1.0
    assert any("confirm" in f.label.lower() for f in c.factors)
    # a single source does not corroborate
    c2 = GameplayClip(start=10, end=30, score=70, peak_t=20)
    apply_corroboration([c2], [CueEvent(t=20, label="kill", similarity=0.9)], [])
    assert c2.score == 70


def test_gameplay_title_fallback_names_reaction_peaks():
    from app.providers.detect_gameplay import GameplayClip, apply_title_fallbacks

    c = GameplayClip(start=2300, end=2335, score=95, peak_t=2325,
                     title="Highlight — 38:45",
                     features={"reaction": 1.0, "intensity": 0.99})
    apply_title_fallbacks([c])
    assert c.title == "Big Reaction — 38:45"


def test_custom_event_padding_and_clap_clips():
    from app.media.ffmpeg import MediaInfo
    from app.providers.audio_events import AudioEvent
    from app.providers.detect_gameplay import (_audio_event_clips, _timing_window,
                                               get_profile)
    info = MediaInfo(duration=200, width=1920, height=1080, fps=30,
                     has_audio=True, has_video=True, codec=None)
    st = ImportSettings(min_len=10, max_len=40, lead_seconds=7, tail_seconds=9)
    lead, tail = _timing_window(st, get_profile("generic"))
    assert (lead, tail) == (7.0, 9.0)
    clips = _audio_event_clips(
        [AudioEvent(t=80, label="crowd_cheering", confidence=0.8,
                    detail="CLAP heard crowd cheering")],
        info, st, lead=lead, tail=tail)
    assert clips and clips[0].start == 73 and clips[0].end == 89
    assert clips[0].features["audio_event"] == 0.8


def test_manual_clip_context_rejects_nonfinite_values():
    from app.api.routes_projects import _clamp_pad

    assert _clamp_pad(None) is None
    assert _clamp_pad(float("nan")) is None
    assert _clamp_pad(float("inf")) is None
    assert _clamp_pad(-5) == 0.0
    assert _clamp_pad(999) == 60.0


def test_cue_learning_pending_labels():
    from app.cue_learning import pending_labels
    from app.providers.detect_ocr import OcrEvent
    evs = [OcrEvent(t=5, label="kill", text="kill", confidence=0.8),
           OcrEvent(t=9, label="kill", text="kill", confidence=0.8),       # dup label
           OcrEvent(t=12, label="victory", text="victory", confidence=0.8)]
    assert pending_labels(evs, set()) == ["kill", "victory"]      # one per label, time order
    assert pending_labels(evs, {"kill"}) == ["victory"]           # skip already-saved


# --------------------------------------------------------------------------- #
# Power-ups: VAD caption snapping, replay scoring, emotion blend, subject pick
# --------------------------------------------------------------------------- #
def test_vad_refine_clamps_and_drops_silent_words():
    from app.providers.vad import refine_words
    ws = [Word(t=0.0, d=2.0, text="hello"),      # overruns into silence -> clamp
          Word(t=5.0, d=0.3, text="ghost"),      # entirely in a silent gap -> drop
          Word(t=10.2, d=0.3, text="world")]
    speech = [(0.0, 1.0), (10.0, 11.0)]
    out = refine_words(ws, speech)
    assert [w.text for w in out] == ["hello", "world"]
    assert out[0].end <= 1.1                      # clamped to the speech end (+pad)
    # no speech info -> unchanged
    assert refine_words(ws, []) == ws


def test_replay_value_rewards_clean_button():
    looped = _words("here is the one secret that changes everything")
    looped[-1].text = "everything."                # complete, strong, concise
    dangling = _words("and then i was going to say something but")
    hi, _ = signals.replay_value(looped, 20.0, signals.get_lexicon("en"))
    lo, _ = signals.replay_value(dangling, 20.0, signals.get_lexicon("en"))
    assert hi > lo


def test_apply_replay_bonus_only_lifts():
    from app.models import ScoreFactor
    from app.providers.score import apply_replay_bonus
    words = _words("this is the biggest secret ever")
    words[-1].text = "ever."
    s, f = apply_replay_bonus(60, [ScoreFactor(label="x", weight=1.0)], words, 18.0, lang="en")
    assert s >= 60 and 1 <= s <= 99
    # a long clip trailing off on a connective gets no bonus (never a penalty)
    d = _words("then i was about to say and")  # ends on dangling "and", not concise
    s2, f2 = apply_replay_bonus(60, [], d, 50.0, lang="en")
    assert s2 == 60


def test_emotion_excitement_blend_is_bounded():
    from app.providers.emotion import _arousal_from_result, apply_excitement_bonus
    hi, fh = apply_excitement_bonus(50, [], 1.0)
    lo, _ = apply_excitement_bonus(50, [], 0.0)
    assert hi > 50 and lo < 50 and 1 <= hi <= 99 and 1 <= lo <= 99
    assert fh and "energy" in fh[0].label.lower()
    # result parsing sums high-arousal label probabilities
    res = [{"labels": ["生气/angry", "中立/neutral"], "scores": [0.7, 0.3]}]
    assert abs(_arousal_from_result(res) - 0.7) < 1e-6


def test_subject_center_prefers_people_then_largest():
    from app.providers.subject import _center_from_boxes
    # a small person box vs a big non-person box -> follow the person
    boxes = [(True, 100, 200, 5000), (False, 800, 1000, 50000)]
    assert abs(_center_from_boxes(boxes, 1000) - 0.15) < 1e-6
    # no people -> largest object
    boxes2 = [(False, 0, 100, 1000), (False, 800, 1000, 9000)]
    assert abs(_center_from_boxes(boxes2, 1000) - 0.9) < 1e-6
    assert _center_from_boxes([], 1000) is None


def test_optional_powerups_off_by_default_in_ci():
    s = _settings()
    r = s.capability_report()
    assert r["vad"] is False and r["emotion"] is False
    assert r["scene_detect"] is False and r["active_speaker"] is False
    assert r["denoise"] is False and r["audio_events"] is False
    assert r["reframe_engine"] in ("haar", "yolo", "mediapipe")


def test_active_speaker_not_advertised_until_adapter_exists():
    from app import config

    old_env = os.environ.pop("CLIPFORGE_ASD_DIR", None)
    try:
        assert config._detect_asd_adapter() is False
    finally:
        if old_env is not None:
            os.environ["CLIPFORGE_ASD_DIR"] = old_env


def test_optional_nested_module_lookup_is_safe_when_parent_missing():
    from app import config

    assert config._has_module("definitely_missing_parent.child") is False


def test_active_speaker_adapter_detects_valid_checkout_fixture():
    from app import config

    old_env = os.environ.get("CLIPFORGE_ASD_DIR")
    old_has_module = config._has_module
    old_cuda = config._torch_cuda_available
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "LR-ASD")
        os.makedirs(os.path.join(root, "model"), exist_ok=True)
        os.makedirs(os.path.join(root, "weight"), exist_ok=True)
        os.makedirs(os.path.join(root, "model", "faceDetector", "s3fd"), exist_ok=True)
        for rel in ("ASD.py", "Columbia_test.py", os.path.join("model", "Model.py")):
            with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
                f.write("# fixture\n")
        with open(os.path.join(root, "weight", "pretrain_AVA.model"), "wb") as f:
            f.write(b"x" * 100_001)
        with open(os.path.join(root, "model", "faceDetector", "s3fd", "sfd_face.pth"), "wb") as f:
            f.write(b"x" * 100_001)
        os.environ["CLIPFORGE_ASD_DIR"] = root
        config._has_module = lambda _name: True
        config._torch_cuda_available = lambda: True
        try:
            assert config._detect_asd_adapter() is True
        finally:
            config._has_module = old_has_module
            config._torch_cuda_available = old_cuda
            if old_env is None:
                os.environ.pop("CLIPFORGE_ASD_DIR", None)
            else:
                os.environ["CLIPFORGE_ASD_DIR"] = old_env


def test_lr_asd_script_compatibility_accepts_patched_scenedetect_fallback():
    from pathlib import Path
    from app import config

    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "LR-ASD")
        os.makedirs(root, exist_ok=True)
        script = os.path.join(root, "Columbia_test.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write("from scenedetect.video_manager import VideoManager\n")
        assert config._lr_asd_script_compatible(Path(root)) is False
        with open(script, "w", encoding="utf-8") as f:
            f.write("from scenedetect.video_manager import VideoManager\n"
                    "from scenedetect import open_video\n"
                    "VideoManager = None\n")
        assert config._lr_asd_script_compatible(Path(root)) is True


def test_lr_asd_parser_selects_highest_speaking_track():
    from app.providers import active_speaker as AS

    tracks = [
        {"track": {"frame": [0, 1, 2], "bbox": [[100, 0, 200, 100],
                                                [110, 0, 210, 100],
                                                [120, 0, 220, 100]]}},
        {"track": {"frame": [0, 1, 2], "bbox": [[700, 0, 800, 100],
                                                [710, 0, 810, 100],
                                                [720, 0, 820, 100]]}},
    ]
    centers = AS._centers_from_tracks(
        tracks, [[-0.2, 0.8, 0.7], [1.0, -0.1, -0.2]], frame_width=1000, fps=25.0)
    assert centers == [(0.0, 0.75), (0.04, 0.16), (0.08, 0.17)]
    assert AS._centers_from_tracks(tracks, [[-1, -1, -1], [-0.5, -0.2, -0.1]], 1000) is None


def test_reframe_uses_lr_asd_centers_before_face_fallback():
    from app.pipeline import reframe as RF
    from app.providers import active_speaker as AS

    old_settings, old_track = RF.get_settings, AS.track_centers
    RF.get_settings = lambda: _settings(has_asd=True, has_opencv=False)
    AS.track_centers = lambda *_args: [(0.0, 0.8), (0.4, 0.82), (0.8, 0.84)]
    try:
        rf = RF.compute_reframe("unused.mp4", 10.0, 11.0, 16 / 9)
    finally:
        RF.get_settings, AS.track_centers = old_settings, old_track
    assert rf.tracked is True
    assert rf.keyframes[0].cx > 0.7


def test_audio_event_reduce_and_bonus():
    from app.models import ScoreFactor
    from app.providers import audio_events as AE
    # a strong cheer + a weak laugh -> combined > the cheer alone, top reason cheer
    res = AE.reduce_scores({"Cheering": 0.8, "Laughter": 0.3, "Speech": 0.9})
    assert res is not None
    hype, reason = res
    assert reason == "crowd cheering" and hype > 0.8
    # nothing viral in the tags -> no signal
    assert AE.reduce_scores({"Speech": 0.95, "Silence": 0.4}) is None
    # bonus is positive-only, bounded, and explainable
    base = [ScoreFactor(label="hook", weight=10.0)]
    ns, nf = AE.apply_event_bonus(60, base, 1.0, "crowd cheering", max_bonus=10.0)
    assert ns == 70 and nf[0].weight == 10.0 and "cheering" in nf[0].label.lower()
    assert AE.apply_event_bonus(60, base, 0.0, "x") == (60, base)  # quiet -> no-op
    clap = AE.reduce_clap_similarities({"crowd cheering": 0.42, "impact": 0.12})
    assert clap is not None and clap[1] == "crowd cheering" and clap[0] > 0.5
    assert AE.reduce_clap_similarities({"crowd cheering": 0.1}) is None
    assert AE.reduce_clap_window(
        {"crowd cheering": 0.42}, {"lobby music": 0.48}) is None
    assert AE.reduce_clap_window(
        {"crowd cheering": 0.34}, {"lobby music": 0.31}) is None
    gated = AE.reduce_clap_window({"crowd cheering": 0.42}, {"lobby music": 0.24})
    assert gated is not None and gated[1] == "crowd cheering"


def test_clap_prompts_include_game_and_german_context():
    from app.providers import audio_events as AE

    pos, neg = AE._prompt_sets("valorant", "de")
    joined_pos = " ".join(p for prompts in pos.values() for p in prompts).lower()
    joined_neg = " ".join(p for prompts in neg.values() for p in prompts).lower()
    assert "valorant" in joined_pos and "spike" in joined_pos
    assert "german streamer" in joined_pos
    assert "lobby" in joined_neg and "menu" in joined_neg

    pos2, neg2 = AE._prompt_sets(
        "generic", "de",
        positive_prompts=["German ace celebration"],
        negative_prompts=["inventory click loop"],
    )
    assert "German ace celebration" in pos2["custom cue"]
    assert "inventory click loop" in neg2["custom non-highlight"]


def test_clap_loader_hides_server_args_and_restores_argv():
    from app.providers import audio_events as AE

    old_clap = AE._clap
    old_mod = sys.modules.get("laion_clap")
    old_argv = sys.argv[:]
    calls = []
    mod = types.ModuleType("laion_clap")
    import importlib.machinery
    mod.__spec__ = importlib.machinery.ModuleSpec("laion_clap", loader=None)

    class FakeClap:
        def __init__(self, **_kwargs):
            calls.append(sys.argv[:])

        def load_ckpt(self):
            pass

    mod.CLAP_Module = FakeClap
    sys.modules["laion_clap"] = mod
    AE._clap = None
    sys.argv = ["uvicorn", "app.main:app", "--port", "8000"]
    try:
        assert AE._load_clap() is not None
        assert calls and all(call == ["uvicorn"] for call in calls)
        assert sys.argv == ["uvicorn", "app.main:app", "--port", "8000"]
    finally:
        AE._clap = old_clap
        sys.argv = old_argv
        if old_mod is None:
            sys.modules.pop("laion_clap", None)
        else:
            sys.modules["laion_clap"] = old_mod


def test_clap_loader_uses_nonstrict_checkpoint_fallback():
    from app.providers import audio_events as AE

    class Core:
        def __init__(self):
            self.strict_values = []

        def load_state_dict(self, _state, *args, **kwargs):
            strict = kwargs.get("strict", args[0] if args else True)
            self.strict_values.append(strict)
            if strict:
                raise RuntimeError(
                    'Unexpected key(s) in state_dict: "text_branch.embeddings.position_ids".'
                )

    class FakeClap:
        def __init__(self):
            self.model = Core()

        def load_ckpt(self):
            self.model.load_state_dict({"ok": 1})

    model = FakeClap()
    AE._load_clap_checkpoint(model)
    assert model.model.strict_values == [True, False]


def test_audio_capability_flags_drop_failed_runtime_detectors():
    from app.providers import audio_events as AE

    old_get, old_tagger, old_clap = AE.get_settings, AE._tagger, AE._clap
    AE.get_settings = lambda: _settings(has_audio_events=True, has_clap=True)
    try:
        AE._tagger = None
        AE._clap = False
        assert AE.capability_flags() == {
            "audio_events": True, "panns_audio": True, "clap_audio": False}
        AE._tagger = False
        assert AE.available() is False
    finally:
        AE.get_settings = old_get
        AE._tagger = old_tagger
        AE._clap = old_clap


def test_clap_embedding_array_supports_old_signature():
    from app.providers import audio_events as AE

    calls = []

    def old_signature(values):
        calls.append(values)
        return [[1.0, 0.0]]

    assert AE._embedding_array(old_signature, ["a"]).shape == (1, 2)
    assert calls == [["a"]]


def test_caption_emphasis_and_emoji():
    from app.models import CaptionWord
    from app.pipeline.caption_fx import annotate
    # "secret" (hook/payoff) and "100" (number) are power words; "the"/"is" aren't.
    ws = [CaptionWord(t=0, d=0.3, text="the"),
          CaptionWord(t=0.3, d=0.3, text="secret"),
          CaptionWord(t=0.6, d=0.3, text="is"),
          CaptionWord(t=0.9, d=0.3, text="100")]
    out = annotate(ws, lang="en", emphasis=True, emoji=False, max_words_per_line=4)
    flags = {w.text: w.emphasis for w in out}
    assert flags["secret"] and flags["100"]
    assert not flags["the"] and not flags["is"]
    # input is not mutated (pure)
    assert all(w.emphasis is False for w in ws)
    # emphasis is capped per on-screen line so a whole line never lights up
    many = [CaptionWord(t=i, d=0.3, text="secret") for i in range(3)]
    capped = annotate(many, lang="en", emphasis=True, max_words_per_line=3,
                      max_emphasis_per_line=2)
    assert sum(w.emphasis for w in capped) == 2
    # emoji appended only when enabled, and only to mapped power words
    emo = annotate([CaptionWord(t=0, d=0.3, text="fire")], lang="en", emoji=True)
    assert emo[0].emoji == "🔥"
    assert annotate([CaptionWord(t=0, d=0.3, text="fire")], lang="en", emoji=False)[0].emoji is None
    # both off → returns the list unchanged
    assert annotate(ws, emphasis=False, emoji=False) is ws


def test_caption_emphasis_renders_in_ass():
    from app.models import CaptionSet, CaptionWord, StyleTemplate
    style = StyleTemplate(id="t", name="t", highlight="00FF00", emphasis=True)
    cs = CaptionSet(words=[CaptionWord(t=0, d=0.4, text="the"),
                           CaptionWord(t=1.0, d=0.4, text="secret")],
                    lang="en")
    ass = C.build_ass(cs, style, 1080, 1920)
    # the emphasised power word carries the highlight colour scale tag
    assert "fscx106" in ass
    # opting the style out removes the standalone emphasis
    plain = C.build_ass(cs, style.model_copy(update={"emphasis": False}), 1080, 1920)
    assert "fscx106" not in plain


def test_vlm_keyframe_times_are_inside_span():
    from app.providers.vlm import keyframe_times
    ts = keyframe_times(10.0, 22.0, n=3)
    assert len(ts) == 3 and all(10.0 < t < 22.0 for t in ts)
    assert ts == sorted(ts)
    assert keyframe_times(5.0, 5.0) == [5.0]  # zero-length span is safe


def test_vlm_model_autopick_accepts_hyphenated_qwen_name():
    from app.providers import vlm
    assert vlm._resolve_model(["qwen3-vl:8b", "qwen2.5vl:32b"]) == "qwen3-vl:8b"
    assert vlm._resolve_model(["qwen2.5-vl:7b", "llava:latest"]) == "qwen2.5-vl:7b"
    assert vlm._resolve_model(["qwen2.5vl:7b", "qwen2.5vl:32b"]) == "qwen2.5vl:32b"


def test_vlm_negative_reason_caps_score():
    from app.providers import vlm
    assert vlm._parse("SCORE: 88 | REASON: loading screen")[0] <= 0.35
    assert "Bewerte" in vlm._prompt_for("de")


def test_orchestrator_passes_source_path_to_vlm_scorer():
    from app.pipeline import orchestrator as O
    from app.providers import vlm

    old_available, old_score_visuals = vlm.available, vlm.score_visuals
    calls = []

    def fake_score_visuals(src_path, spans, *, budget=45.0, max_workers=2,
                           n_frames=3, timeout=30.0, lang="en"):
        calls.append((src_path, spans, budget, max_workers, n_frames, timeout, lang))
        return {0: (0.8, "strong frames")}

    try:
        vlm.available = lambda: True
        vlm.score_visuals = fake_score_visuals
        out = O._score_visual_reads("source.mp4", [Clip(start=1.0, end=2.5)],
                                    lang="de")
    finally:
        vlm.available = old_available
        vlm.score_visuals = old_score_visuals

    assert out == {0: (0.8, "strong frames")}
    assert calls == [("source.mp4", [(1.0, 2.5)], 45.0, 1, 2, 30.0, "de")]


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

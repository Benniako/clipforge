"""Transcription provider — produces a word-timed transcript from audio.

Three engines, in descending order of quality, selected by availability and the
``CLIPFORGE_TRANSCRIBER`` preference (see config):

1. **whisperX** — Whisper + forced word-level alignment (tighter caption timing)
   and optional speaker **diarization** (real speaker labels). Best, heaviest.
2. **faster-whisper** — solid word-level timestamps, single speaker.
3. **synthetic** — deterministic filler so the loop still runs with no ASR.

The chosen engine cascades to the next on failure, and the active path is
recorded on ``Transcript.provider`` and surfaced in the UI — the fallback is
never silent.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg
from ..models import Transcript, Word

log = logging.getLogger("clipforge.transcribe")

# Whisper models aren't guaranteed thread-safe and saturate the device anyway —
# serialize transcriptions so multiple pipeline workers can't overlap them.
_asr_lock = threading.Lock()

_model = None       # lazily-loaded faster-whisper WhisperModel
_wx_model = None    # lazily-loaded whisperX ASR model
_wx_align: dict = {}  # language_code -> (align_model, metadata)
_wx_diarize = None  # lazily-loaded diarization pipeline


def _ensure_ffmpeg_on_path() -> None:
    """whisperX shells out to a bare ``ffmpeg``; make our static binary findable."""
    ff = get_settings().ffmpeg
    if ff:
        d = os.path.dirname(ff)
        if d and d not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _ensure_cuda_dlls() -> None:
    """On Windows, make pip-installed NVIDIA runtime DLLs loadable.

    ctranslate2 detects CUDA via the driver, but executing a model needs
    cuBLAS/cuDNN. The ``nvidia-cublas-cu12``/``nvidia-cudnn-cu12`` wheels (and
    matching cu13 wheels when installed) ship
    them into ``site-packages/nvidia/*/bin`` — outside any default DLL search
    path — so without this hook GPU transcription dies with
    "Library cublas64_12.dll is not found" and the pipeline silently degrades
    to the synthetic transcript. Best-effort no-op everywhere else.
    """
    if os.name != "nt":
        return
    try:
        import nvidia  # namespace package owned by the nvidia-*-cu12/cu13 wheels
    except Exception:
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            d = str(bin_dir)
            # ctranslate2 *delay-loads* cuBLAS/cuDNN via plain LoadLibrary,
            # which ignores add_dll_directory and falls back to PATH — so we
            # need both.
            if d not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(d)
            except Exception:
                pass


def _load_whisper():
    global _model
    if _model is not None:
        return _model
    from faster_whisper import WhisperModel

    _ensure_cuda_dlls()
    s = get_settings()
    ct = "float16" if s.device == "cuda" else "int8"
    # Use every CPU core on the CPU path (ignored on GPU) so transcription
    # isn't artificially single-threaded — the slowest stage should use the
    # whole machine.
    cpu_threads = os.cpu_count() or 4
    log.info("loading faster-whisper model %s (%s/%s, cpu_threads=%d)",
             s.whisper_model, s.device, ct, cpu_threads)
    _model = WhisperModel(s.whisper_model, device=s.device, compute_type=ct,
                          cpu_threads=cpu_threads)
    return _model


_batched = None  # cached BatchedInferencePipeline wrapper (or False if N/A)


def _batched_pipeline(model, batch_size: int | None = None):
    """A faster-whisper BatchedInferencePipeline on GPU (keeps the card
    saturated → ~1.8x faster), or None when unavailable / on CPU."""
    global _batched
    if _batched is not None:
        return _batched or None
    s = get_settings()
    if s.device != "cuda" or (batch_size or s.whisper_batch_size) <= 0:
        _batched = False
        return None
    try:
        from faster_whisper import BatchedInferencePipeline

        _batched = BatchedInferencePipeline(model=model)
        log.info("faster-whisper batched inference on (batch_size=%d)",
                 batch_size or s.whisper_batch_size)
    except Exception as e:
        log.info("batched inference unavailable (%s); sequential", e)
        _batched = False
    return _batched or None


def _initial_prompt(language: str | None) -> str | None:
    if not (language or "").lower().startswith("de"):
        return None
    prompt = (get_settings().german_gaming_prompt or "").strip()
    return prompt or None


def _transcribe_with_prompt(engine, audio, *, initial_prompt: str | None = None,
                            **kwargs):
    if not initial_prompt:
        return engine.transcribe(audio, **kwargs)
    try:
        return engine.transcribe(audio, initial_prompt=initial_prompt, **kwargs)
    except TypeError as e:
        if "initial_prompt" not in str(e):
            raise
        return engine.transcribe(audio, **kwargs)


def transcribe(audio_path: str, *, language: str | None = None,
               progress=None, power_mode: str | None = None) -> Transcript:
    """Transcribe ``audio_path`` to a word-timed :class:`Transcript`."""
    s = get_settings()
    batch_size = s.whisper_batch_for(power_mode)
    lang = None if (language in (None, "auto", "")) else language
    engine = s.transcription_engine

    if engine == "whisperx":
        try:
            with _asr_lock:
                return _whisperx_transcribe(audio_path, lang, progress, batch_size)
        except Exception as e:
            log.warning("whisperX failed (%s); falling back", e)
            engine = "whisper" if s.has_whisper else "synthetic"

    if engine == "whisper":
        try:
            with _asr_lock:
                return _whisper_transcribe(audio_path, lang, progress, batch_size)
        except Exception as e:  # model download blocked, etc. — degrade.
            log.warning("whisper failed (%s); using synthetic transcript", e)

    return synthetic_transcript(audio_path, lang=lang)


# --------------------------------------------------------------------------- #
# whisperX: alignment + optional diarization
# --------------------------------------------------------------------------- #
def _load_whisperx_model():
    global _wx_model
    if _wx_model is not None:
        return _wx_model
    import whisperx

    _ensure_cuda_dlls()
    s = get_settings()
    ct = "float16" if s.device == "cuda" else "int8"
    log.info("loading whisperX model %s (%s/%s)", s.whisper_model, s.device, ct)
    _wx_model = whisperx.load_model(s.whisper_model, s.device, compute_type=ct)
    return _wx_model


def _diarization_pipeline():
    """Lazily build the pyannote diarization pipeline (needs an HF token)."""
    global _wx_diarize
    if _wx_diarize is not None:
        return _wx_diarize
    s = get_settings()
    if not s.hf_token:
        return None
    try:  # location moved across whisperX versions
        from whisperx.diarize import DiarizationPipeline
    except Exception:
        from whisperx import DiarizationPipeline  # type: ignore
    try:
        _wx_diarize = DiarizationPipeline(
            model_name=s.diarization_model,
            token=s.hf_token,
            device=s.device,
        )
    except TypeError:
        _wx_diarize = DiarizationPipeline(
            use_auth_token=s.hf_token,
            device=s.device,
        )
    return _wx_diarize


def _whisperx_transcribe(audio_path, language, progress, batch_size: int) -> Transcript:
    import whisperx

    _ensure_ffmpeg_on_path()
    s = get_settings()
    audio = whisperx.load_audio(audio_path)

    model = _load_whisperx_model()
    result = _transcribe_with_prompt(
        model, audio, batch_size=max(batch_size, 1), language=language,
        initial_prompt=_initial_prompt(language))
    lang = result.get("language", language or "en") or "en"
    if progress:
        progress(0.5)

    # Forced alignment -> precise per-word timings.
    try:
        if lang not in _wx_align:
            _wx_align[lang] = whisperx.load_align_model(language_code=lang, device=s.device)
        amodel, meta = _wx_align[lang]
        result = whisperx.align(result["segments"], amodel, meta, audio, s.device,
                                return_char_alignments=False)
    except Exception as e:
        log.warning("whisperX alignment unavailable for %s (%s); using segment timings", lang, e)
    if progress:
        progress(0.85)

    # Optional diarization -> speaker labels.
    speaker_ids: dict[str, int] = {}
    diar = _diarization_pipeline()
    if diar is not None:
        try:
            diar_segments = diar(audio)
            result = whisperx.assign_word_speakers(diar_segments, result)
        except Exception as e:
            log.warning("whisperX diarization failed (%s); single speaker", e)

    words: list[Word] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = str(w.get("word", "")).strip()
            start, end = w.get("start"), w.get("end")
            if not text or start is None or end is None:
                continue
            spk_label = w.get("speaker")
            # Reserve id 0 for unattributed words (no diarization label). Real
            # speakers start at 1, so the first diarized speaker never collides
            # with the "no speaker" fallback — otherwise two distinct talkers
            # merge and the per-speaker caption toggles can't tell them apart.
            spk = speaker_ids.setdefault(spk_label, len(speaker_ids) + 1) if spk_label else 0
            words.append(Word(t=float(start), d=max(float(end) - float(start), 0.01),
                              text=text, speaker=spk))
    if not words:
        raise RuntimeError("whisperX returned no aligned words")
    if progress:
        progress(1.0)
    return Transcript(words=words, language=lang,
                      speakers=max(len(speaker_ids), 1), provider="whisperx")


def _whisper_transcribe(audio_path, language, progress, batch_size: int) -> Transcript:
    model = _load_whisper()
    batched = _batched_pipeline(model, batch_size)
    prompt = _initial_prompt(language)
    if batched is not None:
        try:
            segments, info = _transcribe_with_prompt(
                batched, audio_path, language=language, word_timestamps=True,
                vad_filter=True, batch_size=max(batch_size, 1),
                initial_prompt=prompt)
        except Exception as e:  # any batched-path issue -> sequential, never fail
            log.warning("batched transcribe failed (%s); sequential", e)
            segments, info = _transcribe_with_prompt(
                model, audio_path, language=language, word_timestamps=True,
                vad_filter=True, beam_size=1, initial_prompt=prompt)
    else:
        segments, info = _transcribe_with_prompt(
            model, audio_path, language=language, word_timestamps=True,
            vad_filter=True, beam_size=1, initial_prompt=prompt)
    total = max(getattr(info, "duration", 0.0), 0.001)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if not text:
                continue
            words.append(Word(t=float(w.start), d=max(float(w.end - w.start), 0.01),
                              text=text, speaker=0))
        if progress and seg.end:
            progress(min(seg.end / total, 1.0))
    if not words:
        raise RuntimeError("whisper returned no words")
    return Transcript(words=words, language=getattr(info, "language", "en") or "en",
                      speakers=1, provider="whisper")


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
_FILLER = {
    "en": (
        "so here is the thing that nobody really tells you about this and it "
        "completely changed how i think about the whole problem you have to "
        "start small stay consistent and let the results compound over time "
        "because that is where the real magic actually happens for everyone"
    ).split(),
    "de": (
        "also hier ist die sache die dir niemand wirklich erzählt und das hat "
        "komplett verändert wie ich über das ganze problem denke du musst "
        "klein anfangen konsequent bleiben und die ergebnisse mit der zeit "
        "wachsen lassen weil genau dort die wahre magie wirklich passiert"
    ).split(),
}


def synthetic_transcript(audio_path: str, *, lang: str | None = None) -> Transcript:
    """Deterministic, evenly-timed filler covering the media duration.

    Used only when no ASR is available. The words won't match the audio, but the
    timing is real and we synthesise sentence boundaries (terminal punctuation +
    a short pause), so detection, captioning, and rendering exercise exactly the
    same code paths as the real transcript.
    """
    try:
        dur = ffmpeg.probe(audio_path).duration
    except Exception:
        dur = 60.0
    dur = max(dur, 5.0)
    code = (lang or "en").lower()[:2]
    filler = _FILLER.get(code, _FILLER["en"])
    per = 0.42  # ~2.4 words/sec
    words: list[Word] = []
    t = 0.0
    i = 0
    sentence_len = 0
    target_len = 10
    # Walk real time forward (word pace + sentence pauses) and stop at the media
    # duration — counting words up front ignored the pauses and overran EOF,
    # producing clips past the end of the file.
    while t + per <= dur:
        text = filler[i % len(filler)]
        sentence_len += 1
        # Close a sentence every ~10-14 words: add punctuation + a real pause.
        if sentence_len >= target_len:
            text = text + "."
            sentence_len = 0
            target_len = 10 + (i % 5)
            words.append(Word(t=round(t, 3), d=round(per * 0.9, 3), text=text, speaker=0))
            t += per + 0.7  # pause that the sentence segmenter will pick up
        else:
            words.append(Word(t=round(t, 3), d=round(per * 0.9, 3), text=text, speaker=0))
            t += per
        i += 1
    return Transcript(words=words, language=code if code in _FILLER else "en",
                      speakers=1, provider="synthetic")

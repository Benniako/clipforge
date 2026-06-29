"""Local ASR benchmark candidates for choosing the best transcription path.

The production pipeline stays conservative: whisperX when installed, otherwise
faster-whisper. This module is an opt-in harness for testing newer alternatives
on the same local audio before changing defaults.
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AsrCandidate:
    engine: str
    model: str
    label: str
    available: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AsrBenchResult:
    candidate: AsrCandidate
    seconds: float
    words: int
    language: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidate"] = self.candidate.to_dict()
        return d


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def candidate_matrix(settings=None, *, models: list[str] | None = None) -> list[AsrCandidate]:
    """Installed-aware ASR candidates worth benchmarking locally."""
    if settings is None:
        from ..config import get_settings
        settings = get_settings()
    fast_ok = _has_module("faster_whisper")
    wx_ok = _has_module("whisperx")
    nemo_ok = _has_module("nemo.collections.asr")

    preferred = models or [
        settings.whisper_model,
        "large-v3-turbo",
        "distil-large-v3",
    ]
    seen: set[str] = set()
    out: list[AsrCandidate] = []
    for model in preferred:
        model = (model or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        out.append(AsrCandidate(
            engine="faster-whisper", model=model,
            label=f"faster-whisper/{model}", available=fast_ok,
            reason="" if fast_ok else "faster-whisper is not installed"))
    out.append(AsrCandidate(
        engine="whisperx", model=settings.whisper_model,
        label=f"whisperX/{settings.whisper_model}", available=wx_ok,
        reason="" if wx_ok else "whisperX is not installed"))
    out.append(AsrCandidate(
        engine="nemo-parakeet", model="nvidia/parakeet-tdt-0.6b-v3",
        label="NeMo Parakeet TDT 0.6B v3", available=nemo_ok,
        reason="" if nemo_ok else "NVIDIA NeMo ASR is not installed"))
    return out


def run_candidate(candidate: AsrCandidate, audio_path: str, *,
                  language: str | None = None, device: str | None = None,
                  compute_type: str | None = None) -> AsrBenchResult:
    """Run one candidate and return timing + rough word count."""
    started = time.perf_counter()
    try:
        if candidate.engine == "faster-whisper":
            words, lang = _run_faster(candidate.model, audio_path, language,
                                      device, compute_type)
        elif candidate.engine == "whisperx":
            words, lang = _run_whisperx(candidate.model, audio_path, language,
                                        device, compute_type)
        elif candidate.engine == "nemo-parakeet":
            words, lang = _run_nemo_parakeet(candidate.model, audio_path)
        else:
            raise ValueError(f"unknown ASR engine: {candidate.engine}")
        return AsrBenchResult(candidate, round(time.perf_counter() - started, 3),
                              words, lang)
    except Exception as exc:
        return AsrBenchResult(candidate, round(time.perf_counter() - started, 3),
                              0, error=str(exc)[:300])


def benchmark(audio_path: str, *, candidates: list[AsrCandidate] | None = None,
              language: str | None = None, device: str | None = None,
              compute_type: str | None = None,
              include_unavailable: bool = False) -> list[AsrBenchResult]:
    """Benchmark available candidates on a local WAV/media file."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(audio_path)
    cand = candidates or candidate_matrix()
    if not include_unavailable:
        cand = [c for c in cand if c.available]
    return [run_candidate(c, str(path), language=language, device=device,
                          compute_type=compute_type) for c in cand]


def _run_faster(model_name: str, audio_path: str, language: str | None,
                device: str | None, compute_type: str | None) -> tuple[int, str | None]:
    from faster_whisper import WhisperModel

    dev = device or "cpu"
    ct = compute_type or ("float16" if dev == "cuda" else "int8")
    model = WhisperModel(model_name, device=dev, compute_type=ct)
    segments, info = model.transcribe(
        audio_path, language=None if language in (None, "", "auto") else language,
        word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
        beam_size=3)
    words = 0
    for seg in segments:
        words += len([w for w in (seg.words or []) if w.word.strip()])
    return words, getattr(info, "language", language)


def _run_whisperx(model_name: str, audio_path: str, language: str | None,
                  device: str | None, compute_type: str | None) -> tuple[int, str | None]:
    import whisperx

    dev = device or "cpu"
    ct = compute_type or ("float16" if dev == "cuda" else "int8")
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(model_name, dev, compute_type=ct)
    result = model.transcribe(
        audio, language=None if language in (None, "", "auto") else language,
        batch_size=8, condition_on_previous_text=False)
    words = sum(len(str(seg.get("text", "")).split()) for seg in result.get("segments", []))
    return words, result.get("language", language)


def _run_nemo_parakeet(model_name: str, audio_path: str) -> tuple[int, str | None]:
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(model_name)
    out = model.transcribe([audio_path])
    text = ""
    if out:
        first = out[0]
        text = getattr(first, "text", first if isinstance(first, str) else str(first))
    return len(text.split()), None

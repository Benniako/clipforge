"""Silero VAD — pin captions to the *exact* speech, not Whisper's loose edges.

Whisper word timestamps drift a little — a word's box can open before the mouth
moves or hang a beat after. Silero VAD is a tiny, fast (sub-ms/chunk on CPU),
enterprise-grade voice-activity model. When it's installed we use it to (a) clamp
every word's on-screen span to the speech region it falls in, and (b) drop words
that land entirely in silence (ASR hallucinations / breaths). The result: caption
words appear and clear precisely when the person is talking.

Fully optional: no ``silero_vad`` ⇒ :func:`speech_intervals` returns None and the
pipeline keeps Whisper's timings unchanged. The clamp/drop logic is a pure
function so it's unit-tested without the model.
"""
from __future__ import annotations

import logging

from ..config import get_settings
from ..models import Word

log = logging.getLogger("clipforge.vad")

_model = None  # cached (model, get_speech_timestamps) or False


def _load():
    global _model
    if _model is not None:
        return _model or None
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        _model = (load_silero_vad(), get_speech_timestamps)
        log.info("Silero VAD loaded")
    except Exception as e:
        log.info("Silero VAD unavailable (%s)", e)
        _model = False
    return _model or None


def available() -> bool:
    """True when Silero VAD can be loaded in this environment."""
    return get_settings().has_vad and _load() is not None


def speech_intervals(wav_path: str, *, sr: int = 16000) -> list[tuple[float, float]] | None:
    """Speech (start, end) spans in seconds, or None when VAD isn't available."""
    if not get_settings().has_vad:
        return None
    loaded = _load()
    if loaded is None:
        return None
    model, get_speech_timestamps = loaded
    try:
        from silero_vad import read_audio

        wav = read_audio(wav_path, sampling_rate=sr)
        ts = get_speech_timestamps(wav, model, sampling_rate=sr)
        return [(t["start"] / sr, t["end"] / sr) for t in ts]
    except Exception as e:
        log.warning("VAD failed (%s); keeping ASR timings", e)
        return None


def refine_words(words: list[Word], speech: list[tuple[float, float]],
                 *, pad: float = 0.06, min_overlap: float = 0.10) -> list[Word]:
    """Clamp each word's span to the speech region it overlaps; drop silent ones.

    Pure (no model needed → unit-tested). ``pad`` keeps a hair of air so a clamp
    never clips the consonant; ``min_overlap`` is how much a word must intersect
    speech (fraction of the word's own duration or absolute seconds, whichever is
    smaller) to be kept. Default 0.10 (100 ms) is more aggressive than the old
    0.04 — hallucinated whispers are short and barely overlap real speech.
    """
    if not speech:
        return words
    spans = sorted(speech)
    out: list[Word] = []
    for w in words:
        ws, we = w.t, w.end
        best: tuple[float, float] | None = None
        best_ov = 0.0
        for a, b in spans:
            ov = min(we, b) - max(ws, a)
            if ov > best_ov:
                best_ov, best = ov, (a, b)
        if best is None or best_ov < min_overlap:
            continue  # word sits in silence — drop it
        a, b = best
        ns = max(ws, a - pad)
        ne = min(we, b + pad)
        if ne - ns < 0.04:
            ne = ns + 0.04
        out.append(Word(t=round(ns, 3), d=round(ne - ns, 3), text=w.text,
                        speaker=w.speaker))
    return out or words  # never wipe everything (defensive)

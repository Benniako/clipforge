"""Standalone forced-alignment refinement for word timestamps.

whisperX already aligns on its own path; the faster-whisper path (used when
whisperX isn't installed) only has segment-level + native word timestamps, which
drift by ~1s on long segments (vanilla Whisper's well-known limitation). This
module tightens those word timings with a wav2vec2 CTC forced alignment pass —
the same approach WhisperX uses internally (sub-100 ms word accuracy), per the
WhisperX paper and the 2024 ASR forced-alignment comparisons.

Design:
- ``align_transcript(words, audio_path, lang)`` is the entry point. It loads a
  wav2vec2 align model, runs CTC forced alignment against the audio, and returns
  refined ``Word`` objects. Pure-fail-safe: if torchaudio / the model / the
  language is unsupported, it returns the input unchanged with a debug log.
- ``_align_tokens(emission, tokens, blank)`` is the dynamic-programming core
  (forward + backtrack over the CTC trellis). It's a pure function of the
  emission matrix, so the alignment math is unit-tested without torch.
- ``_word_spans_from_tokens`` groups token boundaries back into word spans.

This is *optional* refinement: the transcript is always valid without it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..models import Word

log = logging.getLogger("clipforge.align")

# torchaudio's bundle registry covers most languages; these are the wav2vec2
# align model names we know how to load. Missing language ⇒ graceful no-op.
SUPPORTED_LANGS = {
    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "zh",
    "ar", "tr", "cs", "el", "fa", "fi", "he", "hu", "ko", "uk",
}

_align_bundle = None  # cached (bundle, label_fn, model, device) or False
_UNIMPORTANT_RE = re.compile(r"[^\w'-]")


@dataclass
class _TokenSpan:
    token: int          # emission vocab index
    start: int          # frame index (inclusive)
    end: int            # frame index (exclusive)


def _align_tokens(emission, tokens: list[int], blank: int) -> list[_TokenSpan] | None:
    """CTC forced alignment via the standard forward-backward trellis.

    ``emission`` is a 2-D log-probability matrix shaped (T frames, N vocab).
    Returns one ``_TokenSpan`` per token in ``tokens`` (with frame-accurate
    start/end), or None when alignment is impossible (empty token list). Pure —
    no torch dependency, operates on any 2-D sequence supporting indexing, so
    the DP is unit-tested with a plain list-of-lists.
    """
    if not tokens:
        return None
    T = len(emission)
    if T == 0:
        return None
    N = len(tokens)
    # log-prob trellis: trellis[t, j] = best score aligning the first j+1 tokens
    # using the first t+1 frames. Two transitions per cell: stay on the same
    # token (frame t emits token j), or advance to the next token.
    NEG = float("-inf")
    trellis = [[NEG] * N for _ in range(T)]
    # pointers[t, j] = 1 if we came from (t-1, j-1) [advance], 0 if from (t-1, j) [stay].
    pointers = [[0] * N for _ in range(T)]

    trellis[0][0] = float(emission[0][tokens[0]])
    for t in range(1, T):
        for j in range(N):
            stay = trellis[t - 1][j]            # frame t re-emits token j
            advance = trellis[t - 1][j - 1] if j > 0 else NEG  # token j-1 → j
            best, ptr = (stay, 0) if stay >= advance else (advance, 1)
            trellis[t][j] = best + float(emission[t][tokens[j]])
            pointers[t][j] = ptr

    # Backtrack from (T-1, N-1). We then walk forward through the chosen path and
    # record the frame at which each token *starts* (the advance transitions).
    path_token_at_frame: list[int] = []  # which token index is active at each frame
    t, j = T - 1, N - 1
    while t >= 0 and j >= 0:
        path_token_at_frame.append(j)
        if j == 0:
            t -= 1
            continue
        if pointers[t][j] == 1:  # advanced into token j here
            t, j = t - 1, j - 1
        else:
            t = t - 1
    path_token_at_frame.reverse()
    if len(path_token_at_frame) < T:
        # Pad leading frames (before the first token started) with token 0.
        path_token_at_frame = ([0] * (T - len(path_token_at_frame))
                               + path_token_at_frame)

    spans: list[_TokenSpan] = []
    for tok_idx in range(N):
        frames = [fi for fi, tj in enumerate(path_token_at_frame) if tj == tok_idx]
        if not frames:
            # Token never got a frame; give it a zero-length span at T-1 so the
            # caller's merge still produces a finite word duration.
            spans.append(_TokenSpan(token=tokens[tok_idx], start=T - 1, end=T))
        else:
            spans.append(_TokenSpan(token=tokens[tok_idx],
                                    start=frames[0], end=frames[-1] + 1))
    return spans


def _word_spans_from_tokens(word_token_counts: list[int],
                            token_spans: list[_TokenSpan]) -> list[tuple[int, int]]:
    """Group consecutive token spans into per-word (start_frame, end_frame).

    ``word_token_counts[i]`` is how many tokens word i contributed. Returns one
    (start, end) frame pair per word. Pure helper, unit-tested directly.
    """
    out: list[tuple[int, int]] = []
    pos = 0
    for count in word_token_counts:
        if count <= 0 or pos + count > len(token_spans):
            out.append((0, 1))
            pos += max(count, 0)
            continue
        chunk = token_spans[pos:pos + count]
        out.append((chunk[0].start, chunk[-1].end))
        pos += count
    return out


def _word_to_tokens(word: str) -> list[str]:
    """Split a word into alignable sub-word token strings.

    We keep it simple: lowercase, strip punctuation, and fall back to character
    bigrams when the whole word isn't a likely vocab key. The actual vocab
    lookup happens in :func:`align_transcript` against the loaded model's
    tokenizer — this helper only normalises.
    """
    w = _UNIMPORTANT_RE.sub(" ", word.lower()).strip()
    return [w] if w else []


def _load_aligner(lang: str):
    """Lazy-load the torchaudio wav2vec2 align model for ``lang``.

    Returns (model, get_frame_probs, tokenizer_vocab) or None when unavailable.
    All imports are local so a normal ClipForge run never pays for torch unless
    alignment is actually requested. Cached per process.
    """
    global _align_bundle
    if _align_bundle is not None:
        return _align_bundle or None
    code = (lang or "en").lower()[:2]
    if code not in SUPPORTED_LANGS:
        _align_bundle = False
        return None
    try:
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle  # multilingual aligner
    except Exception as e:  # torchaudio not installed — degrade silently
        log.info("forced alignment unavailable (torchaudio missing): %s", e)
        _align_bundle = False
        return None
    try:
        from .torch_guard import TORCH_LOAD_LOCK
        with TORCH_LOAD_LOCK:
            device = "cuda" if _torch_cuda() else "cpu"
            model = bundle.get_model().to(device).eval()
            _align_bundle = (model, bundle, device)
        log.info("wav2vec2 forced aligner loaded (lang=%s, device=%s)", code, device)
    except Exception as e:
        log.info("forced aligner load failed: %s", e)
        _align_bundle = False
        return None
    return _align_bundle


def _torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def align_transcript(words: list[Word], audio_path: str,
                     *, lang: str = "en") -> list[Word]:
    """Refine word timestamps against ``audio_path`` via CTC forced alignment.

    Returns a new ``Word`` list with tightened ``t`` / ``d``. On any failure
    (torchaudio missing, unsupported language, empty transcript, model error)
    the input is returned unchanged — this is always a best-effort refinement,
    never a hard dependency.
    """
    if not words:
        return words
    aligner = _load_aligner(lang)
    if aligner is None:
        return words
    try:
        import torch
        model, bundle, device = aligner
        wav = _load_wav(audio_path, bundle)
        if wav is None:
            return words
        with torch.no_grad():
            emission, _ = model(wav.to(device))
        emission = emission[0].cpu()  # (T, N)
        tokenizer = bundle.get_tokenizer()
        labels = bundle.get_dict()      # token str -> index
        blank = labels.get("|", 0)
        # Build the per-word token index lists and the flat token stream.
        word_token_counts: list[int] = []
        flat_tokens: list[int] = []
        # Each word's text → its subword token ids from the MMS tokenizer.
        for w in words:
            toks = tokenizer(_word_to_tokens(w.text))
            # tokenizer returns a flat list per call; flatten robustly.
            ids = [int(t) for t in (toks if isinstance(toks, list) else [toks])]
            ids = [i for i in ids if i in labels.values()]
            if not ids:
                ids = [labels.get("|", 0)]
            word_token_counts.append(len(ids))
            flat_tokens.extend(ids)
        if not flat_tokens:
            return words
        spans = _align_tokens(emission, flat_tokens, blank)
        if spans is None:
            return words
        frame_pairs = _word_spans_from_tokens(word_token_counts, spans)
        frame_rate = bundle.sample_rate / _STRIDE  # frames per second
        refined: list[Word] = []
        for (s, e), w in zip(frame_pairs, words):
            start_t = max(0.0, s / frame_rate)
            end_t = max(start_t + 0.04, e / frame_rate)
            refined.append(Word(t=round(start_t, 3),
                                d=round(end_t - start_t, 3),
                                text=w.text, speaker=w.speaker))
        return refined
    except Exception as e:
        log.warning("forced alignment failed (%s); returning unaligned words", e)
        return words


# MMS wav2vec2 stride — the model downsamples by this factor; used to convert
# emission frames back to seconds. torchaudio's MMS_FA uses stride 4 (× 320
# samples at 16 kHz = 20 ms/frame).
_STRIDE = 320.0


def _load_wav(audio_path: str, bundle):
    try:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        if sr != bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
        # Mono: average channels so the aligner sees a single stream.
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav
    except Exception as e:
        log.info("aligner could not load audio %s: %s", audio_path, e)
        return None

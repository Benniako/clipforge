"""Transcript word correction via local LLM (Ollama).

After transcription, a light proofread pass sends the transcript text to the
local LLM with instructions to flag individual words that are clearly misheard
(ASR hallucination, homophone confusion, incoherent babble that doesn't fit
context). Only the flagged words are replaced — every other word keeps its
original timing, confidence, and speaker label.

The prompt is designed for a single-shot JSON response so the whole correction
round-trip is one Ollama call per project (~2-5 s).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict

from ..models import Transcript

log = logging.getLogger(__name__)

_CORRECTION_CACHE: OrderedDict[str, dict[str, str]] = OrderedDict()
_CORRECTION_CACHE_LOCK = threading.Lock()
_CORRECTION_CACHE_MAX = 50


def _build_prompt(text: str, lang: str = "de") -> str:
    """One-shot prompt that asks for individual word corrections only."""
    return (
        f"Transcript from an ASR system on a {lang}-language video. "
        "The transcript is about 95 % correct. "
        "Find individual words that are clearly ASR errors:\n"
        "- misheard due to accent, background noise, or fast speech\n"
        "- words that don't fit the context of the surrounding words\n"
        "- homophone confusions (e.g. 'there' vs 'their')\n"
        "- incoherent fragments that are not real words\n\n"
        "For each error, return a JSON object with keys:\n"
        '- "original": the exact wrong word\n'
        '- "corrected": the correct word (empty string to delete it)\n\n'
        "Rules:\n"
        "- ONLY change words that are clearly wrong. Leave everything else.\n"
        "- Do NOT rewrite, rephrase, or summarise sentences.\n"
        "- Do NOT add punctuation or change capitalisation.\n"
        "- If everything looks correct, return an empty array [].\n"
        "- Respond with ONLY the JSON array, no preamble.\n\n"
        f"Transcript:\n{text}"
    )


def _parse_corrections(raw: str) -> dict[str, str]:
    """Parse the LLM response into a {original → corrected} map."""
    # Strip any markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON array in the response
        m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if m:
            try:
                items = json.loads(m.group())
            except json.JSONDecodeError:
                log.warning("correction LLM response unparseable: %.200s", raw)
                return {}
        else:
            log.warning("correction LLM response unparseable: %.200s", raw)
            return {}
    if not isinstance(items, list):
        return {}
    corrections: dict[str, str] = {}
    for item in items:
        orig = str(item.get("original", "")).strip().lower()
        corr = str(item.get("corrected", "")).strip()
        if orig and corr is not None and orig != corr.lower():
            corrections[orig] = corr
    return corrections


def correct_transcript(transcript: Transcript, lang: str = "de") -> int:
    """Run a single Ollama correction pass over the full transcript text.

    Returns the number of words that were corrected (0 if none).
    Mutates ``transcript.words`` in place — timing, speaker, and confidence
    of every uncorrected word are preserved exactly.
    """
    from . import ollama_client as oc

    # Build the full transcript text (lowercase for matching)
    full_text = " ".join(w.text for w in transcript.words)

    # Use the settings-configured text model
    try:
        from ..config import get_settings
        model = (get_settings().ollama_text or "qwen3:8b").rsplit("/", 1)[-1]
    except Exception:
        model = "qwen3:8b"

    # Check cache
    cache_key = f"{model}:{full_text}"  # deterministic, not hash() which varies per process
    with _CORRECTION_CACHE_LOCK:
        cached = _CORRECTION_CACHE.get(cache_key)
        if cached is not None:
            _CORRECTION_CACHE.move_to_end(cache_key)  # LRU refresh
    if cached is None:
        prompt = _build_prompt(full_text, lang=lang)
        try:
            raw = oc.generate(model, prompt, temperature=0.1, max_tokens=2048)
        except Exception as e:
            log.warning("correction LLM call failed: %s", e)
            return 0
        corrections = _parse_corrections(raw)
        with _CORRECTION_CACHE_LOCK:
            _CORRECTION_CACHE[cache_key] = corrections
            _CORRECTION_CACHE.move_to_end(cache_key)
            if len(_CORRECTION_CACHE) > _CORRECTION_CACHE_MAX:
                _CORRECTION_CACHE.popitem(last=False)  # LRU evict oldest
    else:
        corrections = cached

    if not corrections:
        return 0

    # Apply corrections in-place
    fixed = 0
    for w in transcript.words:
        low = w.text.strip().lower()
        if low in corrections:
            replacement = corrections[low]
            # Empty string = delete this word from the output, but preserve
            # its timing slot so every other word keeps its timestamp.
            w.text = replacement
            fixed += 1

    if fixed:
        log.info("corrected %d word(s) via LLM (%s)", fixed, model)
    return fixed

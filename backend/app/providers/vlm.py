"""Optional local vision-language second opinion on virality (Ollama) — looks at
the actual frames, not just the words.

The text LLM re-rank reads the transcript; a VLM reads the *picture* — facial
expression, on-screen action, framing — which is often where short-form virality
actually lives (a shocked face, an explosion, a clean punchline reaction). When
Ollama is running a vision model (qwen2.5-vl, llava, llama3.2-vision, …) we send a
few keyframes per clip and get a 0..1 read + one-line reason, blended in exactly
like the text read: bounded, explainable, never overriding the signal sum.

Fully optional and budgeted: no vision model ⇒ :func:`available` is False and the
pipeline is unchanged; a slow model can never stall a run (per-clip timeout +
overall budget). Runs entirely on the user's own GPU.

The Ollama mechanics are shared with the text LLM via ``ollama_client``; this
module keeps only the vision-model picker, the prompt text, and the
menu/lobby-reason clamp.
"""
from __future__ import annotations

import base64
import logging
import os
import re

from . import ollama_client as _oc

log = logging.getLogger("clipforge.vlm")

_MODEL = os.environ.get("CLIPFORGE_VLM_MODEL", "")  # empty = auto-pick

# Vision-capable Ollama families. Within a family the largest installed size
# wins, so qwen2.5vl:32b beats qwen2.5vl:7b automatically.
_VISION_PREFERRED = ("qwen3-vl", "qwen2.5vl", "qwen2.5-vl", "qwen2-vl",
                     "llama3.2-vision", "llava-llama3", "llava",
                     "bakllava", "minicpm-v", "moondream", "gemma3")


def _is_vision(tag: str) -> bool:
    name = tag.lower()
    return any(name == p or name.startswith(p) for p in _VISION_PREFERRED)


def _rank_vision_model(tag: str) -> tuple[int, int, float, str]:
    low = tag.lower()
    family = next((i for i, name in enumerate(_VISION_PREFERRED)
                   if low == name or low.startswith(name)), len(_VISION_PREFERRED))
    return (1 if _is_vision(low) else 0, -family, _oc.size_b(low), low)


def _resolve_model(tags: list[str]) -> str | None:
    if _MODEL:
        return _MODEL
    vision = [t for t in tags if _is_vision(t)]
    return max(vision, key=_rank_vision_model) if vision else None


# Shared availability cache, wired to the vision-model picker.
_cache = _oc.AvailabilityCache(resolver=_resolve_model)


def available() -> bool:
    """True if Ollama answers and has a usable vision model. Cached 60s."""
    return _cache.available()


def active_model() -> str | None:
    return _cache.active_model()


def keyframe_times(start: float, end: float, n: int = 3) -> list[float]:
    """Evenly-spaced sample times inside [start, end] (pure, unit-tested)."""
    dur = max(end - start, 0.0)
    if dur <= 0:
        return [start]
    n = max(1, n)
    return [round(start + dur * (i + 1) / (n + 1), 3) for i in range(n)]


def _grab_frames_b64(src_path: str, start: float, end: float, n: int) -> list[str]:
    import tempfile
    from pathlib import Path

    from ..media import ffmpeg

    out: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(keyframe_times(start, end, n)):
            f = Path(tmp) / f"k{i}.jpg"
            try:
                ffmpeg.grab_frame(src_path, f, t=t, width=384)
                out.append(base64.b64encode(f.read_bytes()).decode("ascii"))
            except Exception as e:
                log.warning("vlm frame grab at %.1fs failed: %s", t, e)
    return out


# Re-exposed shared regex/parsers for the helpers below.
_THINK_RE = _oc.THINK_RE
_SCORE_RE = _oc.SCORE_RE

_NEGATIVE_REASON_TERMS = (
    "menu", "lobby", "loading", "black screen", "static", "desktop",
    "scoreboard only", "boring", "transition", "blurry", "unclear",
)
_PROMPTS = {
    "en": (
        "Rate how viral these clip frames look. Reward visible action, reaction, "
        "clarity, and a strong first-frame hook. Penalize menu/lobby/loading/"
        "black/desktop/blur/scoreboard-only/no-action frames below 35. Reply:\n"
        "SCORE: <0-100> | REASON: <max 8 words>"
    ),
    "de": (
        "Bewerte, wie viral diese Clip-Frames wirken. Belohne sichtbare Action, "
        "Reaktion, klare Bildsprache und einen starken Hook in den ersten Frames. "
        "Bestrafe Menue/Lobby/Ladebildschirm/schwarzes Bild/Desktop/unscharfe/"
        "nur Scoreboard/keine Action Frames unter 35. Antworte exakt:\n"
        "SCORE: <0-100> | REASON: <max 8 words>"
    ),
}


_CUE_ALLOWLIST_RE = re.compile(r"^[A-Za-z0-9 _\-]{1,24}$")


def _prompt_for(lang: str | None, cues: list[str] | None = None) -> str:
    base = _PROMPTS.get((lang or "en")[:2].lower(), _PROMPTS["en"])
    # Sanitise learned OCR labels before they touch the prompt. These come from
    # on-screen text the user calibrated (or that OCR auto-detected), so they're
    # untrusted input — a crafted overlay could otherwise inject prompt text that
    # persists across every future scan of this game profile. Only short, plain
    # label-like strings make it through; anything that looks like a sentence or
    # an instruction is dropped.
    hints = []
    for c in (cues or []):
        c = (c or "").strip()
        if c and _CUE_ALLOWLIST_RE.match(c):
            hints.append(c)
    if hints:
        # Steer the read toward the project's own visual cues (kill feed,
        # victory screen, …) without overriding the scoring rubric.
        base += "\nWatch especially for: " + ", ".join(hints[:8]) + "."
    return base


def _parse(text: str) -> tuple[float, str] | None:
    """Parse a SCORE|REASON reply, clamping menu/lobby reads below 0.35."""
    return _oc.parse_score_reason(
        text,
        clamp_reason_below=0.35,
        negative_terms=_NEGATIVE_REASON_TERMS,
    )


def score_visual(src_path: str, start: float, end: float, *,
                 n_frames: int = 3, timeout: float = 30.0,
                 lang: str = "en",
                 cues: list[str] | None = None) -> tuple[float, str] | None:
    """Ask the local VLM how viral a clip *looks*. (0..1, reason) or None."""
    if not available():
        return None
    model = active_model()
    images = _grab_frames_b64(src_path, start, end, n_frames)
    if not model or not images:
        return None
    prompt = _prompt_for(lang, cues)
    out = _oc.generate(model=model, prompt=prompt, images=images,
                       timeout=timeout, temperature=0.4, num_predict=60)
    return _parse(out) if out else None


def score_visuals(src_path: str, spans: list[tuple[float, float]], *,
                  budget: float = 45.0, max_workers: int = 2,
                  n_frames: int = 3, timeout: float = 30.0,
                  lang: str = "en", cues: list[str] | None = None
                  ) -> dict[int, tuple[float, str]]:
    """Concurrent VLM reads for many clip spans, capped by a time budget.

    Returns {index: (viral, reason)} for whatever finished in time; the rest keep
    their existing score — a slow model can never stall a run."""
    indexed = list(enumerate(spans))
    items = [(src_path, a, b, n_frames, timeout, lang, cues) for _, (a, b) in indexed]
    result, _ = _oc.run_budgeted(lambda a: score_visual(*a), items,
                              budget=budget, max_workers=max_workers, logger=log)
    return {indexed[pos][0]: val for pos, val in result.items()}

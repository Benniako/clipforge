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


def model_from_tags(tags: str | list[str]) -> str | None:
    """Resolve the vision model from a cached Ollama tag snapshot."""
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return _resolve_model(tags)


# Shared availability cache, wired to the vision-model picker.
_cache = _oc.AvailabilityCache(resolver=_resolve_model)


def available() -> bool:
    """True if Ollama answers and has a usable vision model. Cached 60s."""
    return _cache.available()


def unload_model() -> None:
    """Unload active vision model from Ollama VRAM."""
    model = active_model()
    if model:
        _oc.unload_model(model)


def active_model() -> str | None:
    return _cache.active_model()


def keyframe_times(start: float, end: float, n: int = 3) -> list[float]:
    """Evenly-spaced sample times inside [start, end] (pure, unit-tested)."""
    dur = max(end - start, 0.0)
    if dur <= 0:
        return [start]
    n = max(1, n)
    return [round(start + dur * (i + 1) / (n + 1), 3) for i in range(n)]


def hierarchical_keyframe_times(start: float, end: float,
                                climax_t: float | None = None,
                                n: int = 8) -> list[float]:
    """Climax-anchored hierarchical frame times inside [start, end].

    Designed specifically for viral short-form retention evaluation (TikTok /
    YouTube Shorts / Reels):
      - Frames 1..3: Hook Window (first 0.2s - 2.5s) to evaluate opening visual curiosity.
      - Frame 4: Context / Narrative Buildup (bridge between hook and action).
      - Frames 5..7: Climax Action Cluster (tight anticipation, impact, payoff around climax_t).
      - Frame 8: Resolution / Outro (closing emotion or loop reset).

    When climax_t is unspecified or out-of-range, defaults to the Golden Climax
    Interval (78% of duration). Returns n strictly sorted timestamps bounded
    within [start + 0.04, end - 0.04].
    """
    dur = max(end - start, 0.0)
    if dur <= 0:
        return [start]
    n = max(1, n)
    if n <= 3:
        if n == 1:
            c = climax_t if (climax_t and start < climax_t < end) else start + 0.78 * dur
            return [round(c, 3)]
        if n == 2:
            return [round(start + 0.25 * dur, 3), round(start + 0.8 * dur, 3)]
        c = climax_t if (climax_t and start < climax_t < end) else start + 0.78 * dur
        return [round(start + min(0.8, dur * 0.15), 3), round(c, 3),
                round(end - min(0.5, dur * 0.1), 3)]

    # Golden Climax fallback if climax_t not provided or out of range
    if climax_t is None or climax_t <= start + 0.5 or climax_t >= end - 0.3:
        climax_t = start + 0.78 * dur

    # For very short micro-clips (< 3.0s), even division avoids crowding
    if dur < 3.0:
        return [round(start + dur * (i + 1) / (n + 1), 3) for i in range(n)]

    raw_points = []
    if n >= 8:
        # Hook frames (first 2.5s)
        h0 = start + min(0.3, dur * 0.03)
        h1 = start + min(1.2, dur * 0.10)
        h2 = start + min(2.5, dur * 0.20)
        if h2 >= climax_t - 1.2:
            h2 = start + (climax_t - start) * 0.45
            h1 = start + (climax_t - start) * 0.25
            h0 = start + (climax_t - start) * 0.10
        raw_points.extend([h0, h1, h2])

        # Narrative buildup context frame
        raw_points.append(h2 + 0.5 * (climax_t - 0.6 - h2))

        # Climax frames: anticipation, impact instant, payoff
        c_lead = max(0.35, min(0.65, (climax_t - start) * 0.08))
        c_trail = max(0.35, min(0.65, (end - climax_t) * 0.25))
        raw_points.extend([climax_t - c_lead, climax_t, climax_t + c_trail])

        # Outro resolution frame
        raw_points.append(end - min(0.4, dur * 0.05))
    elif n == 6:
        h0 = start + min(0.4, dur * 0.05)
        h1 = start + min(1.8, (climax_t - start) * 0.4)
        c_lead = max(0.35, min(0.65, (climax_t - start) * 0.08))
        c_trail = max(0.35, min(0.65, (end - climax_t) * 0.25))
        raw_points.extend([h0, h1, climax_t - c_lead, climax_t, climax_t + c_trail,
                           end - min(0.4, dur * 0.05)])
    elif n == 5:
        h0 = start + min(0.4, dur * 0.05)
        h1 = start + min(1.8, (climax_t - start) * 0.4)
        raw_points.extend([h0, h1, climax_t - 0.4, climax_t + 0.4,
                           end - min(0.4, dur * 0.05)])
    else:  # n == 4
        raw_points.extend([start + min(0.5, dur * 0.1), (start + climax_t) * 0.5,
                           climax_t, end - min(0.4, dur * 0.05)])

    clamped = [round(min(max(p, start + 0.04), end - 0.04), 3) for p in raw_points]
    clamped.sort()

    min_gap = min(0.20, dur / (n * 5))
    cleaned = [clamped[0]]
    for p in clamped[1:]:
        if p >= cleaned[-1] + min_gap:
            cleaned.append(p)

    while len(cleaned) < n:
        gaps = [(cleaned[i + 1] - cleaned[i], i) for i in range(len(cleaned) - 1)]
        if not gaps:
            break
        max_gap, idx = max(gaps, key=lambda x: x[0])
        mid = round((cleaned[idx] + cleaned[idx + 1]) / 2.0, 3)
        cleaned.insert(idx + 1, mid)

    return sorted(cleaned[:n])


def _grab_frames_b64(src_path: str, start: float, end: float, n: int,
                     climax_t: float | None = None) -> list[str]:
    import tempfile
    from pathlib import Path

    from ..media import ffmpeg

    out: list[str] = []
    if n >= 4 or climax_t is not None:
        times = hierarchical_keyframe_times(start, end, climax_t=climax_t, n=n)
    else:
        times = keyframe_times(start, end, n)
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(times):
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
        "Rate how viral these clip frames look for TikTok, YouTube Shorts, and Reels. "
        "The frames are ordered chronologically: opening hook (frames 1-3), mid action, "
        "and climax/payoff (anchored at highlight moment). Reward visible high-energy action, "
        "expressive facecam reactions, clear visuals, and strong opening hook. "
        "Penalize menu/lobby/loading/black/desktop/blur/scoreboard-only/static frames below 35. Reply:\n"
        "SCORE: <0-100> | REASON: <max 8 words>"
    ),
    "de": (
        "Bewerte, wie viral diese Clip-Frames fuer TikTok, YouTube Shorts und Reels wirken. "
        "Die Frames sind chronologisch aufgebaut: Opening-Hook (Frames 1-3), Action-Aufbau "
        "und Climax/Payoff (am Highlight-Moment). Belohne sichtbare High-Energy Action, "
        "ausdrucksstarke Facecam-Reaktionen, klare Bildsprache und einen starken Hook zu Beginn. "
        "Bestrafe Menue/Lobby/Ladebildschirm/schwarzes Bild/Desktop/unscharfe/nur Scoreboard Frames unter 35. Antworte exakt:\n"
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
                 climax_t: float | None = None,
                 n_frames: int = 3, timeout: float = 30.0,
                 lang: str = "en",
                 cues: list[str] | None = None) -> tuple[float, str] | None:
    """Ask the local VLM how viral a clip *looks*. (0..1, reason) or None."""
    if not available():
        return None
    model = active_model()
    images = _grab_frames_b64(src_path, start, end, n_frames, climax_t=climax_t)
    if not model or not images:
        return None
    prompt = _prompt_for(lang, cues)
    out = _oc.generate(model=model, prompt=prompt, images=images,
                       timeout=timeout, temperature=0.4, num_predict=60)
    return _parse(out) if out else None


def score_visuals(src_path: str,
                  spans: list[tuple[float, float] | tuple[float, float, float | None]], *,
                  budget: float = 45.0, max_workers: int = 2,
                  n_frames: int = 3, timeout: float = 30.0,
                  lang: str = "en", cues: list[str] | None = None
                  ) -> dict[int, tuple[float, str]]:
    """Concurrent VLM reads for many clip spans, capped by a time budget.

    Supports both (start, end) and (start, end, climax_t) spans.
    Returns {index: (viral, reason)} for whatever finished in time; the rest keep
    their existing score — a slow model can never stall a run."""
    indexed = list(enumerate(spans))
    items = []
    for _, span in indexed:
        a = float(span[0])
        b = float(span[1])
        ct = float(span[2]) if len(span) >= 3 and span[2] is not None else None
        items.append((src_path, a, b, ct, n_frames, timeout, lang, cues))

    def _worker(args):
        path, a, b, ct, nf, to, l, c = args
        return score_visual(path, a, b, climax_t=ct, n_frames=nf,
                            timeout=to, lang=l, cues=c)

    result, _ = _oc.run_budgeted(_worker, items,
                                 budget=budget, max_workers=max_workers, logger=log)
    return {indexed[pos][0]: val for pos, val in result.items()}

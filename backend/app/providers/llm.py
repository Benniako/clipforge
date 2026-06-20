"""Optional local LLM (Ollama) for punchier titles/hooks — no cloud, no API key.

If an Ollama server is reachable (default http://localhost:11434) it's used to
write a scroll-stopping title from the clip's transcript. If it isn't running,
every call returns None and the caller keeps the heuristic title — so this is a
pure, safe upgrade. Runs on the user's own GPU via Ollama.

Enable by installing Ollama (https://ollama.com) and pulling a small model, e.g.
``ollama pull llama3.2``. Configure with CLIPFORGE_OLLAMA_URL / CLIPFORGE_LLM_MODEL.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request

log = logging.getLogger("clipforge.llm")

_URL = os.environ.get("CLIPFORGE_OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Empty = auto-pick the strongest installed model (see _PREFERRED).
_MODEL = os.environ.get("CLIPFORGE_LLM_MODEL", "")

# Auto-pick order when CLIPFORGE_LLM_MODEL isn't set. Within each family the
# largest installed size wins, so pulling qwen3:32b later automatically upgrades
# the local title/virality model.
_FAMILY_RANK = (
    "qwen3", "gemma4", "llama3.3", "llama3.1", "gemma3", "qwen2.5",
    "mistral", "llama3.2", "deepseek-r1",
)
_VISION_HINTS = ("vl", "vision", "llava", "moondream", "minicpm-v")
_SIZE_RE = re.compile(r"(?::|-)(\d+(?:\.\d+)?)b\b", re.IGNORECASE)

_avail: tuple[float, bool, str | None] | None = None  # (checked_at, ok, model)


def _size_b(tag: str) -> float:
    m = _SIZE_RE.search(tag.lower())
    return float(m.group(1)) if m else 0.0


def _rank_text_model(tag: str) -> tuple[int, int, float, str]:
    low = tag.lower()
    family = next((i for i, name in enumerate(_FAMILY_RANK)
                   if low == name or low.startswith(name)), len(_FAMILY_RANK))
    non_vision = 0 if any(h in low for h in _VISION_HINTS) else 1
    return (non_vision, -family, _size_b(low), low)


def _resolve_model(tags: list[str]) -> str | None:
    """The model to use, given what the Ollama server has installed.

    Filters out vision-only models — they accept text prompts but waste their
    multimodal budget and some refuse without images. The VLM provider has its
    own picker; only fall back to a vision model if the user forced it via
    ``CLIPFORGE_LLM_MODEL``.
    """
    if _MODEL:
        return _MODEL  # explicit choice always wins
    text = [t for t in tags if not any(h in t.lower() for h in _VISION_HINTS)]
    return max(text, key=_rank_text_model) if text else None


def _refresh() -> None:
    global _avail
    now = time.time()
    if _avail and now - _avail[0] < 60:
        return
    ok, model = False, None
    try:
        with urllib.request.urlopen(_URL + "/api/tags", timeout=1.5) as r:
            data = json.loads(r.read())
            tags = [m.get("name", "") for m in data.get("models", [])]
            model = _resolve_model([t for t in tags if t])
            ok = r.status == 200 and model is not None
    except Exception:
        ok, model = False, None
    _avail = (now, ok, model)


def available() -> bool:
    """True if an Ollama server answers and has a usable model. Cached 60s."""
    _refresh()
    return _avail[1] if _avail else False


def active_model() -> str | None:
    """The model that will actually write titles, or None."""
    _refresh()
    return _avail[2] if _avail and _avail[1] else None


def _generate(prompt: str, *, timeout: float = 30.0) -> str | None:
    # "think": False — reasoning models (qwen3, deepseek-r1) otherwise spend
    # the whole token budget thinking and return an empty response. Models or
    # Ollama versions that don't know the flag get a retry without it (their
    # inline <think> text, if any, is stripped by _clean_title).
    model = active_model()
    if not model:
        return None
    payload: dict = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0.7, "num_predict": 80},
    }
    for body in (payload, {k: v for k, v in payload.items() if k != "think"}):
        req = urllib.request.Request(_URL + "/api/generate",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read()).get("response", "").strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            if "think" in detail.lower() and "think" in body:
                continue  # flag unsupported here — retry plain
            log.warning("ollama generate failed: %s %s", e, detail[:200])
            return None
        except Exception as e:
            log.warning("ollama generate failed: %s", e)
            return None
    return None


_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def _clean_title(text: str) -> str:
    # Reasoning models (qwen3, deepseek-r1) prepend a <think> block.
    text = _THINK_RE.sub("", text or "").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    text = lines[0] if lines else ""
    text = text.strip().strip('"').strip("'").strip("#").strip()
    # drop a leading "Title:" the model sometimes adds
    for p in ("title:", "hook:", "caption:"):
        if text.lower().startswith(p):
            text = text[len(p):].strip()
    return text[:80]


def suggest_title(transcript_excerpt: str, *, lang: str = "de") -> str | None:
    """Return an LLM-written hook title, or None if unavailable/failed."""
    if not transcript_excerpt.strip() or not available():
        return None
    lang_name = {"de": "German", "en": "English"}.get((lang or "de")[:2].lower(), "the same language")
    prompt = (
        "You write viral short-form video titles. From the transcript below, "
        f"write ONE punchy hook title in {lang_name}, under 60 characters. "
        "No quotes, no hashtags, no emojis, no preamble — just the title.\n\n"
        f"Transcript: {transcript_excerpt[:600]}\n\nTitle:"
    )
    out = _generate(prompt)
    if not out:
        return None
    title = _clean_title(out)
    return title or None


_SCORE_RE = re.compile(r"(\d{1,3})")


def score_viral(transcript_excerpt: str, *, lang: str = "de"
                ) -> tuple[float, str] | None:
    """Ask the local model how viral a clip's content is.

    Returns ``(potential 0..1, one-line reason)`` or None when unavailable /
    unparseable. This is a *second opinion* layered on the transparent heuristic
    scorer — it nudges the rank and adds an explainable reason, never replaces
    the explainable signal sum. Cheap, optional, and fully local.
    """
    if not transcript_excerpt.strip() or not available():
        return None
    lang_name = {"de": "German", "en": "English"}.get((lang or "de")[:2].lower(), "the same language")
    prompt = (
        "You judge short-form video virality. Rate how likely the clip below is "
        "to go viral on TikTok/Reels/Shorts, considering hook strength, emotion, "
        "payoff, and quotability. "
        f"Write the REASON in {lang_name}. Reply with EXACTLY one line:\n"
        "SCORE: <0-100> | REASON: <max 8 words>\n\n"
        f"Transcript: {transcript_excerpt[:600]}\n"
    )
    out = _generate(prompt, timeout=20.0)
    if not out:
        return None
    return _parse_viral(out)


def _parse_viral(text: str) -> tuple[float, str] | None:
    """Parse 'SCORE: 72 | REASON: strong hook' from a model reply."""
    text = _THINK_RE.sub("", text or "").strip()
    line = next((ln for ln in text.splitlines() if "score" in ln.lower()), text)
    m = _SCORE_RE.search(line)
    if not m:
        return None
    val = max(0.0, min(100.0, float(m.group(1)))) / 100.0
    reason = ""
    if "reason" in line.lower():
        reason = line[line.lower().index("reason") + len("reason"):].lstrip(": ").strip()
    reason = reason.strip(' "\'.|')[:48]
    return val, (reason or "AI virality read")


def score_virals(excerpts: list[str], *, lang: str = "de",
                 budget: float = 30.0) -> dict[int, tuple[float, str]]:
    """LLM virality reads for many clips concurrently, capped by a time budget.

    Returns {index: (potential, reason)} for whatever finished in time; the rest
    keep their heuristic score untouched — a slow model can never stall a run."""
    import concurrent.futures as cf

    if not available():
        return {}
    out: dict[int, tuple[float, str]] = {}
    ex = cf.ThreadPoolExecutor(max_workers=3)
    futs = {ex.submit(score_viral, text, lang=lang): i
            for i, text in enumerate(excerpts) if text.strip()}
    done, _ = cf.wait(futs, timeout=budget)
    for f in done:
        try:
            r = f.result()
            if r is not None:
                out[futs[f]] = r
        except Exception:
            pass
    ex.shutdown(wait=False, cancel_futures=True)
    return out


def suggest_titles(excerpts: list[str], *, lang: str = "de",
                   budget: float = 45.0) -> dict[int, str]:
    """Titles for many clips concurrently, capped by an overall time budget.

    Returns {index: title} for whatever finished in time; callers keep their
    heuristic titles for the rest — a slow local model can never stall a run.
    """
    import concurrent.futures as cf

    if not available():
        return {}
    out: dict[int, str] = {}
    ex = cf.ThreadPoolExecutor(max_workers=3)
    futs = {ex.submit(suggest_title, text, lang=lang): i
            for i, text in enumerate(excerpts) if text.strip()}
    done, not_done = cf.wait(futs, timeout=budget)
    for f in done:
        try:
            t = f.result()
            if t:
                out[futs[f]] = t
        except Exception:
            pass
    # Don't join the in-flight requests — a `with` block would block here for
    # up to another request-timeout per worker, blowing through the budget.
    # The stragglers are side-effect-free; let them finish in the background.
    ex.shutdown(wait=False, cancel_futures=True)
    if len(out) < len(excerpts):
        log.info("llm titles: %d/%d within budget", len(out), len(excerpts))
    return out

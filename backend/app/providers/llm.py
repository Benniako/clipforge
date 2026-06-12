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

# Auto-pick order when CLIPFORGE_LLM_MODEL isn't set — strongest commonly-run
# local models first; anything installed beats heuristic titles.
_PREFERRED = ("qwen3:14b", "qwen3:8b", "qwen3", "llama3.1:8b", "llama3.1",
              "gemma3", "qwen2.5", "mistral", "llama3.2")

_avail: tuple[float, bool, str | None] | None = None  # (checked_at, ok, model)


def _resolve_model(tags: list[str]) -> str | None:
    """The model to use, given what the Ollama server has installed."""
    if _MODEL:
        return _MODEL  # explicit choice always wins
    for pref in _PREFERRED:
        for t in tags:
            if t == pref or t.startswith(pref + ":") or t.startswith(pref):
                return t
    return tags[0] if tags else None


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


def suggest_title(transcript_excerpt: str, *, lang: str = "en") -> str | None:
    """Return an LLM-written hook title, or None if unavailable/failed."""
    if not transcript_excerpt.strip() or not available():
        return None
    lang_name = {"de": "German", "en": "English"}.get((lang or "en")[:2], "the same language")
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


def suggest_titles(excerpts: list[str], *, lang: str = "en",
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

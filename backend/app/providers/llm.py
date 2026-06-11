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
import time
import urllib.request

log = logging.getLogger("clipforge.llm")

_URL = os.environ.get("CLIPFORGE_OLLAMA_URL", "http://localhost:11434").rstrip("/")
_MODEL = os.environ.get("CLIPFORGE_LLM_MODEL", "llama3.2")

_avail: tuple[float, bool] | None = None  # (checked_at, ok) — cached briefly


def available() -> bool:
    """True if an Ollama server answers. Cached for 60s."""
    global _avail
    now = time.time()
    if _avail and now - _avail[0] < 60:
        return _avail[1]
    ok = False
    try:
        with urllib.request.urlopen(_URL + "/api/tags", timeout=1.5) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    _avail = (now, ok)
    return ok


def _generate(prompt: str, *, timeout: float = 30.0) -> str | None:
    body = json.dumps({
        "model": _MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.7, "num_predict": 40},
    }).encode()
    req = urllib.request.Request(_URL + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "").strip()
    except Exception as e:
        log.warning("ollama generate failed: %s", e)
        return None


def _clean_title(text: str) -> str:
    text = text.strip().splitlines()[0] if text else ""
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

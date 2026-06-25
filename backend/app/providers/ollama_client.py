"""Shared building blocks for the optional Ollama (local LLM/VLM) providers.

`llm.py` (text) and `vlm.py` (vision) used to each carry a near-identical copy
of: the availability cache, the ``/api/tags`` probe, the ``think: False`` retry
loop, the budget-capped concurrent batch runner, and the ``SCORE: n | REASON:``
parser. That duplication drifted and was a maintenance hazard. Everything that
is genuinely model-agnostic lives here; each provider keeps only what makes it
different (its model picker, its prompt text, its output cleaning).

Everything stays optional and graceful: no Ollama server reachable ⇒ the
availability cache reports False and every caller keeps its heuristic fallback.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from typing import Any

log = logging.getLogger("clipforge.ollama")

# Default Ollama base URL; overridable via the same env var both providers used.
DEFAULT_URL = "http://localhost:11434"

# Shared regexes used by both providers' output cleaning.
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
SCORE_RE = re.compile(r"(\d{1,3})")
_SIZE_RE = re.compile(r"(?::|-)(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def size_b(tag: str) -> float:
    """Extract a model's parameter size (e.g. ``qwen3:14b`` → 14.0)."""
    m = _SIZE_RE.search(tag.lower())
    return float(m.group(1)) if m else 0.0


class AvailabilityCache:
    """Caches the ``/api/tags`` probe for ``ttl`` seconds.

    Holds ``(checked_at, ok, model)``. ``model`` is whatever ``resolver`` the
    provider passes in returns from the installed tag list — so the text and
    vision providers can each keep their own picker while sharing the caching,
    HTTP, and parsing mechanics.
    """

    def __init__(self, *, ttl: float = 60.0, resolver=None):
        self.ttl = ttl
        self._resolver = resolver
        self._state: tuple[float, bool, str | None] | None = None

    def _probe(self) -> tuple[bool, str | None]:
        url = _ollama_url()
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=1.5) as r:
                data = json.loads(r.read())
                tags = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                model = self._resolver(tags) if self._resolver else None
                return r.status == 200 and model is not None, model
        except Exception:
            return False, None

    def refresh(self) -> None:
        now = time.time()
        if self._state and now - self._state[0] < self.ttl:
            return
        ok, model = self._probe()
        self._state = (now, ok, model)

    def available(self) -> bool:
        self.refresh()
        return self._state[1] if self._state else False

    def active_model(self) -> str | None:
        self.refresh()
        return self._state[2] if self._state and self._state[1] else None


def _ollama_url() -> str:
    return (os.environ.get("CLIPFORGE_OLLAMA_URL", DEFAULT_URL) or DEFAULT_URL).rstrip("/")


def generate(*, model: str, prompt: str, timeout: float = 30.0,
             images: list[str] | None = None,
             temperature: float = 0.7, num_predict: int = 80) -> str | None:
    """One-shot ``/api/generate`` call with the ``think: False`` retry loop.

    Reasoning models (qwen3, deepseek-r1) spend the whole budget "thinking" and
    return empty unless ``think: False`` is sent. Older Ollama/server builds
    reject the unknown flag with an HTTP error mentioning "think" — in that case
    we retry once without it. Any other failure is logged and returns None.
    """
    if not model:
        return None
    payload: dict = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if images:
        payload["images"] = images
    url = _ollama_url() + "/api/generate"
    for body in (payload, {k: v for k, v in payload.items() if k != "think"}):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
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


def parse_score_reason(text: str, *, max_reason: int = 48,
                       clamp_reason_below: float | None = None,
                       negative_terms: tuple[str, ...] = ()) -> tuple[float, str] | None:
    """Parse a ``SCORE: <0-100> | REASON: <text>`` reply into ``(0..1, reason)``.

    - Strips any ``<think>`` block.
    - Clamps the score to 0..100 then scales to 0..1.
    - If ``clamp_reason_below`` is set and the reason mentions any
      ``negative_terms``, the score is capped at that value (the VLM uses this
      to force menu/lobby reads low).
    """
    text = THINK_RE.sub("", text or "").strip()
    line = next((ln for ln in text.splitlines() if "score" in ln.lower()), text)
    m = SCORE_RE.search(line)
    if not m:
        return None
    val = max(0.0, min(100.0, float(m.group(1)))) / 100.0
    reason = ""
    if "reason:" in line.lower():
        idx = line.lower().index("reason:")
        reason = line[idx + len("reason:"):].lstrip(": ").strip()
    reason = reason.strip(' "\'.|')[:max_reason]
    if clamp_reason_below is not None and negative_terms:
        if any(t in reason.lower() for t in negative_terms):
            val = min(val, clamp_reason_below)
    return val, (reason or "AI read")


def run_budgeted(fn, items, *, budget: float, max_workers: int = 3,
                 logger=None) -> tuple[dict[int, Any], int]:
    """Run ``fn(item)`` over ``items`` concurrently, capped by a time budget.

    Returns ``(results, timed_out)`` where ``results`` is ``{index: result}``
    for whatever finished in time and returned a truthy result, and
    ``timed_out`` is the count of items that didn't finish within budget.
    In-flight stragglers are abandoned (side-effect-free) so a slow local
    model can never stall a run. Shared by the text/vlm batch calls.
    """
    import concurrent.futures as cf

    out: dict[int, Any] = {}
    if not items:
        return out, 0
    ex = cf.ThreadPoolExecutor(max_workers=max(1, max_workers))
    futs = {ex.submit(fn, item): i for i, item in enumerate(items)}
    done, not_done = cf.wait(futs, timeout=budget)
    for f in done:
        try:
            r = f.result()
            if r:
                out[futs[f]] = r
        except Exception:
            pass
    ex.shutdown(wait=False, cancel_futures=True)
    timed_out = len(not_done)
    if timed_out > 0 and logger:
        logger.info("ollama batch: %d/%d within budget (%d timed out)",
                     len(out), len(items), timed_out)
    return out, timed_out

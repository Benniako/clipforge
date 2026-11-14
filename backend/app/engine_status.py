"""Why an installed engine is not working, for the diagnostics panel.

The capability report says what is installed. An installed engine can still
fail when it starts: a model download refused, a server that does not answer,
a model that does not fit. Providers report those failures here under the
capability key the panel uses ("emotion", "ollama", "panns", "clap", ...), and
clear them once the engine works, so the panel can say what is wrong instead
of showing "available" for something that never ran.
"""
from __future__ import annotations

import threading

_LOCK = threading.Lock()
_PROBLEMS: dict[str, str] = {}
MAX_LEN = 240


def report(key: str, message: str) -> None:
    """Record why ``key`` is not working (one line, shortened)."""
    text = " ".join(str(message).split())
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN - 1] + "…"
    with _LOCK:
        _PROBLEMS[key] = text


def clear(key: str) -> None:
    """``key`` works (again)."""
    with _LOCK:
        _PROBLEMS.pop(key, None)


def problems() -> dict[str, str]:
    with _LOCK:
        return dict(_PROBLEMS)

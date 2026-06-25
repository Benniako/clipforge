"""Optional text-to-speech / voice synthesis for post-generated clips.

The existing pipeline produces captioned video clips. This provider adds
the option to generate an audio narration track via a local TTS engine
(voicebox, Coqui XTTS, Piper, etc.) for accessibility, narration-over-
gameplay, or automated voiceover.

Fully optional: no TTS engine installed ⇒ the pipeline runs unchanged.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("clipforge.tts")


def detected() -> bool:
    """True when a compatible TTS backend is reachable.

    Checks for:
    1. ``piper`` binary on PATH (fast CPU inference, many voices).
    2. Voicebox API endpoint at default port.
    3. ``tts`` CLI from Coqui XTTS.
    """
    if shutil.which("piper"):
        return True
    if shutil.which("tts"):
        return True
    try:
        import urllib.request
        url = os.environ.get("CLIPFORGE_VOICEBOX_URL", "http://127.0.0.1:7760")
        with urllib.request.urlopen(url + "/health", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        pass
    return False


def synthesize(text: str, *, voice: str = "default",
               dst: str | Path | None = None,
               lang: str = "de") -> Path | None:
    """Generate speech audio from ``text`` and save to ``dst``.

    Returns the path to the generated WAV/MP3, or None on failure.
    Currently a no-op stub.
    """
    return None

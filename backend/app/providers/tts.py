"""Optional text-to-speech / voice synthesis for post-generated clips.

The existing pipeline produces captioned video clips. This provider adds
the option to generate an audio narration track via a local TTS engine
(voicebox, Coqui XTTS, Piper, etc.) for accessibility, narration-over-
gameplay, or automated voiceover.

Fully optional: no TTS engine installed ⇒ the pipeline runs unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("clipforge.tts")


def detected() -> bool:
    """True when a compatible TTS backend is reachable.

    Currently a stub. Detection would:
    1. Probe for voicebox API endpoint / CLI
    2. Or check for piper binary + voice model
    3. Or check for xtts checkpoints
    """
    return False


def synthesize(text: str, *, voice: str = "default",
               dst: str | Path | None = None,
               lang: str = "de") -> Path | None:
    """Generate speech audio from ``text`` and save to ``dst``.

    Returns the path to the generated WAV/MP3, or None on failure.
    Currently a no-op stub.
    """
    return None

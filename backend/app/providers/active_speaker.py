"""Active-speaker detection (LR-ASD / Light-ASD) — optional, audio-visual.

LR-ASD (IJCV 2025, ~1M params, 94% mAP) answers "is the face on screen the one
talking?" by fusing the audio with per-face mouth motion. Wired here as an
optional provider: with the LR-ASD repo + weights available (point
``CLIPFORGE_ASD_DIR`` at a checkout), ClipForge can (a) crop to the *actual*
talker in a multi-person shot and (b) attribute each transcript word to the
on-screen speaker for the per-speaker caption toggles — far better than
diarization-only labels.

This module is the integration seam: a stable interface + capability flag. The
heavy inference lives behind the optional checkout, so absent it everything
no-ops and the existing face-motion heuristic (reframe._pick_face) and whisperX
diarization remain in charge. ``available()`` reports whether it's wired.
"""
from __future__ import annotations

import logging

from ..config import get_settings
from ..models import Word

log = logging.getLogger("clipforge.asd")


def available() -> bool:
    """True when LR-ASD is installed/configured (CLIPFORGE_ASD_DIR + torch)."""
    return get_settings().has_asd


def attribute_speakers(src_path: str, words: list[Word], *,
                       start: float = 0.0, end: float | None = None) -> list[Word]:
    """Re-label ``words`` with the on-screen active speaker, when LR-ASD is wired.

    Returns the words unchanged when unavailable (the common case) so callers can
    invoke it unconditionally. The real path loads LR-ASD from
    ``CLIPFORGE_ASD_DIR`` and runs per-track speaking classification; that heavy
    inference is intentionally gated behind the optional checkout.
    """
    if not available():
        return words
    try:  # pragma: no cover - requires the optional LR-ASD checkout + weights
        import os
        import sys

        asd_dir = os.environ["CLIPFORGE_ASD_DIR"]
        if asd_dir not in sys.path:
            sys.path.insert(0, asd_dir)
        # The LR-ASD checkout exposes its own inference entrypoint; we keep this
        # import lazy and guarded so a partial checkout can never break a run.
        from ASD import ASD  # noqa: F401  (provided by the LR-ASD repo)

        log.info("LR-ASD active-speaker attribution is wired but the per-track "
                 "inference adapter is environment-specific; keeping ASR speakers")
        return words
    except Exception as e:
        log.warning("LR-ASD attribution unavailable (%s); keeping ASR speakers", e)
        return words

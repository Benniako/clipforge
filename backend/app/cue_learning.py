"""Learn reusable audio cues from on-screen (OCR) events.

The insight the user asked for: most games signal a moment *both* visually and
audibly — a Valorant kill flashes a banner **and** plays a ding; an EA FC goal
shows the score tick up **and** roars. OCR is the more reliable detector but
needs a vision model installed and a frame grab per check. So when OCR pins an
event, we snip the audio right there and **save it as an audio cue** under
``<data>/game_cues/<profile>/auto_<label>.wav``.

From then on the cheap, dependency-free audio matcher (detect_cues.py) catches
that same event on *future* videos — even with OCR turned off. The visual cue
bootstraps the audio cue library, and it persists across runs and projects.

Saved cues are prefixed ``auto_`` so they never clobber a user-installed cue,
and only one is kept per label. Silence is rejected (a cue that matches nothing
loud would fire everywhere).
"""
from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path

from . import game_packs
from .media import ffmpeg
from .providers.detect_cues import CUE_EXTS
from .providers.detect_gameplay import _cue_dir

log = logging.getLogger("clipforge.cue_learning")


def _scan_dir(profile: str) -> str:
    """The cue folder the detector actually scans for this profile (auto→generic,
    cs→cs2, fifa→eafc) — so a learned cue is found again on the next run."""
    return _cue_dir(profile)

SNIPPET_PRE = 0.25      # seconds of audio before the event timestamp
SNIPPET_LEN = 0.7       # total snippet length
MIN_RMS = 0.02          # reject near-silent snippets (would be a useless cue)
AUTO_PREFIX = "auto_"


def existing_cue_labels(profile: str) -> set[str]:
    """Labels already covered by a cue file for this profile (user or auto)."""
    d = game_packs._dir(_scan_dir(profile))
    if not d.is_dir():
        return set()
    out: set[str] = set()
    for p in d.glob("*"):
        if p.suffix.lower() in CUE_EXTS:
            stem = p.stem.lower()
            out.add(stem[len(AUTO_PREFIX):] if stem.startswith(AUTO_PREFIX) else stem)
    return out


def pending_labels(ocr_events: list, existing: set[str]) -> list[str]:
    """First-seen OCR labels not already backed by a cue — one per label."""
    seen: list[str] = []
    for e in sorted(ocr_events, key=lambda e: e.t):
        label = (getattr(e, "label", None) or "").lower()
        if label and label not in existing and label not in seen:
            seen.append(label)
    return seen


def _snippet_rms(wav_path: str) -> float:
    """RMS amplitude (0..1) of a small wav — used to reject silent snippets."""
    try:
        import numpy as np
    except Exception:
        return 1.0  # can't measure -> don't block saving
    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    if not raw:
        return 0.0
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt((data * data).mean() + 1e-12)) if data.size else 0.0


def save_audio_cues_from_ocr(src_path: str, ocr_events: list, profile: str) -> list[str]:
    """Snip and install an audio cue for each new on-screen event label.

    Returns the labels that were saved. Best-effort and fully guarded — a
    failure here never affects the clips that were already produced."""
    if not ocr_events:
        return []
    # Map first-seen timestamp per label.
    first_t: dict[str, float] = {}
    for e in sorted(ocr_events, key=lambda e: e.t):
        label = (getattr(e, "label", None) or "").lower()
        if label and label not in first_t:
            first_t[label] = float(e.t)

    todo = pending_labels(ocr_events, existing_cue_labels(profile))
    saved: list[str] = []
    for label in todo:
        t = first_t.get(label)
        if t is None:
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                snip = Path(tmp) / "snip.wav"
                ffmpeg.run(["-ss", f"{max(t - SNIPPET_PRE, 0.0):.3f}",
                            "-i", src_path, "-t", f"{SNIPPET_LEN:.3f}",
                            "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(snip)], timeout=30)
                if _snippet_rms(str(snip)) < MIN_RMS:
                    log.info("cue-learn: '%s' snippet too quiet — skipped", label)
                    continue
                game_packs.install_cue(_scan_dir(profile), f"{AUTO_PREFIX}{label}", str(snip))
                saved.append(label)
                log.info("cue-learn: saved audio cue for '%s' @ %.1fs (%s)",
                         label, t, profile)
        except Exception as e:
            log.warning("cue-learn: could not save '%s': %s", label, e)
    return saved

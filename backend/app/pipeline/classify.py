"""Auto-detect whether a source is *talking* footage or *gameplay*.

Talking footage (podcasts, interviews, talks) has a face on screen much of the
time and dense, continuous speech — its best moments live in the transcript.
Gameplay (Valorant, EA FC, …) usually has little/no face and sparse speech — its
best moments are audio-energy spikes and on-screen events, not words.

We decide from two cheap signals:
  • face-presence ratio  — fraction of sampled frames with a detectable face
  • speech coverage      — fraction of the timeline covered by spoken words

The pipeline uses the result to pick the matching detector. A manual override
(ImportSettings.content_type) skips this entirely.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg
from ..media.ffmpeg import MediaInfo
from ..models import Transcript

log = logging.getLogger("clipforge.classify")

SAMPLE_FRAMES = 16


def face_presence_ratio(src_path: str, duration: float) -> float | None:
    """Fraction of sampled frames containing at least one face (None if no cv2).

    Uses N independent fast seeks (keyframe-accurate is fine here) rather than an
    ``fps=`` filter — the filter would force decoding the *entire* source, which
    on an hour-long video takes minutes for 16 sample frames.
    """
    s = get_settings()
    if not s.has_opencv or duration <= 0:
        return None
    try:
        import cv2
    except Exception:
        return None

    from ..media import faces as faces_mod

    hits = 0
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(SAMPLE_FRAMES):
            t = duration * (i + 0.5) / SAMPLE_FRAMES
            fp = Path(tmp) / f"f_{i:03d}.jpg"
            try:
                ffmpeg.grab_frame(src_path, fp, t=t, width=480, quality=5)
            except Exception:
                continue
            img = cv2.imread(str(fp))
            if img is None:
                continue
            total += 1
            hits += 1 if faces_mod.detect_faces(img, min_size_frac=0.08) else 0
    if total == 0:
        return None
    return hits / total


def speech_coverage(transcript: Transcript | None, duration: float) -> float:
    if not transcript or not transcript.words or duration <= 0:
        return 0.0
    spoken = sum(w.d for w in transcript.words)
    return min(spoken / duration, 1.0)


def detect_content_type(src_path: str, info: MediaInfo,
                        transcript: Transcript | None) -> tuple[str, dict]:
    """Return ("talking"|"gameplay", metrics)."""
    face = face_presence_ratio(src_path, info.duration)
    # A synthetic transcript means ASR found no real speech — treat as none, or
    # its filler would masquerade as dense narration and skew us to "talking".
    if transcript and transcript.provider == "synthetic":
        speech = 0.0
    else:
        speech = speech_coverage(transcript, info.duration)
    metrics = {"face_ratio": None if face is None else round(face, 3),
               "speech_coverage": round(speech, 3)}

    # Heuristic decision tree. Bias toward "talking" when unsure, since the
    # transcript detector degrades more gracefully than the gameplay one.
    f = face if face is not None else 0.5
    if speech >= 0.5:
        kind = "talking"                      # dense narration (even audio-only)
    elif f >= 0.4 and speech >= 0.2:
        kind = "talking"                      # face on screen + real speech
    elif speech < 0.15 or f < 0.15:
        kind = "gameplay"                     # little speech or no face
    else:
        kind = "talking"
    log.info("content-type=%s (%s)", kind, metrics)
    return kind, metrics

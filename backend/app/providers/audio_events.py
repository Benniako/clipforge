"""Audio event detection via PANNs — score the *sounds* that signal a highlight.

Cheering, laughter, applause, a scream, an explosion, gunfire — these are the
sounds that make a moment go viral, and the transcript can't see them. PANNs
(Pretrained Audio Neural Networks, CNN14 on AudioSet's 527 classes) tags a clip's
audio in one fast forward pass. We sum the probabilities of the "hype" classes
into a 0..1 score and fold it into virality as an explainable factor, naming the
loudest class as the reason ("crowd cheering", "laughter").

Unlike the cue/OCR matchers this needs no per-game reference — it's zero-shot, so
it solves the cold-start problem for a brand-new game or an IRL clip.

Optional and graceful: no ``panns_inference`` ⇒ :func:`event_score` returns None
and scoring is unchanged. The class→score reduction is a pure function, so it's
unit-tested without the model.
"""
from __future__ import annotations

import logging

from ..config import get_settings
from ..models import ScoreFactor

log = logging.getLogger("clipforge.audio_events")

_tagger = None  # cached (AudioTagging, labels) or False

# AudioSet class-name substrings that read as a short-form highlight, grouped by
# the human reason we'll show. Matched case-insensitively against the 527 labels.
HYPE_CLASSES: dict[str, tuple[str, ...]] = {
    "crowd cheering": ("cheering", "crowd", "applause", "clapping"),
    "laughter": ("laughter", "giggle", "chuckle"),
    "excited shouting": ("screaming", "shout", "yell", "children shouting"),
    "explosive action": ("explosion", "gunshot", "gunfire", "machine gun",
                         "artillery", "boom"),
    "impact": ("smash", "crash", "shatter", "glass", "slam"),
}


def available() -> bool:
    return get_settings().has_audio_events


def _load():
    global _tagger
    if _tagger is not None:
        return _tagger or None
    try:
        from panns_inference import AudioTagging
        from panns_inference.config import labels

        device = "cuda" if get_settings().device == "cuda" else "cpu"
        _tagger = (AudioTagging(checkpoint_path=None, device=device), list(labels))
        log.info("PANNs audio tagging loaded (%d classes)", len(labels))
    except Exception as e:
        log.info("PANNs unavailable (%s)", e)
        _tagger = False
    return _tagger or None


def reduce_scores(probs: dict[str, float]) -> tuple[float, str] | None:
    """Collapse a {label: probability} map into (hype 0..1, reason).

    Pure (no model) so it's unit-testable. ``probs`` is the per-class output keyed
    by AudioSet label; we take, per hype group, its strongest member, then combine
    groups so several different cues stack but one alone still reads clearly.
    """
    group_best: dict[str, float] = {}
    for label, p in probs.items():
        low = label.lower()
        for reason, needles in HYPE_CLASSES.items():
            if any(n in low for n in needles):
                group_best[reason] = max(group_best.get(reason, 0.0), float(p))
    if not group_best:
        return None
    top_reason = max(group_best, key=group_best.get)
    # Combine groups so two distinct cues (cheer + laughter) beat one, but cap at
    # 1.0; the dominant group anchors the score.
    combined = 1.0 - 1.0
    for p in group_best.values():
        combined = combined + p - combined * p  # probabilistic OR
    return max(0.0, min(1.0, combined)), top_reason


def event_score(wav_path: str, start: float, end: float) -> tuple[float, str] | None:
    """(hype 0..1, reason) for a clip's audio span via PANNs, or None.

    Reads only the clip's span so it stays cheap per clip."""
    if not available():
        return None
    loaded = _load()
    if loaded is None:
        return None
    tagger, labels = loaded
    try:
        import tempfile
        from pathlib import Path

        import numpy as np

        from ..media import ffmpeg

        with tempfile.TemporaryDirectory() as tmp:
            seg = Path(tmp) / "seg.wav"
            # PANNs expects 32 kHz mono.
            ffmpeg.run(["-ss", f"{max(start, 0):.3f}", "-i", wav_path,
                        "-t", f"{max(end - start, 0.2):.3f}", "-ac", "1",
                        "-ar", "32000", "-c:a", "pcm_s16le", str(seg)], timeout=60)
            import wave

            with wave.open(str(seg), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return None
        clipwise, _ = tagger.inference(audio[None, :])
        probs = {labels[i]: float(clipwise[0][i]) for i in range(len(labels))}
        return reduce_scores(probs)
    except Exception as e:
        log.warning("audio-event scoring failed (%s)", e)
        return None


def apply_event_bonus(score: int, factors: list[ScoreFactor],
                      hype: float, reason: str, *, max_bonus: float = 10.0
                      ) -> tuple[int, list[ScoreFactor]]:
    """Lift a clip whose audio carries a viral sound (cheer/laugh/explosion).

    Positive-only — a quiet clip shouldn't be punished, it just won't get the
    boost. Shown as an explainable factor. Pure."""
    bonus = int(round(max(0.0, min(1.0, hype)) * max_bonus))
    if bonus <= 0:
        return score, factors
    new_score = int(max(1, min(99, score + bonus)))
    return new_score, [ScoreFactor(
        label=reason.capitalize(), weight=float(bonus),
        detail=f"Audio event detection heard {reason} ({int(hype*100)}/100)"),
        *factors]

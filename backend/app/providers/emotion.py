"""Speech-emotion excitement signal via emotion2vec (FunASR) — optional.

Reaction energy is *the* viral driver on short-form: a laugh, a gasp, rage, hype.
The transcript can't see it, but emotion2vec (a self-supervised speech-emotion
foundation model, SOTA on IEMOCAP, multilingual) can. When FunASR + emotion2vec
are installed we score a clip's audio for arousal/positive emotion and feed it in
as an explainable virality factor — never a black box, just one more reason.

Optional and graceful: no FunASR ⇒ :func:`excitement` returns None and scoring is
unchanged. The score-blend is a pure function (tested without the model).
"""
from __future__ import annotations

import logging

from ..config import get_settings
from ..models import ScoreFactor

log = logging.getLogger("clipforge.emotion")

_model = None  # cached FunASR AutoModel, or False

# emotion2vec_plus labels that read as "high-energy / engaging" for short-form.
_HIGH_AROUSAL = {"happy", "angry", "surprised", "fearful", "excited", "disgusted"}


def _load():
    global _model
    if _model is not None:
        return _model or None
    try:
        from funasr import AutoModel

        _model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)
        log.info("emotion2vec loaded")
    except Exception as e:
        log.info("emotion2vec unavailable (%s)", e)
        _model = False
    return _model or None


def excitement(wav_path: str, start: float, end: float) -> float | None:
    """0..1 high-arousal-emotion score for the clip's audio span, or None.

    A loud, emotional delivery (hype, laughter, anger, shock) scores high; flat
    narration scores low. Reads only the clip's span so it's cheap per clip.
    """
    if not get_settings().has_emotion:
        return None
    model = _load()
    if model is None:
        return None
    try:
        import tempfile
        from pathlib import Path

        from ..media import ffmpeg

        with tempfile.TemporaryDirectory() as tmp:
            seg = Path(tmp) / "seg.wav"
            ffmpeg.run(["-ss", f"{max(start, 0):.3f}", "-i", wav_path,
                        "-t", f"{max(end - start, 0.2):.3f}", "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(seg)], timeout=60)
            res = model.generate(str(seg), granularity="utterance",
                                 extract_embedding=False)
        return _arousal_from_result(res)
    except Exception as e:
        log.warning("emotion scoring failed (%s)", e)
        return None


def _arousal_from_result(res) -> float | None:
    """Sum the probabilities of high-arousal labels from a FunASR result."""
    try:
        item = res[0] if isinstance(res, list) and res else res
        labels = [str(l).split("/")[-1].split("_")[-1].lower()
                  for l in item.get("labels", [])]
        scores = item.get("scores", [])
        total = 0.0
        for lab, sc in zip(labels, scores):
            if any(h in lab for h in _HIGH_AROUSAL):
                total += float(sc)
        return max(0.0, min(1.0, total))
    except Exception:
        return None


def apply_excitement_bonus(score: int, factors: list[ScoreFactor], arousal: float,
                           *, max_swing: float = 10.0) -> tuple[int, list[ScoreFactor]]:
    """Blend an excitement read (0..1) into a score: high arousal lifts it, flat
    delivery nudges it down. Bounded and shown as an explainable factor. Pure."""
    delta = round((max(0.0, min(1.0, arousal)) - 0.45) / 0.55 * max_swing)
    delta = int(max(-max_swing, min(max_swing, delta)))
    new_score = int(max(1, min(99, score + delta)))
    if delta != 0:
        factors = [ScoreFactor(
            label=("High-energy delivery" if delta > 0 else "Flat delivery"),
            weight=float(delta),
            detail=f"Speech-emotion arousal {int(arousal*100)}/100 (emotion2vec)"),
            *factors]
    return new_score, factors

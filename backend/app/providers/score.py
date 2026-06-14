"""Virality scoring — a 0-100 number per clip, with the reasons behind it.

The PRD is emphatic that the score must be *explainable*: an unexplained number
erodes trust faster than a wrong one. So the score is a transparent weighted sum
of signal features, and we return the top contributing factors verbatim.

Feature extraction is kept separate from weighting on purpose: the weights are
what the local learning loop personalises (see app/feedback.py). Because we only
ever learn the *weights of named features*, the score — and its reasons — stay
fully explainable even after personalisation.
"""
from __future__ import annotations

from ..models import ImportSettings, Platform, ScoreFactor, Word
from . import signals

# Per-platform default feature weights (sum ~1.0). TikTok/Reels reward a hard
# hook and energy; Shorts reward clarity/payoff; "generic" is balanced.
BASE_WEIGHTS: dict[Platform, dict[str, float]] = {
    Platform.tiktok:  {"hook": 0.34, "emotion": 0.18, "clarity": 0.12, "quote": 0.12, "pace": 0.14, "length": 0.06, "list": 0.04},
    Platform.reels:   {"hook": 0.30, "emotion": 0.20, "clarity": 0.14, "quote": 0.12, "pace": 0.12, "length": 0.07, "list": 0.05},
    Platform.shorts:  {"hook": 0.24, "emotion": 0.16, "clarity": 0.22, "quote": 0.10, "pace": 0.10, "length": 0.08, "list": 0.10},
    Platform.generic: {"hook": 0.28, "emotion": 0.18, "clarity": 0.18, "quote": 0.12, "pace": 0.12, "length": 0.07, "list": 0.05},
}

FEATURE_LABELS = {
    "hook": "hook",
    "emotion": "emotional payoff",
    "clarity": "standalone clarity",
    "quote": "quotability",
    "pace": "pace & energy",
    "length": "length fit",
    "list": "concrete payoff",
}


def base_weights(settings: ImportSettings) -> dict[str, float]:
    return dict(BASE_WEIGHTS.get(settings.platform, BASE_WEIGHTS[Platform.generic]))


def extract_features(words: list[Word], duration: float, settings: ImportSettings,
                     *, lang: str = "en") -> dict[str, tuple[float, str]]:
    """Return {feature: (value 0..1, human reason)} for a candidate span."""
    lex = signals.get_lexicon(lang)
    return {
        "hook": signals.hook_strength(words, lex),
        "emotion": signals.emotional_payoff(words, lex),
        "clarity": signals.standalone_clarity(words, lex),
        "quote": signals.quotability(words, lex),
        "pace": signals.pace_energy(words, duration),
        "length": signals.length_fit(duration, settings.min_len, settings.max_len),
        "list": signals.list_payoff(words, lex),
    }


def score_from_features(feats: dict[str, tuple[float, str]],
                        weights: dict[str, float]) -> tuple[int, list[ScoreFactor]]:
    """Combine features with weights into a 0-100 score + top reasons."""
    total = 0.0
    contributions: list[tuple[float, str, str, float]] = []  # (points, key, reason, value)
    for key, (value, reason) in feats.items():
        points = value * weights.get(key, 0.0) * 100.0
        total += points
        contributions.append((points, key, reason or FEATURE_LABELS.get(key, key), value))

    score = int(round(_calibrate(total)))

    contributions.sort(key=lambda c: c[0], reverse=True)
    factors: list[ScoreFactor] = []
    for points, key, reason, value in contributions:
        if value < 0.25 or points < 4.0:
            continue
        factors.append(ScoreFactor(label=reason, weight=round(points, 1),
                                   detail=f"{FEATURE_LABELS.get(key, key).capitalize()} scored {int(value*100)}/100"))
        if len(factors) >= 4:
            break
    if len(factors) < 2 and contributions:  # always give the user *something*
        for points, key, reason, value in contributions:
            if all(f.label != reason for f in factors):
                factors.append(ScoreFactor(label=reason, weight=round(points, 1),
                                           detail=f"{FEATURE_LABELS.get(key, key).capitalize()} scored {int(value*100)}/100"))
            if len(factors) >= 2:
                break
    return score, factors


def score_clip(words: list[Word], duration: float, settings: ImportSettings,
               *, lang: str = "en", weights: dict[str, float] | None = None
               ) -> tuple[int, list[ScoreFactor], dict[str, float]]:
    """Score a span. Returns (score, factors, feature_values).

    ``weights`` lets the caller pass personalised weights (from the learner);
    otherwise the per-platform defaults are used. ``feature_values`` is returned
    so the clip can carry them for later learning.
    """
    feats = extract_features(words, duration, settings, lang=lang)
    w = weights or base_weights(settings)
    score, factors = score_from_features(feats, w)
    return score, factors, {k: round(v, 4) for k, (v, _) in feats.items()}


def _calibrate(raw_points: float) -> float:
    """Spread raw weighted points across a usable 0-100 range."""
    centered = (raw_points - 45.0) * 1.35 + 52.0
    return max(1.0, min(99.0, centered))


def apply_viral_boost(score: int, factors: list[ScoreFactor], viral: float,
                      reason: str, *, max_swing: float = 12.0
                      ) -> tuple[int, list[ScoreFactor]]:
    """Blend an optional LLM virality read (0..1) into a heuristic score.

    Centred at 0.5 so the model can push a clip up *or* down by at most
    ``max_swing`` points — it refines the ranking without ever overriding the
    transparent signal sum. The adjustment is shown verbatim as a factor, so the
    score stays explainable. Returns the new (score, factors)."""
    delta = round((max(0.0, min(1.0, viral)) - 0.5) * 2.0 * max_swing)
    new_score = int(max(1, min(99, score + delta)))
    if delta != 0:
        label = (reason or "AI virality read")
        factors = [ScoreFactor(label=f"AI: {label}", weight=float(delta),
                               detail=f"Local model rated virality {int(viral*100)}/100"),
                   *factors]
    return new_score, factors

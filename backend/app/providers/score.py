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
    Platform.tiktok:  {"instant_hook": 0.18, "swipe": 0.12, "hook": 0.22, "emotion": 0.16, "clarity": 0.10, "quote": 0.09, "pace": 0.08, "length": 0.03, "list": 0.02},
    Platform.reels:   {"instant_hook": 0.16, "swipe": 0.12, "hook": 0.20, "emotion": 0.18, "clarity": 0.12, "quote": 0.09, "pace": 0.08, "length": 0.03, "list": 0.02},
    Platform.shorts:  {"instant_hook": 0.14, "swipe": 0.15, "hook": 0.18, "emotion": 0.14, "clarity": 0.18, "quote": 0.08, "pace": 0.07, "length": 0.04, "list": 0.02},
    Platform.generic: {"instant_hook": 0.15, "swipe": 0.12, "hook": 0.20, "emotion": 0.16, "clarity": 0.16, "quote": 0.09, "pace": 0.07, "length": 0.03, "list": 0.02},
}

FEATURE_LABELS = {
    "instant_hook": "first 2s hook",
    "swipe": "swipe resistance",
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
        "instant_hook": signals.instant_hook(words, lex),
        "swipe": signals.swipe_resistance(words, duration, lex),
        "hook": signals.hook_strength(words, lex),
        "emotion": signals.emotional_payoff(words, lex),
        "clarity": signals.standalone_clarity(words, lex),
        "quote": signals.quotability(words, lex),
        "pace": signals.pace_energy(words, duration),
        "length": signals.length_fit(duration, settings.min_len, settings.max_len),
        "list": signals.list_payoff(words, lex),
    }


def hook_analysis(words: list[Word], *, lang: str = "en",
                  threshold: float = 0.5) -> dict:
    """Score the opening ~3 seconds and return a verdict + actionable suggestion.

    The first three seconds decide whether a short gets watched or scrolled.
    This isolates that window from the overall score and returns:
    - ``strength``: 0..1 hook power (from instant_hook + first-3s emotion)
    - ``verdict``: "strong" | "ok" | "weak"
    - ``suggestion``: a concrete opener rewrite hint the UI can surface
    - ``first_words``: the actual opening text, for context

    The suggestion is a *template*, not an LLM call — it's deterministic and
    language-aware so it's always available (no Ollama dependency).
    """
    if not words:
        return {"strength": 0.0, "verdict": "weak",
                "suggestion": "", "first_words": ""}
    lex = signals.get_lexicon(lang) if hasattr(signals, "get_lexicon") else signals._EN
    score_val, _ = signals.instant_hook(words, lex)
    start = words[0].t
    head = [w for w in words if w.t - start < 3.0] or words[:6]
    first_words = " ".join(w.text for w in head).strip()
    # Strong = scroll-stopping hook; weak = the opener won't hold attention.
    if score_val >= threshold + 0.2:
        verdict = "strong"
    elif score_val >= threshold:
        verdict = "ok"
    else:
        verdict = "weak"
    # Deterministic suggestion keyed off what's missing in the opener.
    toks = {w.text.lower() for w in head}
    has_question = any(w.text.rstrip("?!,.") in lex.quote for w in head)
    if verdict == "strong":
        suggestion = ""
    elif not first_words:
        suggestion = "Open on speech in the first second — silence kills retention."
    elif not has_question and not (toks & lex.payoff):
        suggestion = ("Lead with a question or a payoff promise in the first "
                      "two seconds (e.g. 'The reason this works…' / 'Nobody tells you…').")
    else:
        suggestion = ("Tighten the opener — cut any warm-up and land the hook "
                      "inside the first 1.5 seconds.")
    return {"strength": round(score_val, 2), "verdict": verdict,
            "suggestion": suggestion, "first_words": first_words[:120]}


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


def apply_replay_bonus(score: int, factors: list[ScoreFactor], words, duration: float,
                       *, lang: str = "en", max_bonus: float = 8.0
                       ) -> tuple[int, list[ScoreFactor]]:
    """Lift clips that end on a clean, loopable button (rewatch signal).

    Only positive — a weak ending shouldn't tank an otherwise strong clip; it
    just won't get the loop boost. Shown as an explainable factor. Pure."""
    val, reason = signals.replay_value(words, duration, signals.get_lexicon(lang))
    bonus = round(val * max_bonus)
    if bonus <= 0:
        return score, factors
    new_score = int(max(1, min(99, score + bonus)))
    return new_score, [ScoreFactor(label="Loopable ending", weight=float(bonus),
                                   detail=reason), *factors]


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

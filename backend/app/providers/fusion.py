"""Cross-modal fusion scorer — replaces additive virality scoring.

Instead of capping individual signals (apply_evidence_caps), reward clips where
MULTIPLE independent signals agree.  A clip with cue=0.6 AND ocr=0.6 AND
reaction=0.6 should score higher than one with just cue=0.95, because
multi-source confirmation is the strongest virality predictor.

The scorer operates on a flat feature dict so it's modality-agnostic: the same
code handles transcript-only clips, audio-only gameplay, and full multimodal
pipeline output.
"""
from __future__ import annotations

from ..models import ImportSettings, ScoreFactor

# --------------------------------------------------------------------------- #
# Default feature weights — bootstrapped from score.py BASE_WEIGHTS but
# extended for the cross-modal features the old scorer never saw.
# --------------------------------------------------------------------------- #
_DEFAULT_WEIGHTS: dict[str, float] = {
    # Transcript / text features (from score.py BASE_WEIGHTS generic)
    "instant_hook": 0.15,
    "hook": 0.07,
    "emotion": 0.06,
    "clarity": 0.06,
    "quote": 0.04,
    "pace": 0.03,
    "length": 0.02,
    "list": 0.02,
    "controversy": 0.03,
    "qa": 0.02,
    # Audio / gameplay features (from detect_gameplay.py)
    "intensity": 0.08,
    "sustain": 0.05,
    "transient": 0.08,
    "spikes": 0.02,
    # Cross-modal evidence
    "cue": 0.06,
    "reaction": 0.06,
    "ocr": 0.06,
    "audio_event": 0.04,
    # AI / vision features
    "excitement": 0.03,
    "vlm_viral": 0.02,
    "llm_viral": 0.02,
    # Narrative
    "narrative_shape": 0.02,
}

# Thresholds for "this signal is meaningfully present"
_SIGNAL_ON: float = 0.40
# How many signals need to be ON for the multi-signal bonus
_CONFIRM_MIN: int = 2
# Bonus points per extra confirming signal beyond the first
_CONFIRM_BONUS_PER: float = 6.0
# Maximum bonus the confirmation mechanism can add
_CONFIRM_MAX: float = 18.0
# Labels for human-readable multi-signal factor descriptions
_MODALITY_LABELS: dict[str, str] = {
    "cue": "game cue",
    "ocr": "on-screen text",
    "reaction": "facecam reaction",
    "audio_event": "audio event",
    "intensity": "audio intensity",
    "transient": "sudden spike",
    "sustain": "sustained energy",
    "instant_hook": "instant hook",
    "hook": "hook",
    "emotion": "emotional charge",
    "vlm_viral": "visual viral cue",
    "llm_viral": "AI virality read",
}


class FusionScorer:
    """Joint-reasoning scorer that replaces the additive score model.

    Usage::

        scorer = FusionScorer()                        # platform defaults
        scorer = FusionScorer.from_weights(my_dict)    # personalised
        score, factors = scorer.score(feature_dict)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or dict(_DEFAULT_WEIGHTS)

    @classmethod
    def from_weights(cls, weights_dict: dict[str, float]) -> FusionScorer:
        """Create a scorer with personalised weights (from the learner)."""
        return cls(weights=dict(weights_dict))

    @classmethod
    def for_platform(cls, settings: ImportSettings) -> FusionScorer:
        """Bootstrap weights from the platform's BASE_WEIGHTS and merge with
        cross-modal defaults."""
        from . import score as _score_mod

        base = _score_mod.base_weights(settings)
        merged = dict(_DEFAULT_WEIGHTS)
        for k, v in base.items():
            merged[k] = v
        return cls(weights=merged)

    # ------------------------------------------------------------------ #
    # Core scoring
    # ------------------------------------------------------------------ #
    def score(self, features: dict[str, float]) -> tuple[int, list[ScoreFactor]]:
        """Compute a 1-99 virality score with explainable factors.

        ``features`` is a flat dict; absent keys are treated as 0.0 so the
        scorer degrades gracefully when modalities are unavailable.
        """
        safe = {k: float(features.get(k, 0.0)) for k in self._weights}
        raw = 0.0
        contributions: list[tuple[float, str, float]] = []

        for key, w in self._weights.items():
            val = safe.get(key, 0.0)
            pts = val * w * 100.0
            raw += pts
            if val >= 0.15:
                contributions.append((pts, key, val))

        # Multi-signal confirmation bonus
        confirm_bonus, confirming = self._confirmation_bonus(safe)
        raw += confirm_bonus

        score = int(round(max(1, min(99, raw + 42.0))))  # centre ~50

        factors = self._build_factors(contributions, confirming, confirm_bonus)
        return score, factors

    # ------------------------------------------------------------------ #
    # Multi-signal confirmation
    # ------------------------------------------------------------------ #
    def _confirmation_bonus(
        self, features: dict[str, float]
    ) -> tuple[float, list[str]]:
        """Reward clips where multiple independent signals agree.

        Instead of capping individual weak signals, we *reward* a clip where
        two or more different modality signals cross the ON threshold — this
        is the core insight over the old additive + cap model.
        """
        on_signals = [
            k for k, v in features.items()
            if v >= _SIGNAL_ON and self._weights.get(k, 0.0) > 0
        ]
        if len(on_signals) < _CONFIRM_MIN:
            return 0.0, []

        n = len(on_signals)
        bonus = min(
            _CONFIRM_MAX,
            _CONFIRM_BONUS_PER * (n - 1),
        )
        return bonus, on_signals

    # ------------------------------------------------------------------ #
    # Explainable factors
    # ------------------------------------------------------------------ #
    def _build_factors(
        self,
        contributions: list[tuple[float, str, float]],
        confirming: list[str],
        confirm_bonus: float,
    ) -> list[ScoreFactor]:
        """Build the top contributing factors for the user-facing score."""
        contributions.sort(key=lambda c: c[0], reverse=True)
        factors: list[ScoreFactor] = []

        # Multi-signal confirmation is the star — show it first
        if confirming and confirm_bonus > 0:
            labels = [_MODALITY_LABELS.get(s, s) for s in confirming[:5]]
            detail = " + ".join(labels)
            if len(confirming) > 5:
                detail += f" + {len(confirming) - 5} more"
            factors.append(ScoreFactor(
                label=f"{len(confirming)} signals confirm",
                weight=round(confirm_bonus, 1),
                detail=detail,
            ))

        for pts, key, val in contributions:
            if val < 0.20 or pts < 3.0:
                continue
            label = _MODALITY_LABELS.get(key, key)
            factors.append(ScoreFactor(
                label=label,
                weight=round(pts, 1),
                detail=f"{label} scored {int(val * 100)}/100",
            ))
            if len(factors) >= 5:
                break

        if not factors and contributions:
            for pts, key, val in contributions:
                if all(f.label != _MODALITY_LABELS.get(key, key) for f in factors):
                    factors.append(ScoreFactor(
                        label=_MODALITY_LABELS.get(key, key),
                        weight=round(pts, 1),
                        detail=f"{_MODALITY_LABELS.get(key, key)} scored {int(val * 100)}/100",
                    ))
                if len(factors) >= 2:
                    break

        return factors

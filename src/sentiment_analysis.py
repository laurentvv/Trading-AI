import logging
from src.enhanced_decision_engine import ModelResult

logger = logging.getLogger(__name__)


def get_sentiment_decision_from_score(sentiment_score: float) -> ModelResult:
    """
    Converts a sentiment score to a trading decision.

    The BUY/SELL thresholds here are already symmetric (+/-0.15). In prod
    (29/05-25/06) this produced 0 SELL over 610 predictions, not because of
    this branching logic, but because the upstream sentiment_score was never
    observed below -0.15 — a data-side skew (news provider selection /
    scoring aggregation) to be investigated separately. Do NOT "fix" it by
    mirroring the threshold; that would mask the upstream bias. See ADR-002.
    """
    if sentiment_score > 0.15:
        signal = "BUY"
    elif sentiment_score < -0.15:
        signal = "SELL"
    else:
        signal = "HOLD"

    # The confidence can be the absolute value of the score, scaled to be between 0.5 and 1.0
    confidence = min(1.0, 0.5 + abs(sentiment_score))

    analysis = f"Sentiment score from news API is {sentiment_score:.2f}."

    return ModelResult(signal=signal, confidence=confidence, reasoning=analysis)

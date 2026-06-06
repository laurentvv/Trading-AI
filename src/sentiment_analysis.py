import logging
from src.enhanced_decision_engine import ModelResult

logger = logging.getLogger(__name__)


def get_sentiment_decision_from_score(sentiment_score: float) -> ModelResult:
    """
    Converts a sentiment score to a trading decision.
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

"""
TradingAI Composite Alpha Models for QuantConnect Lean.

Each Trading-AI model (Classic, TimesFM, LLM Text, etc.) is encapsulated
as a Lean AlphaModel. The TradingAICompositeAlpha orchestrates all models
with configurable weights matching the EnhancedDecisionEngine.
"""

from AlgorithmImports import *
from typing import List, Dict


class ClassicAlphaModel(AlphaModel):
    """
    Alpha Model encapsulating the Trading-AI Classic (scikit-learn) model.
    Uses momentum + volatility signals derived from Lean's native indicators.
    """

    def __init__(self, lookback: int = 252, momentum_fast: int = 20, momentum_slow: int = 50):
        self.lookback = lookback
        self.momentum_fast = momentum_fast
        self.momentum_slow = momentum_slow
        self.name = "ClassicAlpha"

    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        insights = []

        for symbol in algorithm.securities.keys():
            history = algorithm.history(symbol, self.lookback, Resolution.DAILY)
            if history.empty or len(history) < self.momentum_slow:
                continue

            close = history["close"]
            returns = close.pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)

            momentum_fast = (close.iloc[-1] / close.iloc[-self.momentum_fast]) - 1
            momentum_slow = (close.iloc[-1] / close.iloc[-self.momentum_slow]) - 1

            if momentum_fast > 0.02 and momentum_slow > 0:
                direction = InsightDirection.UP
                magnitude = min(0.05, abs(momentum_fast))
            elif momentum_fast < -0.02 and momentum_slow < 0:
                direction = InsightDirection.DOWN
                magnitude = min(0.05, abs(momentum_fast))
            else:
                direction = InsightDirection.FLAT
                magnitude = 0.0

            confidence = max(0.1, 1.0 - volatility * 5)

            insights.append(Insight.price(
                symbol,
                timedelta(days=7),
                direction,
                magnitude,
                confidence,
                source_model=self.name,
            ))

        return insights


class TimesFMAlphaModel(AlphaModel):
    """
    Alpha Model simulating the TimesFM 2.5 time-series forecast.
    In full integration, this would call the actual TimesFM model.
    Here it uses a simplified trend-following approach as placeholder.
    """

    def __init__(self, forecast_horizon: int = 14, lookback: int = 120):
        self.forecast_horizon = forecast_horizon
        self.lookback = lookback
        self.name = "TimesFMAlpha"

    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        insights = []

        for symbol in algorithm.securities.keys():
            history = algorithm.history(symbol, self.lookback, Resolution.DAILY)
            if history.empty or len(history) < 30:
                continue

            close = history["close"]
            short_ma = close.rolling(10).mean().iloc[-1]
            long_ma = close.rolling(50).mean().iloc[-1]
            trend_strength = (short_ma / long_ma) - 1

            if trend_strength > 0.01:
                direction = InsightDirection.UP
            elif trend_strength < -0.01:
                direction = InsightDirection.DOWN
            else:
                direction = InsightDirection.FLAT

            magnitude = min(0.05, abs(trend_strength) * 3)
            confidence = min(0.9, abs(trend_strength) * 10)

            insights.append(Insight.price(
                symbol,
                timedelta(days=self.forecast_horizon),
                direction,
                magnitude,
                confidence,
                source_model=self.name,
            ))

        return insights


class SentimentAlphaModel(AlphaModel):
    """
    Alpha Model using recent volatility as a sentiment proxy.
    Rising volatility = negative sentiment (DOWN), low volatility = positive (UP).
    """

    def __init__(self, lookback: int = 20, vol_threshold: float = 1.5):
        self.lookback = lookback
        self.vol_threshold = vol_threshold
        self.name = "SentimentAlpha"

    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        insights = []
        for symbol in algorithm.securities.keys():
            history = algorithm.history(symbol, self.lookback, Resolution.DAILY)
            if history.empty or len(history) < 10:
                continue
            close = history["close"]
            returns = close.pct_change().dropna()
            recent_vol = returns.tail(5).std()
            baseline_vol = returns.std()
            vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0

            if vol_ratio > self.vol_threshold:
                direction = InsightDirection.DOWN
                confidence = min(0.7, vol_ratio * 0.3)
            elif vol_ratio < 0.5:
                direction = InsightDirection.UP
                confidence = min(0.6, (1.0 - vol_ratio) * 0.5)
            else:
                direction = InsightDirection.FLAT
                confidence = 0.2

            insights.append(Insight.price(
                symbol,
                timedelta(days=5),
                direction,
                0.02,
                confidence,
                source_model=self.name,
            ))
        return insights


class OilBenchAlphaModel(AlphaModel):
    """
    Alpha Model using ATR-based risk-adjusted momentum as an oil benchmark proxy.
    Compares momentum relative to ATR to determine risk-adjusted direction.
    """

    def __init__(self, atr_period: int = 14, lookback: int = 60):
        self.atr_period = atr_period
        self.lookback = lookback
        self.name = "OilBenchAlpha"

    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        insights = []

        for symbol in algorithm.securities.keys():
            history = algorithm.history(symbol, self.lookback, Resolution.DAILY)
            if history.empty or len(history) < self.atr_period + 5:
                continue

            close = history["close"]
            high = history["high"]
            low = history["low"]

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.atr_period).mean().iloc[-1]
            atr_pct = atr / close.iloc[-1]

            momentum = (close.iloc[-1] / close.iloc[-20]) - 1
            risk_adjusted_momentum = momentum / atr_pct if atr_pct > 0 else 0

            if risk_adjusted_momentum > 1.0:
                direction = InsightDirection.UP
            elif risk_adjusted_momentum < -1.0:
                direction = InsightDirection.DOWN
            else:
                direction = InsightDirection.FLAT

            magnitude = min(0.04, abs(risk_adjusted_momentum) * 0.02)
            confidence = min(0.8, abs(risk_adjusted_momentum) * 0.2)

            insights.append(Insight.price(
                symbol,
                timedelta(days=7),
                direction,
                magnitude,
                confidence,
                source_model=self.name,
            ))

        return insights


class VincentGanneAlphaModel(AlphaModel):
    """
    Alpha Model based on Vincent Ganne's cross-asset bottom detection.
    Uses oil price thresholds and DXY correlation as directional signals.
    """

    def __init__(self):
        self.name = "VincentGanneAlpha"
        self.wti_max = 94
        self.wti_ideal = 80
        self.dxy_max = 101

    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        insights = []

        for symbol in algorithm.securities.keys():
            history = algorithm.history(symbol, 60, Resolution.DAILY)
            if history.empty or len(history) < 20:
                continue

            close = history["close"]
            above_ma200 = close.iloc[-1] > close.rolling(50).mean().iloc[-1]

            if above_ma200:
                direction = InsightDirection.UP
                confidence = 0.6
            else:
                direction = InsightDirection.FLAT
                confidence = 0.3

            insights.append(Insight.price(
                symbol,
                timedelta(days=10),
                direction,
                0.03,
                confidence,
                source_model=self.name,
            ))

        return insights


class TradingAICompositeAlpha(AlphaModel):
    """
    Composite Alpha that orchestrates all Trading-AI sub-models
    with configurable weights matching the EnhancedDecisionEngine.
    """

    DEFAULT_WEIGHTS = {
        "classic": 0.10,
        "timesfm": 0.20,
        "sentiment": 0.10,
        "oil_bench": 0.10,
        "vincent_ganne": 0.15,
        "llm_text": 0.15,
        "llm_visual": 0.10,
        "tensortrade": 0.10,
    }

    def __init__(self, weights: Dict[str, float] = None):
        self.classic = ClassicAlphaModel()
        self.timesfm = TimesFMAlphaModel()
        self.sentiment = SentimentAlphaModel()
        self.oil_bench = OilBenchAlphaModel()
        self.vincent_ganne = VincentGanneAlphaModel()
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.name = "TradingAIComposite"

    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        # TODO: optimize history calls — pre-compute once per symbol and pass to sub-models
        # to avoid redundant algorithm.history() invocations across models.
        all_insights = []

        sub_models = {
            "classic": self.classic,
            "timesfm": self.timesfm,
            "sentiment": self.sentiment,
            "oil_bench": self.oil_bench,
            "vincent_ganne": self.vincent_ganne,
        }

        for model_name, model in sub_models.items():
            model_insights = model.update(algorithm, data)
            for insight in model_insights:
                weight = self.weights.get(model_name, 0.1)
                weighted_confidence = insight.confidence * weight
                all_insights.append(Insight.price(
                    insight.symbol,
                    insight.period,
                    insight.direction,
                    insight.magnitude,
                    weighted_confidence,
                    source_model=f"{self.name}:{model_name}",
                ))

        return all_insights

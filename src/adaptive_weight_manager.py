"""
Adaptive Weighting System for Trading AI Models
Dynamically adjusts model weights based on recent performance, market conditions,
and model reliability metrics.
"""

import logging
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3

logger = logging.getLogger(__name__)

# Dead-zone threshold for HOLD "correctness": a HOLD prediction is considered
# correct when the market barely moved (|return_1d| below this value).
HOLD_NEUTRAL_RETURN_THRESHOLD = 0.005

# Soft win-rate penalty (replaces the former hard 45% kill switch, 2026-07-27).
# Models below WIN_RATE_SOFT_FLOOR are fully suppressed (genuinely bad models);
# between FLOOR and CEIL the weight is scaled linearly; at/above CEIL no
# penalty. A hard cliff collapsed the ensemble to ~3 voters on low-volatility
# European ETF markets — the directional dead-zone (above) keeps the metric
# structurally below 45% there, so a step at 45% was the wrong shape. See the
# 2026-07-27 PROD audit in memory-bank/log.md and AGENTS.md §6.3.
WIN_RATE_SOFT_FLOOR = 0.25
WIN_RATE_SOFT_CEIL = 0.50

# Minimum number of RESOLVED predictions before the soft win-rate penalty may
# suppress a model. A win rate computed on a handful of observations is pure
# noise: the 2026-08-20 PROD incident saw ONE round-trip (recorded at
# return_1d=0.0000 by a fill-price bug) push 6 models to a 0-24% win rate and
# zero their weights for the rest of the 30-day run — the ensemble degenerated
# to classic+oil_bench and stopped trading entirely.
WIN_RATE_MIN_SAMPLES = 20

# Horizon de prédiction par modèle (en jours de cotation). Par défaut 1 jour.
# Les modèles multi-jours (ex: TimesFM 3.0 à 5 jours) ne doivent pas être
# évalués sur le bruit 1-jour.
MODEL_HORIZONS = {"timesfm": 5}


def _signal_correct_mask(df: pd.DataFrame, return_col: str = "return_1d") -> pd.Series:
    """Return a boolean mask: was each prediction directionally correct?

    To avoid rewarding blind BUY models in a naturally drifting market,
    a BUY is only considered correct if the return exceeds the HOLD dead-zone.
    Otherwise, HOLD was the better risk-adjusted decision.
    Supports evaluating multi-day horizon models via `return_col` (e.g. return_5d).
    """
    sig = df["signal_predicted"]
    col = return_col if (return_col in df.columns and df[return_col].notnull().any()) else "return_1d"
    ret = df[col]
    return (
        (sig.isin(["BUY", "STRONG_BUY"]) & (ret > HOLD_NEUTRAL_RETURN_THRESHOLD))
        | (sig.isin(["SELL", "STRONG_SELL"]) & (ret < -HOLD_NEUTRAL_RETURN_THRESHOLD))
        | (sig.isin(["HOLD", "NEUTRAL"]) & (ret.abs() <= HOLD_NEUTRAL_RETURN_THRESHOLD))
    )


@dataclass
class ModelPerformance:
    """Performance metrics for individual models"""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    win_rate: float
    avg_return: float
    volatility: float
    max_drawdown: float
    last_updated: datetime
    # Number of predictions the win_rate was computed on. Below
    # WIN_RATE_MIN_SAMPLES the win rate is statistical noise and must not
    # drive weight suppression (2026-08-20 incident).
    n_observations: int = 0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class WeightAdjustment:
    """Weight adjustment recommendation with reasoning"""

    model_weights: Dict[str, float]
    adjustment_reason: str
    confidence: float
    market_regime: str
    performance_period: str

    def to_dict(self) -> dict:
        return {
            "model_weights": self.model_weights,
            "adjustment_reason": self.adjustment_reason,
            "confidence": self.confidence,
            "market_regime": self.market_regime,
            "performance_period": self.performance_period,
        }


class AdaptiveWeightManager:
    """
    Manages adaptive weighting of trading models based on performance
    and market conditions.
    """

    def __init__(
        self,
        db_path: str = "model_performance.db",
        base_weights: Dict[str, float] = None,
        lookback_days: int = 30,
        min_observations: int = 10,
        config: Dict = None,
    ):
        """
        Initialize the adaptive weight manager.

        Args:
            db_path: Path to performance database
            base_weights: Base weights for models
            lookback_days: Days to look back for performance calculation
            min_observations: Minimum observations needed for weight adjustment
            config: Optional configuration dictionary
        """
        from src.config_weights import DEFAULT_BASE_WEIGHTS

        self.db_path = db_path
        self.config = config or {}
        
        # Adaptive weight manager might receive less default weights but we align it
        # with the central config (excluding grebenkov if it was missing here, but it's cleaner to use one source)
        # It's fine to have grebenkov in adaptive manager too, it will just not be updated if no data.
        self.base_weights = base_weights or DEFAULT_BASE_WEIGHTS.copy()
        self.lookback_days = lookback_days
        self.min_observations = min_observations

        # Performance weight factors
        self.performance_factors = {
            "accuracy": 0.2,
            "sharpe_ratio": 0.25,
            "win_rate": 0.2,
            "max_drawdown": 0.15,  # Lower is better
            "volatility": 0.1,  # Lower is better
            "recency": 0.1,  # More recent performance weighted higher
        }

        # Market regime adjustments
        self.regime_adjustments = {
            "trending": {
                "classic": 1.1,
                "llm_text": 0.9,
                "llm_visual": 1.0,
                "sentiment": 0.8,
                "timesfm": 1.2,
            },
            "volatile": {
                "classic": 0.8,
                "llm_text": 0.9,
                "llm_visual": 1.2,
                "sentiment": 1.1,
                "timesfm": 1.0,
            },
            "sideways": {
                "classic": 1.0,
                "llm_text": 1.1,
                "llm_visual": 0.9,
                "sentiment": 1.0,
                "timesfm": 1.0,
            },
            "crisis": {
                "classic": 0.7,
                "llm_text": 1.3,
                "llm_visual": 1.1,
                "sentiment": 1.4,
                "timesfm": 0.5,
            },
        }

        self._init_database()

    def _init_database(self):
        """Initialize performance tracking database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Create performance tracking table with ticker
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        signal_predicted TEXT,
                        actual_outcome INTEGER,
                        return_1d REAL,
                        return_5d REAL,
                        confidence REAL,
                        market_regime TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ticker TEXT DEFAULT 'default'
                    )
                """)

                # Migration: Ensure ticker column exists if table was created previously
                cursor.execute("PRAGMA table_info(model_performance_history)")
                columns = [row[1] for row in cursor.fetchall()]
                if "ticker" not in columns:
                    cursor.execute("ALTER TABLE model_performance_history ADD COLUMN ticker TEXT DEFAULT 'default'")

                # Create aggregated performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance_summary (
                        model_name TEXT PRIMARY KEY,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        sharpe_ratio REAL,
                        win_rate REAL,
                        avg_return REAL,
                        volatility REAL,
                        max_drawdown REAL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
            logger.info("Performance database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def record_model_prediction(
        self,
        date: str,
        model_name: str,
        signal: str,
        confidence: float,
        market_regime: str = "unknown",
        ticker: str = "default",
    ):
        """Record a model's prediction for later performance evaluation.

        Deduplication: In a 30-min scheduler loop, multiple cycles run each day.
        If an unresolved prediction already exists for (date, ticker, model_name),
        update its signal/confidence instead of inserting duplicate rows that distort
        n_observations and statistical validity.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id FROM model_performance_history
                    WHERE date = ? AND model_name = ? AND ticker = ? AND actual_outcome IS NULL
                    ORDER BY id DESC LIMIT 1
                    """,
                    (date, model_name, ticker),
                )
                existing = cursor.fetchone()
                if existing:
                    cursor.execute(
                        """
                        UPDATE model_performance_history
                        SET signal_predicted = ?, confidence = ?, market_regime = ?, created_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (signal, confidence, market_regime, existing[0]),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO model_performance_history
                        (date, model_name, signal_predicted, confidence, market_regime, ticker)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (date, model_name, signal, confidence, market_regime, ticker),
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")

    def update_prediction_outcome(
        self,
        date: str,
        model_name: str,
        actual_outcome: int,
        return_1d: float,
        return_5d: float = None,
    ):
        """Update the actual outcome for a previously recorded prediction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE model_performance_history
                SET actual_outcome = ?, return_1d = ?, return_5d = ?
                WHERE date = ? AND model_name = ?
            """,
                (actual_outcome, return_1d, return_5d, date, model_name),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update outcome: {e}")

    def update_outcomes_for_date(
        self,
        date: str,
        actual_outcome: int,
        return_1d: float,
        return_5d: float = None,
        ticker: str = None,
    ) -> int:
        """Update outcomes for ALL models that have a recorded prediction on this date.
        Uses a single connection. Returns the number of rows updated."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if ticker:
                    cursor.execute(
                        """
                        UPDATE model_performance_history
                        SET actual_outcome = ?, return_1d = ?, return_5d = ?
                        WHERE date = ? AND (ticker = ? OR ticker = 'default') AND actual_outcome IS NULL
                    """,
                        (actual_outcome, return_1d, return_5d, date, ticker),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE model_performance_history
                        SET actual_outcome = ?, return_1d = ?, return_5d = ?
                        WHERE date = ? AND actual_outcome IS NULL
                    """,
                        (actual_outcome, return_1d, return_5d, date),
                    )
                updated = cursor.rowcount
                conn.commit()
                return updated
        except Exception as e:
            logger.error(f"Failed to batch-update outcomes for {date}: {e}")
            return 0

    def calculate_model_performance(
        self, model_name: str, days_back: int = None, ticker: str = None
    ) -> Optional[ModelPerformance]:
        """
        Calculate comprehensive performance metrics for a model.

        Args:
            model_name: Name of the model
            days_back: Days to look back (default: self.lookback_days)
            ticker: Optional ticker symbol to filter by (e.g. SXRV.DE, CRUDP.PA)

        Returns:
            ModelPerformance object or None if insufficient data
        """
        days_back = days_back or self.lookback_days
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            conn = sqlite3.connect(self.db_path)

            # Get recent predictions with outcomes
            query = """
                SELECT signal_predicted, actual_outcome, return_1d, return_5d, confidence
                FROM model_performance_history
                WHERE model_name = ? AND date >= ? AND actual_outcome IS NOT NULL
            """
            params = [model_name, cutoff_date]
            if ticker:
                query += " AND (ticker = ? OR ticker = 'default')"
                params.append(ticker)
            query += " ORDER BY date DESC"

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            ret_col = "return_5d" if (model_name in MODEL_HORIZONS and MODEL_HORIZONS[model_name] == 5) else "return_1d"
            valid_df = df.dropna(subset=[ret_col])

            if len(valid_df) < self.min_observations:
                logger.warning(f"Insufficient data for {model_name}: {len(valid_df)} observations")
                return None

            # Calculate performance metrics
            # Dynamically compute actual outcome using the threshold to avoid legacy DB 0/1 bias
            actual = pd.Series(0, index=valid_df.index)
            actual[valid_df[ret_col] > HOLD_NEUTRAL_RETURN_THRESHOLD] = 1
            actual[valid_df[ret_col] < -HOLD_NEUTRAL_RETURN_THRESHOLD] = -1

            # Convert signals to -1, 0, 1
            signal_map = {"STRONG_SELL": -1, "SELL": -1, "HOLD": 0, "NEUTRAL": 0, "BUY": 1, "STRONG_BUY": 1}
            predicted = valid_df["signal_predicted"].map(signal_map).fillna(0).astype(int)

            # Classification metrics
            accuracy = (predicted == actual).mean()

            # Precision/Recall focus on the BUY signal (1) vs non-BUY
            tp = ((predicted == 1) & (actual == 1)).sum()
            fp = ((predicted == 1) & (actual != 1)).sum()
            fn = ((predicted != 1) & (actual == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Return-based metrics
            returns = valid_df[ret_col].dropna()
            if len(returns) > 0:
                avg_return = returns.mean()
                volatility = returns.std()
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0

                win_rate = float(_signal_correct_mask(valid_df, return_col=ret_col).mean())

                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min())
            else:
                avg_return = 0.0
                volatility = 0.0
                sharpe_ratio = 0.0
                win_rate = 0.0
                max_drawdown = 0.0

            return ModelPerformance(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                avg_return=avg_return,
                volatility=volatility,
                max_drawdown=max_drawdown,
                last_updated=datetime.now(),
                n_observations=int(len(valid_df)),
            )

        except Exception as e:
            logger.error(f"Failed to calculate performance for {model_name}: {e}")
            return None

    def calculate_all_models_performance(
        self, models: list[str], days_back: int = None, ticker: str = None
    ) -> dict[str, Optional[ModelPerformance]]:
        """
        Calculate comprehensive performance metrics for multiple models in a single query.

        Args:
            models: List of model names
            days_back: Days to look back (default: self.lookback_days)
            ticker: Optional ticker symbol to filter by (e.g. SXRV.DE, CRUDP.PA)

        Returns:
            Dictionary mapping model names to ModelPerformance objects (or None if insufficient data)
        """
        days_back = days_back or self.lookback_days
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        results = {model: None for model in models}

        try:
            conn = sqlite3.connect(self.db_path)

            placeholders = ",".join("?" for _ in models)
            query = f"""
                SELECT model_name, signal_predicted, actual_outcome, return_1d, return_5d, confidence
                FROM model_performance_history
                WHERE model_name IN ({placeholders}) AND date >= ? AND actual_outcome IS NOT NULL
            """
            params = list(models) + [cutoff_date]
            if ticker:
                query += " AND (ticker = ? OR ticker = 'default')"
                params.append(ticker)

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if df.empty:
                return results

            for model_name, group in df.groupby("model_name"):
                ret_col = "return_5d" if (model_name in MODEL_HORIZONS and MODEL_HORIZONS[model_name] == 5) else "return_1d"
                valid_group = group.dropna(subset=[ret_col])
                if len(valid_group) < self.min_observations:
                    logger.warning(f"Insufficient data for {model_name}: {len(valid_group)} observations")
                    continue

                # Calculate performance metrics
                # Dynamically compute actual outcome using the threshold to avoid legacy DB 0/1 bias
                actual = pd.Series(0, index=valid_group.index)
                actual[valid_group[ret_col] > HOLD_NEUTRAL_RETURN_THRESHOLD] = 1
                actual[valid_group[ret_col] < -HOLD_NEUTRAL_RETURN_THRESHOLD] = -1

                # Convert signals to -1, 0, 1
                signal_map = {"STRONG_SELL": -1, "SELL": -1, "HOLD": 0, "NEUTRAL": 0, "BUY": 1, "STRONG_BUY": 1}
                predicted = valid_group["signal_predicted"].map(signal_map).fillna(0).astype(int)

                # Classification metrics
                accuracy = (predicted == actual).mean()

                tp = ((predicted == 1) & (actual == 1)).sum()
                fp = ((predicted == 1) & (actual != 1)).sum()
                fn = ((predicted != 1) & (actual == 1)).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                # Return-based metrics
                returns = valid_group[ret_col].dropna()
                if len(returns) > 0:
                    avg_return = returns.mean()
                    volatility = returns.std()
                    sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0

                    # Per-signal win rate: direction-correctness of the signal
                    # (see _signal_correct_mask / ADR-002).
                    win_rate = float(_signal_correct_mask(valid_group, return_col=ret_col).mean())

                    # Maximum drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = abs(drawdown.min())
                else:
                    avg_return = 0.0
                    volatility = 0.0
                    sharpe_ratio = 0.0
                    win_rate = 0.0
                    max_drawdown = 0.0

                results[model_name] = ModelPerformance(
                    model_name=model_name,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    sharpe_ratio=sharpe_ratio,
                    win_rate=win_rate,
                    avg_return=avg_return,
                    volatility=volatility,
                    max_drawdown=max_drawdown,
                    last_updated=datetime.now(),
                    n_observations=int(len(valid_group)),
                )

        except Exception as e:
            logger.error(f"Failed to calculate all models performance: {e}")

        return results

    def detect_market_regime(self, market_data: pd.Series, volatility: float) -> str:
        """
        Detect current market regime.

        Args:
            market_data: Recent price data
            volatility: Current volatility measure — DAILY std (see
                enhanced_trading_example.compute_daily_volatility). The
                thresholds below (0.03 default) are daily-scale: an annualized
                figure (~0.15+) classified every single session as
                "volatile"/"crisis" (GO-gate 4, audit 2026-08-19 C4).

        Returns:
            Market regime string
        """
        if len(market_data) < 20:
            return "unknown"

        # Calculate trend strength
        recent_return = (market_data.iloc[-1] / market_data.iloc[-20]) - 1

        # Volatility thresholds from config
        wm_config = self.config.get("weight_manager", {})
        regime_thresholds = wm_config.get("regime_thresholds", {})
        high_vol_threshold = regime_thresholds.get("high_vol", 0.03)

        # Trend thresholds
        strong_trend_threshold = 0.05

        # Regime classification
        if volatility > high_vol_threshold:
            if abs(recent_return) > strong_trend_threshold:
                return "crisis"  # High volatility + strong move
            else:
                return "volatile"  # High volatility, no strong trend
        elif abs(recent_return) > strong_trend_threshold:
            return "trending"  # Low/medium volatility + strong trend
        else:
            return "sideways"  # Low volatility + weak trend

    def calculate_performance_score(self, performance: ModelPerformance) -> float:
        """
        Calculate weighted performance score for a model.

        Args:
            performance: ModelPerformance object

        Returns:
            Weighted performance score (0-1)
        """
        # Normalize metrics to 0-1 scale
        accuracy_score = performance.accuracy
        sharpe_score = min(1.0, max(0.0, (performance.sharpe_ratio + 1) / 3))  # Normalize Sharpe
        win_rate_score = performance.win_rate

        # Invert negative metrics (lower is better)
        drawdown_score = 1.0 - min(1.0, performance.max_drawdown * 10)  # Scale drawdown
        volatility_score = 1.0 - min(1.0, performance.volatility * 20)  # Scale volatility

        # Recency bonus (placeholder - could be enhanced with time-weighted scoring)
        recency_score = 1.0  # Full score for recent performance

        # Calculate weighted score
        weighted_score = (
            accuracy_score * self.performance_factors["accuracy"]
            + sharpe_score * self.performance_factors["sharpe_ratio"]
            + win_rate_score * self.performance_factors["win_rate"]
            + drawdown_score * self.performance_factors["max_drawdown"]
            + volatility_score * self.performance_factors["volatility"]
            + recency_score * self.performance_factors["recency"]
        )

        return max(0.0, min(1.0, weighted_score))

    def calculate_adaptive_weights(
        self,
        market_data: pd.Series = None,
        volatility: float = None,
        force_update: bool = False,
        ticker: str = None,
    ) -> WeightAdjustment:
        """
        Calculate adaptive weights based on recent model performance.

        Args:
            market_data: Recent market price data
            volatility: Current market volatility
            force_update: Force weight recalculation even with limited data
            ticker: Optional ticker to filter model performance by (e.g. SXRV.DE, CRUDP.PA)

        Returns:
            WeightAdjustment object with new weights and reasoning
        """
        # Detect market regime
        market_regime = "unknown"
        if market_data is not None and volatility is not None:
            market_regime = self.detect_market_regime(market_data, volatility)

        # Calculate performance for each model
        model_performances = {}
        performance_scores = {}

        # Fetch all performances in a single query to fix N+1 issue
        models = list(self.base_weights.keys())
        all_performances = self.calculate_all_models_performance(models, ticker=ticker)

        for model_name in models:
            performance = all_performances.get(model_name)
            if performance is not None:
                model_performances[model_name] = performance
                performance_scores[model_name] = self.calculate_performance_score(performance)
            else:
                # Use base performance if no data available
                performance_scores[model_name] = 0.5  # Neutral score

        # Check if we have enough data for adaptation
        models_with_data = len([p for p in model_performances.values() if p is not None])

        if models_with_data < 2 and not force_update:
            normalized = self.base_weights.copy()
            total_bw = sum(normalized.values())
            if total_bw > 0:
                normalized = {k: v / total_bw for k, v in normalized.items()}
            return WeightAdjustment(
                model_weights=normalized,
                adjustment_reason="Insufficient performance data for adaptation",
                confidence=0.3,
                market_regime=market_regime,
                performance_period=f"{self.lookback_days} days",
            )

        # Calculate performance-based weights
        total_performance = sum(performance_scores.values())
        if total_performance == 0:
            performance_weights = self.base_weights.copy()
        else:
            performance_weights = {model: score / total_performance for model, score in performance_scores.items()}

        # Apply market regime adjustments
        regime_adjusted_weights = {}
        if market_regime in self.regime_adjustments:
            regime_factors = self.regime_adjustments[market_regime]
            for model in self.base_weights.keys():
                base_weight = performance_weights.get(model, self.base_weights[model])
                regime_factor = regime_factors.get(model, 1.0)
                regime_adjusted_weights[model] = base_weight * regime_factor
        else:
            regime_adjusted_weights = performance_weights.copy()

        # Normalize weights to sum to 1.0
        total_weight = sum(regime_adjusted_weights.values())
        if total_weight > 0:
            final_weights = {model: weight / total_weight for model, weight in regime_adjusted_weights.items()}
        else:
            final_weights = self.base_weights.copy()

        # Smooth transition from base weights (avoid dramatic changes)
        smoothing_factor = 0.7  # 70% new weights, 30% base weights
        # Models exempt from performance adaptation: their weight is fixed by
        # design (not driven by trade outcomes). The weekend council is a
        # weekly strategic verdict with no resolvable per-cycle outcome, so
        # letting the adaptive loop rescale it on a neutral 0.5 score would
        # silently drift its 0.10 base weight. Its temporal relevance is
        # already handled by age-decay upstream (get_council_ticker_stance).
        fixed_weight_models = {"council"}
        smoothed_weights = {}
        for model in self.base_weights.keys():
            if model in fixed_weight_models:
                # Keep the configured base weight (no performance blend).
                smoothed_weights[model] = self.base_weights[model]
                continue
            new_weight = final_weights.get(model, self.base_weights[model])
            base_weight = self.base_weights[model]
            smoothed_weights[model] = smoothing_factor * new_weight + (1 - smoothing_factor) * base_weight

        # SOFT win-rate penalty
        smoothed_weights = self._apply_soft_win_rate_penalties(smoothed_weights, all_performances)

        # Generate adjustment reasoning
        reasoning = self._build_adjustment_reasoning(market_regime, performance_scores, smoothed_weights)

        # Calculate confidence based on data quality and consistency
        confidence = min(0.9, 0.3 + (models_with_data / len(self.base_weights)) * 0.6)

        return WeightAdjustment(
            model_weights=smoothed_weights,
            adjustment_reason=reasoning,
            confidence=confidence,
            market_regime=market_regime,
            performance_period=f"{self.lookback_days} days",
        )

    def _apply_soft_win_rate_penalties(
        self,
        weights: dict[str, float],
        all_performances: dict,
    ) -> dict[str, float]:
        """Apply soft continuous win-rate penalty for models below threshold."""
        adjusted = weights.copy()
        for model in self.base_weights.keys():
            perf = all_performances.get(model)
            if perf is not None and perf.win_rate >= 0:
                # Minimum-sample guard (2026-08-20 incident): a win rate over
                # a handful of predictions is noise and must not suppress a
                # model — one mis-recorded round-trip zeroed 6 models for the
                # whole 30-day run.
                n_obs = getattr(perf, "n_observations", 0)
                if n_obs < WIN_RATE_MIN_SAMPLES:
                    logger.debug(
                        f"Win rate {model} ignoré : {n_obs} observation(s) < {WIN_RATE_MIN_SAMPLES} "
                        f"(échantillon insuffisant, pas de pénalité)."
                    )
                    continue
                wr = perf.win_rate
                if wr < WIN_RATE_SOFT_CEIL:
                    factor = max(
                        0.0,
                        (wr - WIN_RATE_SOFT_FLOOR)
                        / (WIN_RATE_SOFT_CEIL - WIN_RATE_SOFT_FLOOR),
                    )
                    adjusted[model] *= factor
                    if factor < 0.1:
                        logger.info(
                            f"Réduction forte de {model} : win_rate {wr:.2%} "
                            f"→ facteur {factor:.2f} (sous le plancher "
                            f"{WIN_RATE_SOFT_FLOOR:.0%})."
                        )
                    else:
                        logger.info(
                            f"Réduction de {model} : win_rate {wr:.2%} "
                            f"→ facteur {factor:.2f}."
                        )

        total_adjusted = sum(adjusted.values())
        if total_adjusted > 0:
            return {k: v / total_adjusted for k, v in adjusted.items()}
        return adjusted

    def _build_adjustment_reasoning(
        self,
        market_regime: str,
        performance_scores: dict[str, float],
        smoothed_weights: dict[str, float],
    ) -> str:
        """Build summary string explaining the weight changes."""
        reasoning_parts = [f"Market regime: {market_regime}"]

        if performance_scores:
            top_model = max(performance_scores.keys(), key=lambda k: performance_scores[k])
            reasoning_parts.append(f"Top performer: {top_model}")

        significant_changes = []
        for model, new_weight in smoothed_weights.items():
            base_weight = self.base_weights[model]
            change = (new_weight - base_weight) / base_weight
            if abs(change) > 0.1:  # 10% change threshold
                direction = "increased" if change > 0 else "decreased"
                significant_changes.append(f"{model} {direction} by {abs(change):.1%}")

        if significant_changes:
            reasoning_parts.extend(significant_changes[:2])

        return "; ".join(reasoning_parts)

    @staticmethod
    def _extract_returns_and_outcomes(price_info):
        """Extract prices and compute 1d/5d returns and outcomes."""
        if isinstance(price_info, dict):
            price_today = price_info.get("today")
            price_next_1d = price_info.get("next_1d")
            price_next_5d = price_info.get("next_5d")
        elif isinstance(price_info, (tuple, list)):
            price_today = price_info[0]
            price_next_1d = price_info[1] if len(price_info) > 1 else None
            price_next_5d = price_info[2] if len(price_info) > 2 else None
        else:
            return None, None, None, None

        if price_today is None or price_today == 0:
            return None, None, None, None

        return_1d = (
            ((price_next_1d - price_today) / price_today)
            if (price_next_1d is not None and price_next_1d > 0)
            else None
        )
        return_5d = (
            ((price_next_5d - price_today) / price_today)
            if (price_next_5d is not None and price_next_5d > 0)
            else None
        )

        if return_1d is not None and return_1d == 0.0 and price_next_1d == price_today:
            return None, None, None, None

        actual_outcome_1d = None
        if return_1d is not None:
            actual_outcome_1d = (
                1 if return_1d > HOLD_NEUTRAL_RETURN_THRESHOLD
                else (-1 if return_1d < -HOLD_NEUTRAL_RETURN_THRESHOLD else 0)
            )

        actual_outcome_5d = None
        if return_5d is not None:
            actual_outcome_5d = (
                1 if return_5d > HOLD_NEUTRAL_RETURN_THRESHOLD
                else (-1 if return_5d < -HOLD_NEUTRAL_RETURN_THRESHOLD else 0)
            )

        return return_1d, return_5d, actual_outcome_1d, actual_outcome_5d

    def _resolve_1d_records(
        self,
        cursor,
        date_key: str,
        outcome_1d: int,
        ret_1d: float,
        ret_5d: float,
        ticker: str = None,
    ) -> int:
        """Update predictions for models with 1-day horizon."""
        if outcome_1d is None:
            return 0
        h1_models = [m for m, h in MODEL_HORIZONS.items() if h != 1]
        h1_placeholders = ",".join("?" for _ in h1_models)
        not_in = f"AND model_name NOT IN ({h1_placeholders})" if h1_models else ""

        if ticker:
            sql = f"""
                UPDATE model_performance_history
                SET actual_outcome = ?, return_1d = ?, return_5d = ?
                WHERE date = ? AND (ticker = ? OR ticker = 'default') AND actual_outcome IS NULL {not_in}
            """
            params = [outcome_1d, ret_1d, ret_5d, date_key, ticker] + h1_models
        else:
            sql = f"""
                UPDATE model_performance_history
                SET actual_outcome = ?, return_1d = ?, return_5d = ?
                WHERE date = ? AND actual_outcome IS NULL {not_in}
            """
            params = [outcome_1d, ret_1d, ret_5d, date_key] + h1_models

        cursor.execute(sql, params)
        return cursor.rowcount

    def _resolve_multi_day_records(
        self,
        cursor,
        date_key: str,
        outcome_5d: int,
        ret_1d: float,
        ret_5d: float,
        ticker: str = None,
    ) -> int:
        """Update predictions for multi-day models (e.g. timesfm 5-day horizon)."""
        count = 0
        for m_name, h_days in MODEL_HORIZONS.items():
            if h_days != 5:
                continue
            if outcome_5d is not None:
                if ticker:
                    sql = """
                        UPDATE model_performance_history
                        SET actual_outcome = ?, return_1d = ?, return_5d = ?
                        WHERE date = ? AND model_name = ? AND (ticker = ? OR ticker = 'default') AND actual_outcome IS NULL
                    """
                    params = [outcome_5d, ret_1d, ret_5d, date_key, m_name, ticker]
                else:
                    sql = """
                        UPDATE model_performance_history
                        SET actual_outcome = ?, return_1d = ?, return_5d = ?
                        WHERE date = ? AND model_name = ? AND actual_outcome IS NULL
                    """
                    params = [outcome_5d, ret_1d, ret_5d, date_key, m_name]
                cursor.execute(sql, params)
                count += cursor.rowcount
            elif ret_1d is not None:
                if ticker:
                    sql = """
                        UPDATE model_performance_history
                        SET return_1d = ?
                        WHERE date = ? AND model_name = ? AND (ticker = ? OR ticker = 'default') AND actual_outcome IS NULL
                    """
                    params = [ret_1d, date_key, m_name, ticker]
                else:
                    sql = """
                        UPDATE model_performance_history
                        SET return_1d = ?
                        WHERE date = ? AND model_name = ? AND actual_outcome IS NULL
                    """
                    params = [ret_1d, date_key, m_name]
                cursor.execute(sql, params)
        return count

    def resolve_previous_predictions(self, dates_prices: dict, ticker: str = None):
        """
        Resolve unresolved predictions (actual_outcome IS NULL) by computing
        the actual returns (1-day and 5-day) from historical prices.

        Args:
            dates_prices: dict of {date_str: (price_today, price_next)} or
                          {date_str: {"today": p0, "next_1d": p1, "next_5d": p5}}
            ticker: Ticker symbol to resolve (e.g. SXRV.DE, CRUDP.PA). If None, resolves all.
        """
        if not dates_prices:
            return 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if ticker:
                    cursor.execute(
                        "SELECT DISTINCT date FROM model_performance_history WHERE (ticker = ? OR ticker = 'default') AND actual_outcome IS NULL",
                        (ticker,),
                    )
                else:
                    cursor.execute("SELECT DISTINCT date FROM model_performance_history WHERE actual_outcome IS NULL")
                unresolved_dates = {row[0] for row in cursor.fetchall()}

                resolved_count = 0
                for date_str, price_info in dates_prices.items():
                    date_key = date_str.strftime("%Y-%m-%d") if hasattr(date_str, "strftime") else str(date_str)[:10]
                    if date_key not in unresolved_dates:
                        continue

                    r_1d, r_5d, out_1d, out_5d = self._extract_returns_and_outcomes(price_info)
                    if r_1d is None and r_5d is None:
                        continue

                    resolved_count += self._resolve_1d_records(cursor, date_key, out_1d, r_1d, r_5d, ticker)
                    resolved_count += self._resolve_multi_day_records(cursor, date_key, out_5d, r_1d, r_5d, ticker)

                conn.commit()
                if resolved_count > 0:
                    logger.info(f"Resolved {resolved_count} predictions for ticker={ticker or 'ALL'}")
                return resolved_count

        except Exception as e:
            logger.error(f"Failed to resolve previous predictions: {e}")
            return 0

    def get_current_weights(
        self, market_data: pd.Series = None, volatility: float = None, ticker: str = None
    ) -> Dict[str, float]:
        """
        Get current recommended weights (convenience method).

        Args:
            market_data: Recent market data
            volatility: Current volatility
            ticker: Optional ticker to filter model performance by

        Returns:
            Dictionary of model weights
        """
        weight_adjustment = self.calculate_adaptive_weights(market_data, volatility, ticker=ticker)
        return weight_adjustment.model_weights

"""
Adaptive Weighting System for Trading AI Models
Dynamically adjusts model weights based on recent performance, market conditions,
and model reliability metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

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
    
    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'avg_return': self.avg_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'last_updated': self.last_updated.isoformat()
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
            'model_weights': self.model_weights,
            'adjustment_reason': self.adjustment_reason,
            'confidence': self.confidence,
            'market_regime': self.market_regime,
            'performance_period': self.performance_period
        }

class AdaptiveWeightManager:
    """
    Manages adaptive weighting of trading models based on performance
    and market conditions.
    """
    
    def __init__(self, 
                 db_path: str = "model_performance.db",
                 base_weights: Dict[str, float] = None,
                 lookback_days: int = 30,
                 min_observations: int = 10):
        """
        Initialize the adaptive weight manager.
        
        Args:
            db_path: Path to performance database
            base_weights: Base weights for models
            lookback_days: Days to look back for performance calculation
            min_observations: Minimum observations needed for weight adjustment
        """
        self.db_path = db_path
        self.base_weights = base_weights or {
            'classic': 0.35,
            'llm_text': 0.25,
            'llm_visual': 0.25,
            'sentiment': 0.15
        }
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        
        # Performance weight factors
        self.performance_factors = {
            'accuracy': 0.2,
            'sharpe_ratio': 0.25,
            'win_rate': 0.2,
            'max_drawdown': 0.15,  # Lower is better
            'volatility': 0.1,     # Lower is better
            'recency': 0.1         # More recent performance weighted higher
        }
        
        # Market regime adjustments
        self.regime_adjustments = {
            'trending': {'classic': 1.1, 'llm_text': 0.9, 'llm_visual': 1.0, 'sentiment': 0.8},
            'volatile': {'classic': 0.8, 'llm_text': 0.9, 'llm_visual': 1.2, 'sentiment': 1.1},
            'sideways': {'classic': 1.0, 'llm_text': 1.1, 'llm_visual': 0.9, 'sentiment': 1.0},
            'crisis': {'classic': 0.7, 'llm_text': 1.3, 'llm_visual': 1.1, 'sentiment': 1.4}
        }
        
        self._init_database()
    
    def _init_database(self):
        """Initialize performance tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create performance tracking table
            cursor.execute('''
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create aggregated performance table
            cursor.execute('''
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
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Performance database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def record_model_prediction(self, 
                              date: str,
                              model_name: str,
                              signal: str,
                              confidence: float,
                              market_regime: str = 'unknown'):
        """Record a model's prediction for later performance evaluation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance_history 
                (date, model_name, signal_predicted, confidence, market_regime)
                VALUES (?, ?, ?, ?, ?)
            ''', (date, model_name, signal, confidence, market_regime))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")
    
    def update_prediction_outcome(self,
                                date: str,
                                model_name: str,
                                actual_outcome: int,
                                return_1d: float,
                                return_5d: float = None):
        """Update the actual outcome for a previously recorded prediction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE model_performance_history 
                SET actual_outcome = ?, return_1d = ?, return_5d = ?
                WHERE date = ? AND model_name = ?
            ''', (actual_outcome, return_1d, return_5d, date, model_name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update outcome: {e}")
    
    def calculate_model_performance(self, 
                                  model_name: str,
                                  days_back: int = None) -> Optional[ModelPerformance]:
        """
        Calculate comprehensive performance metrics for a model.
        
        Args:
            model_name: Name of the model
            days_back: Days to look back (default: self.lookback_days)
            
        Returns:
            ModelPerformance object or None if insufficient data
        """
        days_back = days_back or self.lookback_days
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent predictions with outcomes
            query = '''
                SELECT signal_predicted, actual_outcome, return_1d, return_5d, confidence
                FROM model_performance_history 
                WHERE model_name = ? AND date >= ? AND actual_outcome IS NOT NULL
                ORDER BY date DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(model_name, cutoff_date))
            conn.close()
            
            if len(df) < self.min_observations:
                logger.warning(f"Insufficient data for {model_name}: {len(df)} observations")
                return None
            
            # Calculate performance metrics
            # Convert signals to binary (1 for BUY/STRONG_BUY, 0 for others)
            predicted = (df['signal_predicted'].isin(['BUY', 'STRONG_BUY'])).astype(int)
            actual = df['actual_outcome']
            
            # Classification metrics
            accuracy = (predicted == actual).mean()
            
            tp = ((predicted == 1) & (actual == 1)).sum()
            fp = ((predicted == 1) & (actual == 0)).sum()
            fn = ((predicted == 0) & (actual == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Return-based metrics
            returns = df['return_1d'].dropna()
            if len(returns) > 0:
                avg_return = returns.mean()
                volatility = returns.std()
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
                
                # Win rate (percentage of positive returns)
                win_rate = (returns > 0).mean()
                
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
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate performance for {model_name}: {e}")
            return None
    
    def detect_market_regime(self, market_data: pd.Series, 
                           volatility: float) -> str:
        """
        Detect current market regime.
        
        Args:
            market_data: Recent price data
            volatility: Current volatility measure
            
        Returns:
            Market regime string
        """
        if len(market_data) < 20:
            return 'unknown'
        
        # Calculate trend strength
        recent_return = (market_data.iloc[-1] / market_data.iloc[-20]) - 1
        
        # Volatility thresholds
        high_vol_threshold = 0.03
        low_vol_threshold = 0.015
        
        # Trend thresholds
        strong_trend_threshold = 0.05
        
        # Regime classification
        if volatility > high_vol_threshold:
            if abs(recent_return) > strong_trend_threshold:
                return 'crisis'  # High volatility + strong move
            else:
                return 'volatile'  # High volatility, no strong trend
        elif abs(recent_return) > strong_trend_threshold:
            return 'trending'  # Low/medium volatility + strong trend
        else:
            return 'sideways'  # Low volatility + weak trend
    
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
            accuracy_score * self.performance_factors['accuracy'] +
            sharpe_score * self.performance_factors['sharpe_ratio'] +
            win_rate_score * self.performance_factors['win_rate'] +
            drawdown_score * self.performance_factors['max_drawdown'] +
            volatility_score * self.performance_factors['volatility'] +
            recency_score * self.performance_factors['recency']
        )
        
        return max(0.0, min(1.0, weighted_score))
    
    def calculate_adaptive_weights(self, 
                                 market_data: pd.Series = None,
                                 volatility: float = None,
                                 force_update: bool = False) -> WeightAdjustment:
        """
        Calculate adaptive weights based on recent model performance.
        
        Args:
            market_data: Recent market price data
            volatility: Current market volatility
            force_update: Force weight recalculation even with limited data
            
        Returns:
            WeightAdjustment object with new weights and reasoning
        """
        # Detect market regime
        market_regime = 'unknown'
        if market_data is not None and volatility is not None:
            market_regime = self.detect_market_regime(market_data, volatility)
        
        # Calculate performance for each model
        model_performances = {}
        performance_scores = {}
        
        for model_name in self.base_weights.keys():
            performance = self.calculate_model_performance(model_name)
            if performance is not None:
                model_performances[model_name] = performance
                performance_scores[model_name] = self.calculate_performance_score(performance)
            else:
                # Use base performance if no data available
                performance_scores[model_name] = 0.5  # Neutral score
        
        # Check if we have enough data for adaptation
        models_with_data = len([p for p in model_performances.values() if p is not None])
        
        if models_with_data < 2 and not force_update:
            # Not enough data, return base weights
            return WeightAdjustment(
                model_weights=self.base_weights.copy(),
                adjustment_reason="Insufficient performance data for adaptation",
                confidence=0.3,
                market_regime=market_regime,
                performance_period=f"{self.lookback_days} days"
            )
        
        # Calculate performance-based weights
        total_performance = sum(performance_scores.values())
        if total_performance == 0:
            performance_weights = self.base_weights.copy()
        else:
            performance_weights = {
                model: score / total_performance 
                for model, score in performance_scores.items()
            }
        
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
            final_weights = {
                model: weight / total_weight 
                for model, weight in regime_adjusted_weights.items()
            }
        else:
            final_weights = self.base_weights.copy()
        
        # Smooth transition from base weights (avoid dramatic changes)
        smoothing_factor = 0.7  # 70% new weights, 30% base weights
        smoothed_weights = {}
        for model in self.base_weights.keys():
            new_weight = final_weights.get(model, self.base_weights[model])
            base_weight = self.base_weights[model]
            smoothed_weights[model] = (
                smoothing_factor * new_weight + 
                (1 - smoothing_factor) * base_weight
            )
        
        # Generate adjustment reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Market regime: {market_regime}")
        
        # Identify top performing model
        if performance_scores:
            top_model = max(performance_scores.keys(), key=lambda k: performance_scores[k])
            reasoning_parts.append(f"Top performer: {top_model}")
        
        # Identify significant weight changes
        significant_changes = []
        for model, new_weight in smoothed_weights.items():
            base_weight = self.base_weights[model]
            change = (new_weight - base_weight) / base_weight
            if abs(change) > 0.1:  # 10% change threshold
                direction = "increased" if change > 0 else "decreased"
                significant_changes.append(f"{model} {direction} by {abs(change):.1%}")
        
        if significant_changes:
            reasoning_parts.extend(significant_changes[:2])  # Limit to top 2 changes
        
        reasoning = "; ".join(reasoning_parts)
        
        # Calculate confidence based on data quality and consistency
        confidence = min(0.9, 0.3 + (models_with_data / len(self.base_weights)) * 0.6)
        
        return WeightAdjustment(
            model_weights=smoothed_weights,
            adjustment_reason=reasoning,
            confidence=confidence,
            market_regime=market_regime,
            performance_period=f"{self.lookback_days} days"
        )
    
    def get_current_weights(self, 
                          market_data: pd.Series = None,
                          volatility: float = None) -> Dict[str, float]:
        """
        Get current recommended weights (convenience method).
        
        Args:
            market_data: Recent market data
            volatility: Current volatility
            
        Returns:
            Dictionary of model weights
        """
        weight_adjustment = self.calculate_adaptive_weights(market_data, volatility)
        return weight_adjustment.model_weights
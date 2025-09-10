"""
Enhanced Decision Engine for Trading AI System
Provides advanced decision logic with consensus validation, confidence scoring, 
and adaptive thresholds.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

@dataclass
class ModelDecision:
    """Structured representation of a model's decision"""
    signal: str
    confidence: float
    strength: SignalStrength
    timestamp: datetime
    model_name: str
    reasoning: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'signal': self.signal,
            'confidence': self.confidence,
            'strength': self.strength.value,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'reasoning': self.reasoning
        }

@dataclass
class HybridDecision:
    """Final hybrid decision with detailed metadata"""
    final_signal: str
    final_confidence: float
    consensus_score: float
    disagreement_factor: float
    risk_adjusted_signal: str
    individual_decisions: List[ModelDecision]
    reasoning: str
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'final_signal': self.final_signal,
            'final_confidence': self.final_confidence,
            'consensus_score': self.consensus_score,
            'disagreement_factor': self.disagreement_factor,
            'risk_adjusted_signal': self.risk_adjusted_signal,
            'individual_decisions': [d.to_dict() for d in self.individual_decisions],
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }

class EnhancedDecisionEngine:
    """
    Enhanced decision engine with consensus validation, adaptive thresholds,
    and risk-adjusted decision making.
    """
    
    def __init__(self, base_weights: Dict[str, float] = None):
        """
        Initialize the enhanced decision engine.
        
        Args:
            base_weights: Base weights for each model type
        """
        self.base_weights = base_weights or {
            'classic': 0.35,
            'llm_text': 0.25,
            'llm_visual': 0.25,
            'sentiment': 0.15
        }
        
        # Adaptive thresholds based on market conditions
        self.adaptive_thresholds = {
            'strong_buy': 0.6,
            'buy': 0.2,
            'hold_upper': 0.1,
            'hold_lower': -0.1,
            'sell': -0.2,
            'strong_sell': -0.6
        }
        
        # Track model performance for weight adjustment
        self.model_performance_history = {}
        
    def _normalize_signal(self, signal: str) -> SignalStrength:
        """Convert signal string to SignalStrength enum"""
        signal = signal.upper()
        if signal in ['STRONG BUY', 'STRONG_BUY']:
            return SignalStrength.STRONG_BUY
        elif signal == 'BUY':
            return SignalStrength.BUY
        elif signal == 'HOLD':
            return SignalStrength.HOLD
        elif signal == 'SELL':
            return SignalStrength.SELL
        elif signal in ['STRONG SELL', 'STRONG_SELL']:
            return SignalStrength.STRONG_SELL
        else:
            logger.warning(f"Unknown signal: {signal}, defaulting to HOLD")
            return SignalStrength.HOLD
    
    def _calculate_consensus_score(self, decisions: List[ModelDecision]) -> float:
        """
        Calculate consensus score based on agreement between models.
        Returns value between 0 (no consensus) and 1 (perfect consensus).
        """
        if len(decisions) < 2:
            return 1.0
        
        signals = [d.strength.value for d in decisions]
        confidences = [d.confidence for d in decisions]
        
        # Calculate signal agreement (how close are the signals)
        signal_variance = np.var(signals)
        max_variance = 4  # Maximum possible variance for signals (-2 to 2)
        signal_agreement = 1 - (signal_variance / max_variance)
        
        # Calculate confidence alignment (do high confidence models agree?)
        weighted_signal = np.average(signals, weights=confidences)
        confidence_alignment = 1 - np.average([abs(s - weighted_signal) for s in signals])
        
        # Combined consensus score
        consensus = (signal_agreement + confidence_alignment) / 2
        return max(0, min(1, consensus))
    
    def _calculate_disagreement_factor(self, decisions: List[ModelDecision]) -> float:
        """
        Calculate disagreement factor to identify conflicting signals.
        Returns value between 0 (perfect agreement) and 1 (maximum disagreement).
        """
        if len(decisions) < 2:
            return 0.0
        
        signals = [d.strength.value for d in decisions]
        
        # Count opposing signals (buy vs sell)
        buy_signals = sum(1 for s in signals if s > 0)
        sell_signals = sum(1 for s in signals if s < 0)
        
        if buy_signals > 0 and sell_signals > 0:
            # There are conflicting signals
            conflict_ratio = min(buy_signals, sell_signals) / len(signals)
            return conflict_ratio * 2  # Scale to 0-1 range
        
        return 0.0
    
    def _adjust_for_market_regime(self, score: float, market_data: Dict) -> float:
        """
        Adjust decision score based on current market regime.
        
        Args:
            score: Base decision score
            market_data: Current market indicators (volatility, trend, etc.)
        """
        # Example regime adjustments
        volatility = market_data.get('volatility', 0.02)
        rsi = market_data.get('rsi', 50)
        
        # Reduce signal strength in high volatility environments
        if volatility > 0.04:  # High volatility
            score *= 0.8
        elif volatility < 0.01:  # Low volatility
            score *= 1.1
        
        # Adjust for overbought/oversold conditions
        if rsi > 80 and score > 0:  # Overbought, reduce buy signals
            score *= 0.7
        elif rsi < 20 and score < 0:  # Oversold, reduce sell signals
            score *= 0.7
        
        return score
    
    def _apply_risk_management(self, decision: str, confidence: float, 
                             market_data: Dict) -> str:
        """
        Apply risk management rules to adjust the final decision.
        
        Args:
            decision: Original decision
            confidence: Decision confidence
            market_data: Current market conditions
        """
        # Conservative adjustment for low confidence
        if confidence < 0.6:
            if decision in ['STRONG_BUY', 'STRONG_SELL']:
                # Downgrade strong signals to regular signals
                return 'BUY' if 'BUY' in decision else 'SELL'
            elif decision in ['BUY', 'SELL'] and confidence < 0.4:
                # Very low confidence, default to HOLD
                return 'HOLD'
        
        # Market regime-based adjustments
        volatility = market_data.get('volatility', 0.02)
        if volatility > 0.05:  # Very high volatility
            # Be more conservative in volatile markets
            if decision in ['BUY', 'SELL']:
                return 'HOLD'
        
        return decision
    
    def make_enhanced_decision(
        self, 
        classic_pred: int, 
        classic_conf: float,
        text_llm_decision: Dict,
        visual_llm_decision: Dict,
        sentiment_decision: Dict,
        market_data: Dict = None,
        adaptive_weights: Dict[str, float] = None
    ) -> HybridDecision:
        """
        Enhanced decision making with consensus validation and risk management.
        
        Args:
            classic_pred: Classic model prediction (0 or 1)
            classic_conf: Classic model confidence
            text_llm_decision: Text LLM decision dict
            visual_llm_decision: Visual LLM decision dict
            sentiment_decision: Sentiment analysis decision dict
            market_data: Current market indicators
            adaptive_weights: Optional adaptive weights for models
            
        Returns:
            HybridDecision object with detailed decision information
        """
        timestamp = datetime.now()
        market_data = market_data or {}
        
        # Use adaptive weights if provided, otherwise use base weights
        weights = adaptive_weights or self.base_weights
        
        # Create structured decisions for each model
        decisions = [
            ModelDecision(
                signal='BUY' if classic_pred == 1 else 'SELL',
                confidence=classic_conf,
                strength=self._normalize_signal('BUY' if classic_pred == 1 else 'SELL'),
                timestamp=timestamp,
                model_name='classic',
                reasoning='Quantitative model prediction'
            ),
            ModelDecision(
                signal=text_llm_decision.get('signal', 'HOLD'),
                confidence=text_llm_decision.get('confidence', 0.0),
                strength=self._normalize_signal(text_llm_decision.get('signal', 'HOLD')),
                timestamp=timestamp,
                model_name='llm_text',
                reasoning=text_llm_decision.get('analysis', 'Text-based analysis')
            ),
            ModelDecision(
                signal=visual_llm_decision.get('signal', 'HOLD'),
                confidence=visual_llm_decision.get('confidence', 0.0),
                strength=self._normalize_signal(visual_llm_decision.get('signal', 'HOLD')),
                timestamp=timestamp,
                model_name='llm_visual',
                reasoning=visual_llm_decision.get('analysis', 'Visual chart analysis')
            ),
            ModelDecision(
                signal=sentiment_decision.get('signal', 'HOLD'),
                confidence=sentiment_decision.get('confidence', 0.0),
                strength=self._normalize_signal(sentiment_decision.get('signal', 'HOLD')),
                timestamp=timestamp,
                model_name='sentiment',
                reasoning='News sentiment analysis'
            )
        ]
        
        # Calculate weighted score
        weighted_score = 0.0
        for decision in decisions:
            model_weight = weights.get(decision.model_name, 0.25)
            signal_value = decision.strength.value
            weighted_score += signal_value * decision.confidence * model_weight
        
        # Adjust for market regime
        adjusted_score = self._adjust_for_market_regime(weighted_score, market_data)
        
        # Calculate consensus metrics
        consensus_score = self._calculate_consensus_score(decisions)
        disagreement_factor = self._calculate_disagreement_factor(decisions)
        
        # Determine final signal based on adjusted score and adaptive thresholds
        if adjusted_score >= self.adaptive_thresholds['strong_buy']:
            final_signal = 'STRONG_BUY'
        elif adjusted_score >= self.adaptive_thresholds['buy']:
            final_signal = 'BUY'
        elif adjusted_score <= self.adaptive_thresholds['strong_sell']:
            final_signal = 'STRONG_SELL'
        elif adjusted_score <= self.adaptive_thresholds['sell']:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Calculate final confidence considering consensus
        base_confidence = abs(adjusted_score) / 2  # Normalize to 0-1
        consensus_adjustment = consensus_score * 0.3  # Boost confidence for consensus
        disagreement_penalty = disagreement_factor * 0.2  # Reduce confidence for disagreement
        
        final_confidence = min(1.0, max(0.0, 
            base_confidence + consensus_adjustment - disagreement_penalty))
        
        # Apply risk management
        risk_adjusted_signal = self._apply_risk_management(
            final_signal, final_confidence, market_data)
        
        # Generate reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Weighted score: {adjusted_score:.3f}")
        reasoning_parts.append(f"Consensus: {consensus_score:.2f}")
        
        if disagreement_factor > 0.3:
            reasoning_parts.append(f"High disagreement detected ({disagreement_factor:.2f})")
        
        if risk_adjusted_signal != final_signal:
            reasoning_parts.append(f"Risk management adjusted from {final_signal}")
        
        reasoning = "; ".join(reasoning_parts)
        
        return HybridDecision(
            final_signal=final_signal,
            final_confidence=final_confidence,
            consensus_score=consensus_score,
            disagreement_factor=disagreement_factor,
            risk_adjusted_signal=risk_adjusted_signal,
            individual_decisions=decisions,
            reasoning=reasoning,
            timestamp=timestamp
        )
    
    def update_adaptive_thresholds(self, market_volatility: float, trend_strength: float):
        """
        Update adaptive thresholds based on market conditions.
        
        Args:
            market_volatility: Current market volatility
            trend_strength: Current trend strength indicator
        """
        base_adjustment = market_volatility * 0.1
        
        # Widen thresholds in volatile markets (be more conservative)
        if market_volatility > 0.03:
            self.adaptive_thresholds['strong_buy'] += base_adjustment
            self.adaptive_thresholds['strong_sell'] -= base_adjustment
        
        # Narrow thresholds in trending markets (be more responsive)
        if abs(trend_strength) > 0.7:
            trend_adjustment = base_adjustment * 0.5
            self.adaptive_thresholds['buy'] -= trend_adjustment
            self.adaptive_thresholds['sell'] += trend_adjustment
    
    def get_model_weights_recommendation(self, performance_history: Dict) -> Dict[str, float]:
        """
        Recommend adaptive weights based on recent model performance.
        
        Args:
            performance_history: Recent performance metrics for each model
            
        Returns:
            Recommended weights for each model
        """
        if not performance_history:
            return self.base_weights
        
        # Simple performance-based weighting
        # In practice, this could be more sophisticated
        total_performance = sum(performance_history.values())
        if total_performance <= 0:
            return self.base_weights
        
        adaptive_weights = {}
        for model_name, performance in performance_history.items():
            # Performance-based weight with smoothing
            raw_weight = performance / total_performance
            base_weight = self.base_weights.get(model_name, 0.25)
            # Smooth transition: 70% current performance, 30% base weight
            adaptive_weights[model_name] = 0.7 * raw_weight + 0.3 * base_weight
        
        return adaptive_weights
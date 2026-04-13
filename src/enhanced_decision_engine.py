"""
Enhanced Decision Engine for Trading AI System
Provides advanced decision logic with consensus validation, confidence scoring, 
and adaptive thresholds.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
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

class VincentGanneModel:
    """
    Decision model based on Vincent Ganne's criteria for market bottoms.
    Analyzes cross-asset indicators (Oil, Gas, Yields, DXY) and technicals (MA200).
    """
    def __init__(self):
        self.thresholds = {
            'WTI': {'max': 94, 'ideal': 80},
            'Brent': {'max': 95, 'ideal': 83},
            'Gas': {'max': 55, 'ideal': 38},
            'Urea': {'max': 506},
            'DXY': {'max': 101, 'ideal': 100}
        }

    def evaluate(self, indicators: dict) -> dict:
        """
        Evaluates the current indicators against Vincent Ganne's rules.
        CRITICAL: Oil prices (WTI/Brent) are the primary indicators for a market bottom.
        """
        score = 0
        max_score = 0
        reasons = []
        
        # 1 & 2. Oil Prices (CRITICAL - Priority 1 & 2)
        wti = indicators.get('WTI_price')
        brent = indicators.get('Brent_price')
        
        # Priority 1: WTI (Weight 10)
        if wti:
            max_score += 10
            if wti < self.thresholds['WTI']['ideal']:
                score += 10
                reasons.append(f"WTI at IDEAL level ({wti:.2f})")
            elif wti < self.thresholds['WTI']['max']:
                score += 5
                reasons.append(f"WTI below threshold ({wti:.2f})")
            else:
                reasons.append(f"WTI TOO HIGH ({wti:.2f} > 94$)")

        # Priority 2: Brent (Weight 8)
        if brent:
            max_score += 8
            if brent < self.thresholds['Brent']['ideal']:
                score += 8
                reasons.append(f"Brent at IDEAL level ({brent:.2f})")
            elif brent < self.thresholds['Brent']['max']:
                score += 4
                reasons.append(f"Brent below threshold ({brent:.2f})")
            else:
                reasons.append(f"Brent TOO HIGH ({brent:.2f} > 95$)")

        # HARD BLOCK: If Oil is too high, it's NOT a market bottom (Vincent Ganne rule)
        oil_is_too_high = (wti and wti >= 94) or (brent and brent >= 95)

        # 3. Natural Gas (Weight 5)
        gas = indicators.get('NaturalGas_price')
        if gas:
            max_score += 5
            if gas < self.thresholds['Gas']['ideal']:
                score += 5
                reasons.append(f"Natural Gas IDEAL ({gas:.2f})")
            elif gas < self.thresholds['Gas']['max']:
                score += 2.5
                reasons.append(f"Natural Gas below threshold ({gas:.2f})")

        # 4. Urea Fertilizer (Weight 4)
        urea = indicators.get('Urea_price')
        if urea:
            max_score += 4
            if urea < self.thresholds['Urea']['max']:
                score += 4
                reasons.append("Urea below threshold")

        # US 2Y vs Fed Rate (Weight 3)
        yield_2y = indicators.get('US2Y_yield')
        fed_rate = indicators.get('Fed_rate')
        if yield_2y and fed_rate:
            max_score += 3
            if abs(yield_2y - fed_rate) < 0.25:
                score += 3
                reasons.append("Yields normalized")

        # US Dollar DXY (Weight 3)
        dxy = indicators.get('DXY_price')
        if dxy:
            max_score += 3
            if dxy < self.thresholds['DXY']['max']:
                score += 3
                reasons.append(f"Dollar weak ({dxy:.2f})")

        # Indices MA200 (Weight 1 each - Confirmation only)
        for idx in ['SP500', 'Nasdaq', 'DowJones', 'TechSector']:
            if indicators.get(f'{idx}_above_ma200'):
                max_score += 1
                score += 1

        # FINAL DECISION LOGIC
        confidence = score / max_score if max_score > 0 else 0
        
        if oil_is_too_high:
            signal = "SELL" if confidence < 0.3 else "HOLD"
            reasons.insert(0, "GEO-POLITICAL PRESSURE (OIL TOO HIGH)")
        else:
            if confidence > 0.75:
                signal = "STRONG_BUY"
            elif confidence > 0.5:
                signal = "BUY"
            else:
                signal = "HOLD"

        return {
            'signal': signal,
            'confidence': confidence,
            'analysis': " | ".join(reasons)
        }

class EnhancedDecisionEngine:
    """
    Enhanced decision engine with consensus validation, adaptive thresholds,
    and risk-adjusted decision making.
    """

    # Decision thresholds and tuning constants
    BULLISH_BIAS = 0.0  # No hardcoded bias — signal should be evidence-based
    VOLATILITY_HIGH_THRESHOLD = 0.04
    VOLATILITY_LOW_THRESHOLD = 0.01
    VOLATILITY_EXTREME_THRESHOLD = 0.06
    RSI_OVERBOUGHT_THRESHOLD = 80
    RSI_OVERSOLD_THRESHOLD = 25
    CLASSIC_BUY_BONUS_THRESHOLD = 0.4
    TIMESFM_BUY_BONUS_THRESHOLD = 0.2
    QUANT_MODEL_BUY_BONUS = 0.0  # Removed: was +0.1 per quant model, creating structural bullish bias
    SUPER_CONSENSUS_BOOST = 0.15

    # Confidence thresholds for risk management
    MIN_CONFIDENCE_FOR_ACTION = 0.20
    MIN_CONFIDENCE_FOR_SELL = 0.40

    def __init__(self, base_weights: Dict[str, float] = None):
        """
        Initialize the enhanced decision engine.
        
        Args:
            base_weights: Base weights for each model type
        """
        self.base_weights = base_weights or {
            'classic': 0.15,
            'llm_text': 0.20,
            'llm_visual': 0.15,
            'sentiment': 0.10,
            'timesfm': 0.20,
            'vincent_ganne': 0.20  # Significant weight for geopolitical/cross-asset model
        }
        self.vincent_ganne_model = VincentGanneModel()
        
        # Adaptive thresholds (Balanced for Index trading)
        self.adaptive_thresholds = {
            'strong_buy': 0.35,
            'buy': 0.12,
            'hold_upper': 0.05,
            'hold_lower': -0.05,
            'sell': -0.15,
            'strong_sell': -0.45
        }

        # Track model performance for weight adjustment
        self.model_performance_history = {}

        # Market regime adaptation thresholds
        self.regime_volatility_high = 0.03
        self.trend_strength_threshold = 0.7
        
    def _normalize_signal(self, signal: str) -> SignalStrength:
        """Convert signal string to SignalStrength enum"""
        if not signal:
            return SignalStrength.HOLD
            
        signal = signal.upper().replace(' ', '_')
        if signal in ['STRONG_BUY', 'VERY_BULLISH']:
            return SignalStrength.STRONG_BUY
        elif signal in ['BUY', 'BULLISH']:
            return SignalStrength.BUY
        elif signal in ['HOLD', 'NEUTRAL', 'STAY']:
            return SignalStrength.HOLD
        elif signal in ['SELL', 'BEARISH']:
            return SignalStrength.SELL
        elif signal in ['STRONG_SELL', 'VERY_BEARISH']:
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
        if sum(confidences) == 0:
            weighted_signal = np.mean(signals)
        else:
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
        """
        volatility = market_data.get('volatility', 0.02)
        rsi = market_data.get('rsi', 50)

        # Reduce signal strength in high volatility environments
        if volatility > self.VOLATILITY_HIGH_THRESHOLD:
            score *= 0.8
        elif volatility < self.VOLATILITY_LOW_THRESHOLD:
            score *= 1.1

        # Adjust for overbought/oversold conditions
        if rsi > self.RSI_OVERBOUGHT_THRESHOLD and score > 0:
            score *= 0.7
        elif rsi < self.RSI_OVERSOLD_THRESHOLD and score < 0:
            score *= 0.7

        return score
    
    def _apply_risk_management(self, decision: str, confidence: float,
                             market_data: Dict) -> str:
        """
        Apply risk management rules to adjust the final decision.
        """
        if decision in ['BUY', 'STRONG_BUY']:
            if confidence < self.MIN_CONFIDENCE_FOR_ACTION:
                return 'HOLD'
            return decision

        # Conservative adjustment for low confidence on SELL signals
        if confidence < self.MIN_CONFIDENCE_FOR_SELL:
            if decision in ['STRONG_SELL']:
                return 'SELL'
            elif decision == 'SELL':
                return 'HOLD'

        # Market regime-based adjustments
        volatility = market_data.get('volatility', 0.02)
        if volatility > self.VOLATILITY_EXTREME_THRESHOLD:
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
        timesfm_decision: Dict = None,
        vincent_ganne_indicators: Dict = None,
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
            timesfm_decision: TimesFM forecasting decision dict
            vincent_ganne_indicators: Indicators for Vincent Ganne model
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
        
        if timesfm_decision:
            decisions.append(
                ModelDecision(
                    signal=timesfm_decision.get('signal', 'HOLD'),
                    confidence=timesfm_decision.get('confidence', 0.0),
                    strength=self._normalize_signal(timesfm_decision.get('signal', 'HOLD')),
                    timestamp=timestamp,
                    model_name='timesfm',
                    reasoning=timesfm_decision.get('analysis', 'TimesFM time series forecasting')
                )
            )

        if vincent_ganne_indicators:
            vg_decision = self.vincent_ganne_model.evaluate(vincent_ganne_indicators)
            decisions.append(
                ModelDecision(
                    signal=vg_decision['signal'],
                    confidence=vg_decision['confidence'],
                    strength=self._normalize_signal(vg_decision['signal']),
                    timestamp=timestamp,
                    model_name='vincent_ganne',
                    reasoning=vg_decision['analysis']
                )
            )

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
        
        # Super-Consensus Boost: if Classic and TimesFM agree on a non-HOLD signal
        classic_decision = next((d for d in decisions if d.model_name == 'classic'), None)
        timesfm_decision = next((d for d in decisions if d.model_name == 'timesfm'), None)
        
        boost = 0.0
        if classic_decision and timesfm_decision:
            if (classic_decision.strength.value * timesfm_decision.strength.value) > 0:
                # Both agree on BUY or both agree on SELL
                if classic_decision.confidence > 0.5 and timesfm_decision.confidence > 0.3:
                    boost = self.SUPER_CONSENSUS_BOOST
        
        final_confidence = min(1.0, max(0.0, 
            base_confidence + consensus_adjustment - disagreement_penalty + boost))
        
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
        if market_volatility > self.regime_volatility_high:
            self.adaptive_thresholds['strong_buy'] += base_adjustment
            self.adaptive_thresholds['strong_sell'] -= base_adjustment

        # Narrow thresholds in trending markets (be more responsive)
        if abs(trend_strength) > self.trend_strength_threshold:
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
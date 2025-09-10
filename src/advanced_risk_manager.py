"""
Advanced Risk Management Module for Trading AI System
Provides comprehensive risk assessment, position sizing, and risk-adjusted decision making.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for current market conditions"""
    volatility_risk: float
    drawdown_risk: float
    correlation_risk: float
    liquidity_risk: float
    overall_risk_score: float
    risk_level: RiskLevel
    
    def to_dict(self) -> dict:
        return {
            'volatility_risk': self.volatility_risk,
            'drawdown_risk': self.drawdown_risk,
            'correlation_risk': self.correlation_risk,
            'liquidity_risk': self.liquidity_risk,
            'overall_risk_score': self.overall_risk_score,
            'risk_level': self.risk_level.name
        }

@dataclass
class PositionSizing:
    """Position sizing recommendation based on risk assessment"""
    recommended_size: float
    max_position_size: float
    risk_adjusted_size: float
    kelly_criterion_size: float
    reasoning: str
    
    def to_dict(self) -> dict:
        return {
            'recommended_size': self.recommended_size,
            'max_position_size': self.max_position_size,
            'risk_adjusted_size': self.risk_adjusted_size,
            'kelly_criterion_size': self.kelly_criterion_size,
            'reasoning': self.reasoning
        }

class AdvancedRiskManager:
    """
    Advanced risk management system with dynamic risk assessment,
    position sizing, and risk-adjusted decision making.
    """
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,
                 max_position_risk: float = 0.01,
                 lookback_period: int = 252):
        """
        Initialize the risk manager.
        
        Args:
            max_portfolio_risk: Maximum daily portfolio risk (as fraction)
            max_position_risk: Maximum risk per position (as fraction)
            lookback_period: Period for risk calculations (trading days)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.lookback_period = lookback_period
        
        # Risk thresholds
        self.volatility_thresholds = {
            RiskLevel.VERY_LOW: 0.01,
            RiskLevel.LOW: 0.015,
            RiskLevel.MODERATE: 0.025,
            RiskLevel.HIGH: 0.04,
            RiskLevel.VERY_HIGH: float('inf')
        }
        
        # Market regime indicators
        self.market_regimes = {
            'bull_market': {'threshold': 0.15, 'risk_multiplier': 0.8},
            'bear_market': {'threshold': -0.15, 'risk_multiplier': 1.5},
            'sideways': {'threshold': 0.05, 'risk_multiplier': 1.0}
        }
    
    def calculate_volatility_risk(self, price_data: pd.Series, 
                                window: int = 20) -> float:
        """
        Calculate volatility-based risk score.
        
        Args:
            price_data: Historical price series
            window: Rolling window for volatility calculation
            
        Returns:
            Volatility risk score (0-1)
        """
        if len(price_data) < window:
            return 0.5  # Default moderate risk
        
        returns = price_data.pct_change().dropna()
        current_vol = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
        
        # Compare to historical volatility percentiles
        historical_vols = returns.rolling(window=window).std() * np.sqrt(252)
        vol_percentile = (historical_vols <= current_vol).mean()
        
        return min(1.0, vol_percentile)
    
    def calculate_drawdown_risk(self, price_data: pd.Series) -> float:
        """
        Calculate maximum drawdown risk.
        
        Args:
            price_data: Historical price series
            
        Returns:
            Drawdown risk score (0-1)
        """
        if len(price_data) < 2:
            return 0.5
        
        # Calculate rolling maximum and drawdown
        rolling_max = price_data.expanding().max()
        drawdown = (price_data - rolling_max) / rolling_max
        
        # Current drawdown
        current_drawdown = abs(drawdown.iloc[-1])
        
        # Historical drawdown percentiles
        max_drawdowns = []
        for i in range(60, len(drawdown)):  # 60-day rolling max drawdown
            period_dd = drawdown.iloc[i-60:i].min()
            max_drawdowns.append(abs(period_dd))
        
        if not max_drawdowns:
            return current_drawdown * 10  # Simple scaling if no history
        
        dd_percentile = np.percentile(max_drawdowns, 90)  # 90th percentile
        risk_score = min(1.0, current_drawdown / max(0.01, dd_percentile))
        
        return risk_score
    
    def calculate_correlation_risk(self, returns: pd.Series, 
                                 market_returns: pd.Series = None) -> float:
        """
        Calculate correlation risk with market benchmark.
        
        Args:
            returns: Asset returns
            market_returns: Market benchmark returns (e.g., SPY)
            
        Returns:
            Correlation risk score (0-1)
        """
        if market_returns is None or len(returns) < 30:
            return 0.3  # Default low correlation risk
        
        # Align the series
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 30:
            return 0.3
        
        correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
        
        # High correlation increases systemic risk
        # Risk increases as correlation approaches 1.0
        correlation_risk = abs(correlation) ** 2
        
        return min(1.0, correlation_risk)
    
    def calculate_liquidity_risk(self, volume_data: pd.Series, 
                               price_data: pd.Series) -> float:
        """
        Calculate liquidity risk based on volume patterns.
        
        Args:
            volume_data: Historical volume series
            price_data: Historical price series
            
        Returns:
            Liquidity risk score (0-1)
        """
        if len(volume_data) < 20:
            return 0.4  # Default moderate liquidity risk
        
        # Volume trend analysis
        recent_volume = volume_data.tail(5).mean()
        historical_volume = volume_data.tail(60).mean()
        volume_ratio = recent_volume / max(historical_volume, 1)
        
        # Price-volume relationship
        returns = price_data.pct_change()
        volume_returns_corr = abs(returns.corr(volume_data))
        
        # Low volume or abnormal volume patterns increase liquidity risk
        volume_risk = 1.0 / (1.0 + volume_ratio)  # Lower volume = higher risk
        pattern_risk = 1.0 - volume_returns_corr  # Low correlation = higher risk
        
        liquidity_risk = (volume_risk + pattern_risk) / 2
        return min(1.0, max(0.0, liquidity_risk))
    
    def assess_market_regime(self, price_data: pd.Series, 
                           lookback: int = 60) -> Dict[str, float]:
        """
        Assess current market regime (bull/bear/sideways).
        
        Args:
            price_data: Historical price series
            lookback: Lookback period for regime assessment
            
        Returns:
            Market regime probabilities
        """
        if len(price_data) < lookback:
            return {'bull_market': 0.33, 'bear_market': 0.33, 'sideways': 0.34}
        
        # Calculate return over lookback period
        total_return = (price_data.iloc[-1] / price_data.iloc[-lookback]) - 1
        
        # Trend strength
        ma_short = price_data.tail(20).mean()
        ma_long = price_data.tail(lookback).mean()
        trend_strength = (ma_short / ma_long) - 1
        
        # Regime classification
        if total_return > self.market_regimes['bull_market']['threshold']:
            return {'bull_market': 0.7, 'bear_market': 0.1, 'sideways': 0.2}
        elif total_return < self.market_regimes['bear_market']['threshold']:
            return {'bull_market': 0.1, 'bear_market': 0.7, 'sideways': 0.2}
        else:
            return {'bull_market': 0.25, 'bear_market': 0.25, 'sideways': 0.5}
    
    def calculate_comprehensive_risk(self, 
                                   price_data: pd.Series,
                                   volume_data: pd.Series = None,
                                   market_data: pd.Series = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            price_data: Historical price series
            volume_data: Historical volume series
            market_data: Market benchmark data
            
        Returns:
            Comprehensive risk metrics
        """
        # Individual risk components
        volatility_risk = self.calculate_volatility_risk(price_data)
        drawdown_risk = self.calculate_drawdown_risk(price_data)
        
        returns = price_data.pct_change().dropna()
        market_returns = market_data.pct_change().dropna() if market_data is not None else None
        correlation_risk = self.calculate_correlation_risk(returns, market_returns)
        
        liquidity_risk = 0.3  # Default if no volume data
        if volume_data is not None:
            liquidity_risk = self.calculate_liquidity_risk(volume_data, price_data)
        
        # Weighted overall risk score
        weights = {
            'volatility': 0.35,
            'drawdown': 0.25,
            'correlation': 0.25,
            'liquidity': 0.15
        }
        
        overall_risk_score = (
            volatility_risk * weights['volatility'] +
            drawdown_risk * weights['drawdown'] +
            correlation_risk * weights['correlation'] +
            liquidity_risk * weights['liquidity']
        )
        
        # Determine risk level
        risk_level = RiskLevel.VERY_LOW
        for level, threshold in self.volatility_thresholds.items():
            if overall_risk_score <= threshold:
                risk_level = level
                break
        
        return RiskMetrics(
            volatility_risk=volatility_risk,
            drawdown_risk=drawdown_risk,
            correlation_risk=correlation_risk,
            liquidity_risk=liquidity_risk,
            overall_risk_score=overall_risk_score,
            risk_level=risk_level
        )
    
    def calculate_kelly_criterion(self, win_rate: float, 
                                avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            
        Returns:
            Kelly optimal fraction (0-1)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.1  # Conservative default
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly for safety (typically 25-50% of full Kelly)
        fractional_kelly = max(0, min(0.25, kelly_fraction * 0.25))
        
        return fractional_kelly
    
    def calculate_position_sizing(self, 
                                signal_strength: float,
                                confidence: float,
                                risk_metrics: RiskMetrics,
                                portfolio_value: float,
                                current_price: float,
                                historical_performance: Dict = None) -> PositionSizing:
        """
        Calculate optimal position size considering multiple factors.
        
        Args:
            signal_strength: Signal strength (-2 to 2)
            confidence: Model confidence (0-1)
            risk_metrics: Current risk assessment
            portfolio_value: Total portfolio value
            current_price: Current asset price
            historical_performance: Historical trading performance
            
        Returns:
            Position sizing recommendation
        """
        # Base position size as percentage of portfolio
        base_size_pct = 0.1  # 10% base allocation
        
        # Adjust for signal strength and confidence
        signal_adjustment = abs(signal_strength) / 2  # Normalize to 0-1
        confidence_adjustment = confidence
        strength_factor = (signal_adjustment + confidence_adjustment) / 2
        
        # Risk adjustment
        risk_factor = 1.0 - (risk_metrics.overall_risk_score * 0.5)
        risk_factor = max(0.2, risk_factor)  # Minimum 20% of base size
        
        # Market regime adjustment
        regime_factor = 1.0  # Default, could be enhanced with regime detection
        
        # Kelly criterion if historical performance available
        kelly_size = 0.1  # Default
        if historical_performance:
            win_rate = historical_performance.get('win_rate', 0.5)
            avg_win = historical_performance.get('avg_win', 0.02)
            avg_loss = historical_performance.get('avg_loss', 0.015)
            kelly_size = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Calculate various position sizes
        strength_adjusted_size = base_size_pct * strength_factor
        risk_adjusted_size = strength_adjusted_size * risk_factor
        kelly_criterion_size = kelly_size
        
        # Final recommended size (conservative approach)
        recommended_size = min(
            strength_adjusted_size,
            risk_adjusted_size,
            kelly_criterion_size * 2,  # Allow up to 2x Kelly for strong signals
            self.max_position_risk  # Hard limit
        )
        
        # Maximum position size (for risk management)
        max_position_size = min(self.max_position_risk, base_size_pct * 1.5)
        
        # Convert to dollar amounts
        recommended_dollar = recommended_size * portfolio_value
        max_dollar = max_position_size * portfolio_value
        risk_adjusted_dollar = risk_adjusted_size * portfolio_value
        kelly_dollar = kelly_criterion_size * portfolio_value
        
        # Generate reasoning
        reasoning_parts = [
            f"Signal strength: {signal_strength:.2f}",
            f"Confidence: {confidence:.2f}",
            f"Risk score: {risk_metrics.overall_risk_score:.2f}",
            f"Risk level: {risk_metrics.risk_level.name}"
        ]
        reasoning = "; ".join(reasoning_parts)
        
        return PositionSizing(
            recommended_size=recommended_dollar,
            max_position_size=max_dollar,
            risk_adjusted_size=risk_adjusted_dollar,
            kelly_criterion_size=kelly_dollar,
            reasoning=reasoning
        )
    
    def should_override_signal(self, signal: str, confidence: float, 
                             risk_metrics: RiskMetrics) -> Tuple[bool, str]:
        """
        Determine if signal should be overridden due to risk concerns.
        
        Args:
            signal: Trading signal
            confidence: Signal confidence
            risk_metrics: Current risk assessment
            
        Returns:
            Tuple of (should_override, reason)
        """
        # High risk override conditions
        if risk_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            if signal in ['BUY', 'STRONG_BUY'] and confidence < 0.8:
                return True, f"High risk environment ({risk_metrics.risk_level.name}) requires high confidence"
        
        # Extreme volatility override
        if risk_metrics.volatility_risk > 0.8:
            if signal in ['STRONG_BUY', 'STRONG_SELL']:
                return True, "Extreme volatility detected, reducing signal strength"
        
        # Significant drawdown override
        if risk_metrics.drawdown_risk > 0.7:
            if signal in ['BUY', 'STRONG_BUY']:
                return True, "Significant drawdown detected, avoiding new long positions"
        
        return False, ""
    
    def get_risk_adjusted_signal(self, original_signal: str, confidence: float,
                               risk_metrics: RiskMetrics) -> Tuple[str, str]:
        """
        Get risk-adjusted trading signal.
        
        Args:
            original_signal: Original trading signal
            confidence: Signal confidence
            risk_metrics: Current risk assessment
            
        Returns:
            Tuple of (adjusted_signal, adjustment_reason)
        """
        should_override, reason = self.should_override_signal(
            original_signal, confidence, risk_metrics)
        
        if should_override:
            # Override logic
            if original_signal in ['STRONG_BUY', 'STRONG_SELL']:
                # Downgrade strong signals
                adjusted = 'BUY' if 'BUY' in original_signal else 'SELL'
                return adjusted, f"Downgraded due to: {reason}"
            elif original_signal in ['BUY', 'SELL']:
                # Convert to HOLD in high risk
                return 'HOLD', f"Converted to HOLD due to: {reason}"
        
        return original_signal, "No risk-based adjustment needed"
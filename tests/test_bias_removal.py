"""Test that EnhancedDecisionEngine has no hardcoded bullish bias."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from enhanced_decision_engine import EnhancedDecisionEngine

def test_no_bullish_bias():
    """Verify that the decision engine doesn't add hardcoded bullish bias."""
    engine = EnhancedDecisionEngine()

    # Test 1: Verify BULLISH_BIAS constant is 0
    assert engine.BULLISH_BIAS == 0.0, f"BULLISH_BIAS should be 0, got {engine.BULLISH_BIAS}"
    print("PASS: BULLISH_BIAS is 0.0")

    # Test 2: Verify QUANT_MODEL_BUY_BONUS is 0
    assert engine.QUANT_MODEL_BUY_BONUS == 0.0, f"QUANT_MODEL_BUY_BONUS should be 0, got {engine.QUANT_MODEL_BUY_BONUS}"
    print("PASS: QUANT_MODEL_BUY_BONUS is 0.0")

    # Test 3: Symmetric signal test — neutral market should produce neutral decision
    # With all models at HOLD and neutral market data, the result should be HOLD
    market_data = {
        'volatility': 0.02,
        'rsi': 50,
        'macd': 0,
        'bb_position': 0.5
    }

    decision = engine.make_enhanced_decision(
        classic_pred=0,  # SELL
        classic_conf=0.5,
        text_llm_decision={'signal': 'HOLD', 'confidence': 0.5, 'analysis': 'Neutral text analysis'},
        visual_llm_decision={'signal': 'HOLD', 'confidence': 0.5, 'analysis': 'Neutral visual analysis'},
        sentiment_decision={'signal': 'HOLD', 'confidence': 0.5},
        timesfm_decision={'signal': 'HOLD', 'confidence': 0.5, 'analysis': 'Neutral forecast'},
        market_data=market_data
    )

    assert decision.final_signal == 'HOLD', f"Expected HOLD with neutral inputs, got {decision.final_signal}"
    print("PASS: Neutral inputs produce HOLD (score would be near 0)")

    # Test 4: Asymmetric test — equal BUY and SELL should NOT bias toward BUY
    decision2 = engine.make_enhanced_decision(
        classic_pred=1,  # BUY
        classic_conf=0.5,
        text_llm_decision={'signal': 'SELL', 'confidence': 0.5, 'analysis': 'Bearish text'},
        visual_llm_decision={'signal': 'SELL', 'confidence': 0.5, 'analysis': 'Bearish visual'},
        sentiment_decision={'signal': 'HOLD', 'confidence': 0.5},
        timesfm_decision={'signal': 'HOLD', 'confidence': 0.5, 'analysis': 'Neutral forecast'},
        market_data=market_data
    )

    # With 1 BUY, 2 SELL, 2 HOLD — no bias should push it to BUY
    assert decision2.final_signal != 'BUY', f"With more SELL than BUY, result should not be BUY (got {decision2.final_signal})"
    print(f"PASS: Mixed signals don't bias toward BUY (result: {decision2.final_signal})")

    # Test 5: Verify magic numbers are named constants
    assert hasattr(engine, 'VOLATILITY_HIGH_THRESHOLD'), "Missing VOLATILITY_HIGH_THRESHOLD"
    assert hasattr(engine, 'RSI_OVERBOUGHT_THRESHOLD'), "Missing RSI_OVERBOUGHT_THRESHOLD"
    assert engine.RSI_OVERBOUGHT_THRESHOLD == 80, f"RSI overbought should be 80, got {engine.RSI_OVERBOUGHT_THRESHOLD}"
    print("PASS: Magic numbers replaced with named constants")

    print("\nAll bias removal tests passed!")

if __name__ == '__main__':
    test_no_bullish_bias()

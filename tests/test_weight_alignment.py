"""Test that weights are aligned between AdaptiveWeightManager and EnhancedDecisionEngine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_weight_manager import AdaptiveWeightManager
from enhanced_decision_engine import EnhancedDecisionEngine


def test_weight_alignment():
    """Verify both modules use the same base weights."""
    engine = EnhancedDecisionEngine()
    weight_mgr = AdaptiveWeightManager()

    expected = {
        "classic": 0.12,
        "llm_text": 0.20,
        "llm_visual": 0.18,
        "sentiment": 0.15,
        "timesfm": 0.20,
        "vincent_ganne": 0.05,
        "oil_bench": 0.05,
        "tensortrade": 0.05,
        "kronos": 0.05,
    }

    # Verify weights sum approximately to 1.0 (test models add ~0.05 each)
    total = sum(expected.values())
    assert abs(total - 1.05) < 1e-6, f"Weights should sum to 1.05, got {total}"
    print(f"PASS: Weights sum to {total}")

    for model_name, expected_weight in expected.items():
        eng_weight = engine.base_weights.get(model_name)
        mgr_weight = weight_mgr.base_weights.get(model_name)
        assert eng_weight == expected_weight, f"Engine {model_name}: expected {expected_weight}, got {eng_weight}"
        assert mgr_weight == expected_weight, f"Manager {model_name}: expected {expected_weight}, got {mgr_weight}"
        print(f"PASS: {model_name} = {expected_weight} (both modules aligned)")

    print("\nAll weight alignment tests passed!")


if __name__ == "__main__":
    test_weight_alignment()

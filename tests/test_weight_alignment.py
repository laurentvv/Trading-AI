"""Test that weights are aligned between AdaptiveWeightManager and EnhancedDecisionEngine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adaptive_weight_manager import AdaptiveWeightManager
from enhanced_decision_engine import EnhancedDecisionEngine

def test_weight_alignment():
    """Verify both modules use the same base weights."""
    engine = EnhancedDecisionEngine()
    weight_mgr = AdaptiveWeightManager()

    expected = {
        'classic': 0.15,
        'llm_text': 0.25,
        'llm_visual': 0.20,
        'sentiment': 0.15,
        'timesfm': 0.25
    }

    for model_name, expected_weight in expected.items():
        eng_weight = engine.base_weights.get(model_name)
        mgr_weight = weight_mgr.base_weights.get(model_name)
        assert eng_weight == expected_weight, f"Engine {model_name}: expected {expected_weight}, got {eng_weight}"
        assert mgr_weight == expected_weight, f"Manager {model_name}: expected {expected_weight}, got {mgr_weight}"
        print(f"PASS: {model_name} = {expected_weight} (both modules aligned)")

    # Verify weights sum to 1.0
    total = sum(expected.values())
    assert total == 1.0, f"Weights should sum to 1.0, got {total}"
    print(f"PASS: Weights sum to {total}")

    print("\nAll weight alignment tests passed!")

if __name__ == '__main__':
    test_weight_alignment()

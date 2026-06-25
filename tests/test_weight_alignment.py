"""Test that weights are aligned between AdaptiveWeightManager and EnhancedDecisionEngine.

Both modules must read from the single source of truth in config_weights.py
to avoid configuration drift (ADR-002).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_weight_manager import AdaptiveWeightManager
from enhanced_decision_engine import EnhancedDecisionEngine
from config_weights import DEFAULT_BASE_WEIGHTS


def test_weight_alignment():
    """Verify both modules use the same base weights as config_weights."""
    engine = EnhancedDecisionEngine()
    weight_mgr = AdaptiveWeightManager()

    for model_name, expected_weight in DEFAULT_BASE_WEIGHTS.items():
        eng_weight = engine.base_weights.get(model_name)
        mgr_weight = weight_mgr.base_weights.get(model_name)
        assert eng_weight == expected_weight, (
            f"Engine {model_name}: expected {expected_weight}, got {eng_weight}"
        )
        assert mgr_weight == expected_weight, (
            f"Manager {model_name}: expected {expected_weight}, got {mgr_weight}"
        )

    # Sanity: the key we previously forgot to keep aligned is now present in both.
    assert set(engine.base_weights.keys()) == set(weight_mgr.base_weights.keys()), (
        "Engine and manager must reference the same set of models"
    )
    print(f"PASS: all {len(DEFAULT_BASE_WEIGHTS)} weights aligned (sum={sum(DEFAULT_BASE_WEIGHTS.values())})")


if __name__ == "__main__":
    test_weight_alignment()

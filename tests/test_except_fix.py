"""Test bare except fix in t212_executor."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.t212_executor import load_portfolio_state, save_portfolio_state, STATE_FILE

def test_except_handling():
    """Test that corrupted state file is handled gracefully."""
    # Test 1: Corrupted JSON file
    backup = None
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            backup = f.read()

    try:
        # Write corrupted JSON
        with open(STATE_FILE, 'w') as f:
            f.write("{corrupted json!!!")

        # Should handle gracefully
        state = load_portfolio_state()
        assert state == {"tickers": {}}, f"Expected empty state, got {state}"
        print("PASS: Corrupted JSON handled gracefully")

        # Test 2: Valid state save/load
        test_state = {
            "initial_budget": 1000.0,
            "current_capital": 1100.0,
            "total_realized_pl": 100.0,
            "active_position": None
        }
        save_portfolio_state(test_state, "TEST")

        loaded = load_portfolio_state("TEST")
        assert loaded["current_capital"] == 1100.0, f"Expected 1100, got {loaded['current_capital']}"
        print("PASS: State save/load works correctly")

    finally:
        # Restore original state
        if backup is not None:
            with open(STATE_FILE, 'w') as f:
                f.write(backup)
        elif os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)

    print("\nAll except handling tests passed!")

if __name__ == '__main__':
    test_except_handling()

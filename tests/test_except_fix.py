"""Test bare except fix in t212_executor."""

import sys
from pathlib import Path

from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.t212_executor import load_portfolio_state, save_portfolio_state


@patch("src.t212_executor.sync_state_from_t212")
def test_except_handling(mock_sync, tmp_path):
    """Test that corrupted state file is handled gracefully."""
    mock_sync.side_effect = Exception("Mocked T212 error")

    state_file = tmp_path / "test_state.json"

    with patch("src.t212_executor.STATE_FILE", str(state_file)):
        # Test 1: Corrupted JSON file
        state_file.write_text("{corrupted json!!!")

        state = load_portfolio_state()
        assert state == {"tickers": {}}, f"Expected empty state, got {state}"

        # Test 2: Valid state save/load
        test_state = {
            "initial_budget": 1000.0,
            "current_capital": 1100.0,
            "total_realized_pl": 100.0,
            "active_position": None,
        }
        save_portfolio_state(test_state, "TEST")

        loaded = load_portfolio_state("TEST")
        assert loaded["current_capital"] == 1100.0, f"Expected 1100, got {loaded['current_capital']}"


if __name__ == "__main__":
    test_except_handling()

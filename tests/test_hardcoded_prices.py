"""Test that hardcoded fallback prices are removed from t212_executor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_no_hardcoded_prices():
    """Verify no hardcoded prices exist in get_real_price_eur."""
    from t212_executor import get_real_price_eur
    import inspect

    source = inspect.getsource(get_real_price_eur)

    # Check no hardcoded price patterns
    assert "1240.0" not in source, "Hardcoded SXRV price 1240.0 still present"
    assert "12.50" not in source, "Hardcoded CRUD price 12.50 still present"
    assert "return 100.0" not in source, "Generic fallback price 100.0 still present"
    print("PASS: No hardcoded prices in get_real_price_eur")

    # Check that ValueError is raised when all sources fail
    assert "raise ValueError" in source, (
        "Should raise ValueError when price unavailable"
    )
    print("PASS: Raises ValueError when price unavailable")

    # Check yfinance fallback exists
    assert "yfinance" in source, "Should have yfinance as fallback"
    print("PASS: Has yfinance as fallback")

    print("\nAll hardcoded price tests passed!")


if __name__ == "__main__":
    test_no_hardcoded_prices()

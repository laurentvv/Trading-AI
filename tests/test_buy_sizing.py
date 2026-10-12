"""Buy sizing vs T212 order rejections (demo 2026-10-08).

Every QDVF.DE buy of 2026-10-08 was refused: 11 times "quantity-precision-mismatch" (4 decimals
sent, 3 allowed) and twice "insufficient-free-for-stocks-buy" (order sized at 99 % of the free
cash, T212 keeps ~5 % for price moves). These tests pin the quantity sent to the broker.
"""

import unittest
from unittest.mock import patch

from src import t212_executor as t212


def _posted_quantity(price: float, budget: float, cash: float, t212_ticker: str) -> float:
    """Run _execute_buy_order with every broker call mocked; return the quantity POSTed."""
    with patch("src.t212_executor.get_real_price_eur", return_value=price), \
         patch("src.t212_executor.post_order_market", return_value=(None, False)) as mock_post, \
         patch("src.t212_executor.save_portfolio_state"), \
         patch("src.t212_executor.insert_transaction"):
        t212._execute_buy_order(
            {"current_capital": budget, "active_position": None}, None, "QDVF.DE", t212_ticker,
            {"cash": cash, "cash_ok": True, "positions": []}, "https://x", {}, "2026-10-08 10:33:00", "TEST",
        )
    mock_post.assert_called_once()
    return mock_post.call_args[0][0]["quantity"]


class TestBuySizing(unittest.TestCase):
    def test_qdvf_precision_is_three_decimals(self):
        self.assertEqual(t212.TICKER_QUANTITY_PRECISION["QDVFd_EQ"], 3)

    def test_cash_bound_buy_stays_under_t212_cash_cap(self):
        # The 2026-10-08 10:33 case: budget 1000 €, free cash 993.23 €, price 12.16 €.
        qty = _posted_quantity(price=12.16, budget=1000.0, cash=993.23, t212_ticker="QDVFd_EQ")
        self.assertEqual(qty, round(qty, 3))
        self.assertLessEqual(qty * 12.16, 993.23 * t212.MAX_CASH_FRACTION_PER_BUY)

    def test_budget_just_under_cash_is_also_capped(self):
        # Cash barely above the budget: spending the whole budget would use 99 % of the cash.
        qty = _posted_quantity(price=100.0, budget=1000.0, cash=1010.0, t212_ticker="SXRVd_EQ")
        self.assertEqual(qty, 9.595)  # 1010 € × 0.95 / 100 €, not 1000 € / 100 €

    def test_budget_bound_buy_uses_full_budget(self):
        qty = _posted_quantity(price=100.0, budget=1000.0, cash=5000.0, t212_ticker="SXRVd_EQ")
        self.assertEqual(qty, 10.0)

    def test_quantity_is_rounded_down(self):
        # 1000 / 7 = 142.857… : rounding to nearest (142.86) would cost 1000.02 € > budget.
        qty = _posted_quantity(price=7.0, budget=1000.0, cash=5000.0, t212_ticker="OD7Fd_EQ")
        self.assertEqual(qty, 142.85)
        self.assertLessEqual(qty * 7.0, 1000.0)


if __name__ == "__main__":
    unittest.main()

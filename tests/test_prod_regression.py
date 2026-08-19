"""
Non-regression test replaying the REAL production data from logs_prod/.

Guards against three bugs found in the 2026-07-15 PROD audit:

1. Risk manager calibration — `Risk_Level` was VERY_HIGH on 100% of 294 PROD
   cycles because volatility-scale thresholds (0.01-0.04) were applied to a
   0-1 composite score. This neutralised every SXRV.DE BUY (147/147
   Risk_Adjusted = HOLD). Asserts the rescaled thresholds let SXRV.DE drop
   out of VERY_HIGH.

2. SELL never fired — 0 SELL across 294 cycles despite ~400 individual SELL
   votes, because HOLD-model abstention diluted the weighted score below the
   SELL threshold. Asserts the renormalised consensus reaches SELL on the
   most-bearish PROD cycle.

3. EIA crude_imports degenerate cache — a 1-row payload was cached with a
   fresh mtime. Asserts the fetcher refuses to cache a degenerate payload.
   Extended 31 July 2026: a 3-row payload that passes the row-count check but
   is months stale in content must also be refused (the live PROD incident).

Skipped when logs_prod/ is absent (dev/CI without PROD data).
Run: .venv\\Scripts\\python.exe -m pytest tests/test_prod_regression.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# Make src/ importable
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

PROD = Path(__file__).parent.parent / "logs_prod"
PROD_CACHE = PROD / "data_cache"

HAS_PROD = PROD.exists() and PROD_CACHE.exists()

# Ticker -> parquet price file (mirrors audit_prod_logs.py)
TICKER_PRICE_FILES = {
    "SXRV.DE": "SXRV_DE_max_with_vix.parquet",
    "CRUDP.PA": "CRUDP_PA_max_with_vix.parquet",
}


def _load_prices(ticker: str) -> pd.DataFrame:
    """Load a PROD price parquet, return df with a datetime index."""
    df = pd.read_parquet(PROD_CACHE / TICKER_PRICE_FILES[ticker])
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df


@unittest.skipUnless(HAS_PROD, "logs_prod/ absent — PROD regression test skipped")
class TestRiskManagerCalibration(unittest.TestCase):
    """Bug #1: thresholds were volatility-scale applied to a composite score."""

    def test_sxrv_not_always_very_high(self):
        """SXRV.DE must NOT be classified VERY_HIGH on 100% of recent cycles.

        Before the fix, the composite score (~0.42) exceeded the old 0.04
        threshold on every cycle, so every SXRV.DE BUY was neutralised.
        """
        from advanced_risk_manager import AdvancedRiskManager, RiskLevel

        rm = AdvancedRiskManager()
        df = _load_prices("SXRV.DE")

        # Evaluate risk over the last 120 trading days, checking the level
        # reported on each of the last 30 sessions.
        levels = []
        for end in range(len(df) - 30, len(df)):
            window = df["Close"].iloc[max(0, end - 250) : end]
            vol = df["Volume"].iloc[max(0, end - 250) : end] if "Volume" in df else None
            metrics = rm.calculate_comprehensive_risk(window, vol)
            levels.append(metrics.risk_level)

        very_high_share = sum(1 for lv in levels if lv == RiskLevel.VERY_HIGH) / len(levels)
        # The audit found 100% VERY_HIGH; after rescaling it must drop below
        # 50% (SXRV.DE's composite is ~0.42, well under the new 0.65 cutoff).
        self.assertLess(
            very_high_share, 0.5,
            f"SXRV.DE was VERY_HIGH on {very_high_share:.0%} of recent cycles; "
            "the rescaled thresholds are not taking effect.",
        )

    def test_thresholds_are_composite_scale(self):
        """Thresholds must be sized for a 0-1 composite, not a vol fraction."""
        from advanced_risk_manager import AdvancedRiskManager, RiskLevel

        rm = AdvancedRiskManager()
        # The old buggy values were 0.01/0.015/0.025/0.04. The MODERATE band
        # must now sit well above 0.04 so a normal composite (~0.4) is not
        # forced to VERY_HIGH.
        self.assertGreater(rm.volatility_thresholds[RiskLevel.MODERATE], 0.10)
        self.assertGreater(rm.volatility_thresholds[RiskLevel.HIGH], 0.30)


@unittest.skipUnless(HAS_PROD, "logs_prod/ absent — PROD regression test skipped")
class TestLiquidityRiskNotInflated(unittest.TestCase):
    """Bug #2: pattern_risk = 1-|corr(volume,returns)| inflated ETFs to ~0.98."""

    def test_etf_liquidity_below_old_ceiling(self):
        from advanced_risk_manager import AdvancedRiskManager

        rm = AdvancedRiskManager()
        df = _load_prices("SXRV.DE")
        window_price = df["Close"].iloc[-120:]
        window_vol = df["Volume"].iloc[-120:] if "Volume" in df else None
        if window_vol is None:
            self.skipTest("No Volume column in SXRV.DE parquet")
        liq = rm.calculate_liquidity_risk(window_vol, window_price)
        # Before the fix, liquidity_risk was ~0.74 for SXRV.DE driven by the
        # un-capped pattern_risk term. After capping+reweighting it should be
        # materially lower.
        self.assertLess(
            liq, 0.60,
            f"liquidity_risk={liq:.3f} still inflated for SXRV.DE; the "
            "pattern_risk cap / reweight is not taking effect.",
        )


class TestConsensusSellReachable(unittest.TestCase):
    """Bug #3: SELL never fired because HOLD abstention diluted the score.

    Uses the most-bearish PROD cycle (2026-07-14 10:02 SXRV.DE, per the audit)
    reconstructed from the journal. The renormalised weighted score must now
    cross the SELL threshold. This test does NOT need logs_prod/ — it uses the
    documented vote breakdown.
    """

    def test_sell_fires_on_bearish_cycle(self):
        from enhanced_decision_engine import EnhancedDecisionEngine, ModelDecision, SignalStrength

        engine = EnhancedDecisionEngine()

        # Most-bearish PROD cycle (SXRV.DE 2026-07-14): three models SELL,
        # vincent_ganne BUY (now disabled in prod, but included here to prove
        # the renormalisation alone is sufficient), the rest HOLD.
        decisions = [
            ModelDecision(signal="SELL", confidence=0.65, strength=SignalStrength.SELL,
                          timestamp=None, model_name="classic", reasoning=""),
            ModelDecision(signal="SELL", confidence=0.85, strength=SignalStrength.SELL,
                          timestamp=None, model_name="llm_visual", reasoning=""),
            ModelDecision(signal="SELL", confidence=0.52, strength=SignalStrength.SELL,
                          timestamp=None, model_name="tensortrade", reasoning=""),
            ModelDecision(signal="BUY", confidence=0.58, strength=SignalStrength.BUY,
                          timestamp=None, model_name="vincent_ganne", reasoning=""),
            # HOLDs (abstainers) — these used to dilute the score silently
            ModelDecision(signal="HOLD", confidence=0.90, strength=SignalStrength.HOLD,
                          timestamp=None, model_name="llm_text", reasoning=""),
            ModelDecision(signal="HOLD", confidence=0.50, strength=SignalStrength.HOLD,
                          timestamp=None, model_name="sentiment", reasoning=""),
            ModelDecision(signal="HOLD", confidence=0.50, strength=SignalStrength.HOLD,
                          timestamp=None, model_name="timesfm", reasoning=""),
            ModelDecision(signal="HOLD", confidence=0.60, strength=SignalStrength.HOLD,
                          timestamp=None, model_name="council", reasoning=""),
        ]
        weights = {
            "classic": 0.082, "llm_visual": 0.114, "tensortrade": 0.066,
            "vincent_ganne": 0.082, "llm_text": 0.094, "sentiment": 0.120,
            "timesfm": 0.105, "council": 0.098,
        }
        # High-volatility regime (as in PROD) applies the *0.8 damping.
        market_data = {"volatility": 0.05, "rsi": 50}

        score = engine._calculate_weighted_score(decisions, weights, market_data)
        signal = engine._determine_final_signal(score)

        self.assertLess(
            score, engine.adaptive_thresholds["sell"],
            f"Weighted score {score:.4f} did not cross SELL threshold "
            f"{engine.adaptive_thresholds['sell']}; SELL remains unreachable.",
        )
        self.assertIn(
            signal, ("SELL", "STRONG_SELL"),
            f"Expected SELL/STRONG_SELL on the most-bearish PROD cycle, got {signal}.",
        )

    def test_sell_threshold_is_reachable(self):
        """The SELL threshold must be above the -0.139 floor observed in PROD."""
        from enhanced_decision_engine import EnhancedDecisionEngine

        engine = EnhancedDecisionEngine()
        # Old buggy value was -0.15; the most bearish cycle reached -0.139
        # (before renorm). The loosened threshold must be > -0.139.
        self.assertGreater(engine.adaptive_thresholds["sell"], -0.139)


class TestVolatilityUnitFix(unittest.TestCase):
    """GO-gate 4 (audit 2026-08-19 C4): the volatility fed to the decision
    engine / weight manager used to be ANNUALIZED while every threshold
    (VOLATILITY_HIGH_THRESHOLD=0.04, regime high_vol=0.03) is DAILY-scale —
    the score was damped x0.8 on EVERY cycle and the regime was permanently
    "volatile"/"crisis"."""

    def test_compute_daily_volatility_is_daily_scale(self):
        from enhanced_trading_example import compute_daily_volatility

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0, 0.01, 60))  # calm daily regime
        vol = compute_daily_volatility(returns)
        self.assertLess(vol, 0.04, "A 1%/day series must NOT look 'high volatility'")
        self.assertGreater(vol, 0.0)
        # The old annualized formula would have returned ~0.01*sqrt(252)=0.159.
        self.assertLess(vol, 0.05)

    def test_compute_daily_volatility_uses_recent_window(self):
        from enhanced_trading_example import compute_daily_volatility, VOLATILITY_WINDOW_DAYS

        calm = pd.Series([0.001] * 60)
        wild_tail = pd.Series([0.05, -0.06] * 10)  # last 20 days: wild
        vol = compute_daily_volatility(pd.concat([calm, wild_tail]).reset_index(drop=True))
        self.assertGreater(vol, 0.04, "A wild recent window must trip the high-vol threshold")
        self.assertLessEqual(len(wild_tail), VOLATILITY_WINDOW_DAYS * 2)

    def test_compute_daily_volatility_insufficient_data_defaults_calm(self):
        from enhanced_trading_example import compute_daily_volatility, DEFAULT_DAILY_VOLATILITY

        self.assertEqual(compute_daily_volatility(pd.Series([], dtype=float)), DEFAULT_DAILY_VOLATILITY)
        self.assertEqual(compute_daily_volatility(None), DEFAULT_DAILY_VOLATILITY)
        self.assertLess(DEFAULT_DAILY_VOLATILITY, 0.04)

    def test_calm_daily_vol_does_not_damp_score(self):
        """A calm ETF (2%/day) must NOT trigger the x0.8 regime damping —
        the exact production bug: annualized ~0.15 tripped it every cycle."""
        from enhanced_decision_engine import EnhancedDecisionEngine

        engine = EnhancedDecisionEngine()
        score = engine._adjust_for_market_regime(0.5, {"volatility": 0.02, "rsi": 50})
        self.assertAlmostEqual(score, 0.5)

    def test_high_daily_vol_still_damps(self):
        from enhanced_decision_engine import EnhancedDecisionEngine

        engine = EnhancedDecisionEngine()
        score = engine._adjust_for_market_regime(0.5, {"volatility": 0.05, "rsi": 50})
        self.assertAlmostEqual(score, 0.4)

    def test_thresholds_rescaled_to_preserve_behaviour(self):
        """After removing the permanent x0.8 damping, thresholds are rescaled
        by 1/0.8 so the EFFECTIVE cut-offs stay identical to the calibrated
        historical behaviour (decision: comparable 30-day demo run)."""
        from enhanced_decision_engine import EnhancedDecisionEngine

        engine = EnhancedDecisionEngine()
        t = engine.adaptive_thresholds
        self.assertAlmostEqual(t["buy"], 0.15)
        self.assertAlmostEqual(t["sell"], -0.125)
        self.assertAlmostEqual(t["strong_buy"], 0.4375)
        self.assertAlmostEqual(t["strong_sell"], -0.5625)
        # Effective equivalence: old_threshold == new_threshold * 0.8
        self.assertAlmostEqual(t["buy"] * 0.8, 0.12)
        self.assertAlmostEqual(t["sell"] * 0.8, -0.10)


class TestEiaCrudeImportsNoDegenerateCache(unittest.TestCase):
    """Bug #4: a 1-row payload was cached with a fresh mtime, hiding staleness.

    Extended 31 July 2026: a 3-row payload that passes the row-count check but
    is months stale in CONTENT must also be refused (the live PROD incident).
    Dates in fixtures are relative to ``now`` so the tests never go stale.
    """

    @staticmethod
    def _periods(months_ago_start: int, count: int) -> list[dict]:
        """Build `count` monthly EIA rows ending `months_ago_start` months back."""
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        rows = []
        # EIA months are anchored to the 1st; walk backwards from (now - start).
        base = datetime(datetime.now().year, datetime.now().month, 1) - relativedelta(months=months_ago_start)
        for i in range(count):
            period = (base - relativedelta(months=i)).strftime("%Y-%m-01")
            rows.append({"period": period, "quantity": str(40000 + i * 100)})
        return rows

    def test_degenerate_payload_not_cached(self):
        from eia_client import EIAClient

        client = EIAClient()
        client._cache = {}  # bypass memory cache

        # Simulate the degenerate upstream response (1 row — the original bug)
        degenerate = self._periods(months_ago_start=1, count=1)
        with patch.object(client, "_make_request", return_value=degenerate), \
             patch.object(client, "_save_to_cache") as mock_save, \
             patch.object(client, "_get_from_cache", return_value=None):
            df = client.get_crude_imports(months=6)

        # The fetcher returns what it got, but MUST NOT cache a < 3-row frame.
        self.assertEqual(len(df), 1, "fetcher should still return the rows it got")
        mock_save.assert_not_called()
        # Sanity: the happy path (>= 3 recent rows) DOES cache.
        full = self._periods(months_ago_start=1, count=6)
        with patch.object(client, "_make_request", return_value=full), \
             patch.object(client, "_save_to_cache") as mock_save2, \
             patch.object(client, "_get_from_cache", return_value=None):
            client.get_crude_imports(months=6)
        mock_save2.assert_called_once()

    def test_stale_content_not_cached_even_with_enough_rows(self):
        """31 July 2026 PROD bug: 3 rows but the latest period was 4 months old.

        The row-count guard (`len >= 3`) passed, so the stale payload was cached
        with a fresh mtime and never refreshed. The content-freshness guard
        (`age > MAX_CRUDE_IMPORTS_AGE_DAYS`) must refuse it.
        """
        from eia_client import EIAClient, MAX_CRUDE_IMPORTS_AGE_DAYS

        client = EIAClient()
        client._cache = {}

        # 3 rows ending 5 months ago — passes the row-count check, fails the
        # freshness check (5 months > 70 days).
        stale = self._periods(months_ago_start=5, count=3)
        self.assertGreater(
            5 * 30, MAX_CRUDE_IMPORTS_AGE_DAYS,
            "test fixture must be older than the staleness threshold",
        )
        with patch.object(client, "_make_request", return_value=stale), \
             patch.object(client, "_save_to_cache") as mock_save, \
             patch.object(client, "_get_from_cache", return_value=None):
            df = client.get_crude_imports(months=6)

        # The fetcher still returns the rows it got...
        self.assertEqual(len(df), 3)
        # ...but MUST NOT cache stale content (the row count is not enough).
        mock_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()

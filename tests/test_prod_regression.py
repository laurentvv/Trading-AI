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


class TestEiaCrudeImportsNoDegenerateCache(unittest.TestCase):
    """Bug #4: a 1-row payload was cached with a fresh mtime, hiding staleness."""

    def test_degenerate_payload_not_cached(self):
        from eia_client import EIAClient

        client = EIAClient()
        client._cache = {}  # bypass memory cache

        # Simulate the degenerate upstream response (1 month, the original bug)
        degenerate = [{"period": "2026-04-01", "quantity": "45878"}]
        with patch.object(client, "_make_request", return_value=degenerate), \
             patch.object(client, "_save_to_cache") as mock_save, \
             patch.object(client, "_get_from_cache", return_value=None):
            df = client.get_crude_imports(months=6)

        # The fetcher returns what it got, but MUST NOT cache a < 3-row frame.
        self.assertEqual(len(df), 1, "fetcher should still return the rows it got")
        mock_save.assert_not_called()
        # Sanity: the happy path (>= 3 rows) DOES cache.
        full = [{"period": f"2026-0{m}-01", "quantity": str(40000 + m * 100)}
                for m in range(1, 7)]
        with patch.object(client, "_make_request", return_value=full), \
             patch.object(client, "_save_to_cache") as mock_save2, \
             patch.object(client, "_get_from_cache", return_value=None):
            client.get_crude_imports(months=6)
        mock_save2.assert_called_once()


if __name__ == "__main__":
    unittest.main()

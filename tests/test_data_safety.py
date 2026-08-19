"""GO-gate 5 tests — data safety: no synthetic macro, cache TTLs, stale refusal.

Audit 2026-08-19 (docs/AUDIT_PROD_INDEPENDANT_2026-08-19.md):
  C5 — data.py "Method 4" fabricated 24 months of RANDOM macro data around a
       default value AND persisted it to cache, with no TTL on macro caches.
  C6 — the price-cache fallback after failed downloads accepted a cache of
       ANY age (trading on week-old prices was possible).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data as data_mod


class TestNoSyntheticMacroData(unittest.TestCase):
    """GO-gate 5 (C5): fabricated data must not exist anywhere in data.py."""

    def test_source_contains_no_random_data_generation(self):
        source = Path(data_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("np.random.normal", source, "Random macro fabrication is forbidden")
        self.assertNotIn("realistic default data", source, "The 'Method 4' block must stay deleted")

    def test_all_sources_failed_returns_empty_not_synthetic(self):
        """get_macro_data_multi_source with every source failing -> empty df,
        nothing written to the macro cache."""
        indicator = "fed_funds_rate"
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache_path = cache_dir / "MULTI_fed_funds_rate_monthly.parquet"
            with patch.object(data_mod, "_get_macro_cache_filepath", return_value=cache_path), \
                 patch.object(data_mod, "get_fred_data_via_pdr", return_value=pd.DataFrame()), \
                 patch.object(data_mod, "_av_fallback", return_value=pd.DataFrame()), \
                 patch.object(data_mod, "ALPHA_VANTAGE_API_KEY", None):
                # Silence the Yahoo path (Method 2) via the yfinance helper.
                with patch.object(data_mod, "_yf_download", return_value=pd.DataFrame()):
                    result = data_mod.get_macro_data_multi_source(indicator)

            self.assertTrue(result is None or (isinstance(result, pd.DataFrame) and result.empty))
            self.assertFalse(cache_path.exists(), "No synthetic data may ever be cached")


class TestMacroCacheTtl(unittest.TestCase):
    """GO-gate 5 (C5): macro caches expire after 7 days (mtime-based)."""

    def _write_cache(self, tmp, age_days):
        path = Path(tmp) / "AV_TEST_monthly.parquet"
        df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "value": [1.0, 2.0, 3.0]})
        df.to_parquet(path)
        old = os.path.getmtime(path) - age_days * 86400
        os.utime(path, (old, old))
        return path

    def test_expired_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_cache(tmp, age_days=10)
            result = data_mod._load_macro_data_from_cache(path)
            self.assertTrue(result.empty)

    def test_fresh_cache_is_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_cache(tmp, age_days=1)
            result = data_mod._load_macro_data_from_cache(path)
            self.assertEqual(len(result), 3)

    def test_default_ttl_is_seven_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_cache(tmp, age_days=5)
            self.assertFalse(data_mod._load_macro_data_from_cache(path).empty)
            path = self._write_cache(tmp, age_days=9)
            self.assertTrue(data_mod._load_macro_data_from_cache(path).empty)


class TestPriceCacheStalenessRefusal(unittest.TestCase):
    """GO-gate 5 (C6): the emergency fallback refuses a >3-day-old cache."""

    def _write_price_cache(self, tmp, ticker, last_date):
        path = Path(tmp) / f"{ticker.replace('.', '_')}_max_with_vix.parquet"
        idx = pd.date_range(end=last_date, periods=50, freq="D")
        df = pd.DataFrame({"Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 100, "VIX": 15}, index=idx)
        df.to_parquet(path)
        return path

    def test_fresh_enough_cache_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_price_cache(tmp, "X_EQ", pd.Timestamp.now() - pd.Timedelta(days=1))
            self.assertTrue(data_mod._price_cache_is_fresh(path))

    def test_stale_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_price_cache(tmp, "X_EQ", pd.Timestamp.now() - pd.Timedelta(days=10))
            self.assertFalse(data_mod._price_cache_is_fresh(path))

    def test_tz_aware_index_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = pd.date_range(
                end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1), periods=10, freq="D", tz="UTC"
            )
            path = Path(tmp) / "TZ_EQ_max_with_vix.parquet"
            df = pd.DataFrame({"Close": 1}, index=idx)
            df.to_parquet(path)
            self.assertTrue(data_mod._price_cache_is_fresh(path))

    def test_get_etf_data_fallback_refuses_stale_cache(self):
        """End-to-end: all downloads fail + stale cache -> raises (cycle
        aborts) instead of silently trading on old prices."""
        boom = RuntimeError("yahoo down")
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            stale_path = self._write_price_cache(tmp, "STALE_EQ", pd.Timestamp.now() - pd.Timedelta(days=8))
            with patch.object(data_mod, "CACHE_DIR", cache_dir), \
                 patch.object(data_mod, "_yf_download", side_effect=boom), \
                 patch.object(data_mod, "_inject_t212_live_price", side_effect=lambda df, t: df), \
                 patch("time.sleep", MagicMock()):
                with self.assertRaises(RuntimeError):
                    data_mod.get_etf_data("STALE_EQ")
            self.assertTrue(stale_path.exists())  # untouched, just refused

    def test_get_etf_data_fallback_accepts_recent_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self._write_price_cache(tmp, "OK_EQ", pd.Timestamp.now() - pd.Timedelta(hours=20))
            with patch.object(data_mod, "CACHE_DIR", cache_dir), \
                 patch.object(data_mod, "_yf_download", side_effect=RuntimeError("yahoo down")), \
                 patch.object(data_mod, "_inject_t212_live_price", side_effect=lambda df, t: df), \
                 patch("time.sleep", MagicMock()):
                hist, _ = data_mod.get_etf_data("OK_EQ")
            self.assertEqual(len(hist), 50)


if __name__ == "__main__":
    unittest.main()

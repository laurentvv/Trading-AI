"""
Benchmark for TensorTrade prediction pipeline.
Measures training time, prediction time, memory usage, and output format consistency
across different dataset sizes (100, 500, 1000 rows).

Usage:
    python -m tests.bench_tensortrade
    pytest tests/bench_tensortrade.py -v -s --timeout=600
"""

import gc
import time
import tracemalloc
import unittest

import numpy as np
import pandas as pd

from tensortrade_model import get_tensortrade_prediction

SIZES = [100, 500, 1000]


def _make_df(n):
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({"Close": prices}, index=dates)


class BenchmarkResults:
    def __init__(self):
        self.rows = []

    def add(self, size, phase, duration_s, mem_peak_mb, result):
        self.rows.append(
            {
                "size": size,
                "phase": phase,
                "duration_s": round(duration_s, 3),
                "mem_peak_mb": round(mem_peak_mb, 2),
                "signal": result["signal"],
                "confidence": round(result["confidence"], 4),
            }
        )

    def print_markdown(self):
        print("\n## TensorTrade Benchmark Results\n")
        print(
            "| Size | Phase | Duration (s) | Peak Mem (MB) | Signal | Confidence |"
        )
        print(
            "|------|-------|-------------|---------------|--------|------------|"
        )
        for r in self.rows:
            print(
                f"| {r['size']} | {r['phase']} | {r['duration_s']} | "
                f"{r['mem_peak_mb']} | {r['signal']} | {r['confidence']} |"
            )
        print()


class TestBenchmarkTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = BenchmarkResults()

    def test_benchmark_all_sizes(self):
        for size in SIZES:
            gc.collect()
            df = _make_df(size)

            tracemalloc.start()
            t0 = time.perf_counter()
            result = get_tensortrade_prediction(df)
            duration = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024 * 1024)

            self.results.add(size, "full_pipeline", duration, peak_mb, result)
            self.assertIn(result["signal"], ["BUY", "SELL", "HOLD"])
            self.assertTrue(0.0 <= result["confidence"] <= 1.0)

        self.results.print_markdown()


class TestOutputFormatConsistency(unittest.TestCase):
    def test_output_format_consistent(self):
        for run in range(3):
            df = _make_df(100)
            result = get_tensortrade_prediction(df)
            self.assertIn(result["signal"], ["BUY", "SELL", "HOLD"], f"Run {run}: invalid signal")
            self.assertTrue(0.0 <= result["confidence"] <= 1.0, f"Run {run}: confidence out of range")


if __name__ == "__main__":
    unittest.main()

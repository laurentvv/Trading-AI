"""
Run Lean backtest from Trading-AI root directory.

Usage:
    python run_lean_backtest.py [--algorithm main.py|TradingAIFrameworkAlgorithm.py] [--validate]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lean_bridge import LeanSignalBridge
from lean_validator import LeanValidator


def main():
    parser = argparse.ArgumentParser(description="Run Lean backtest for Trading-AI")
    parser.add_argument(
        "--algorithm",
        default="main.py",
        choices=["main.py", "TradingAIFrameworkAlgorithm.py"],
        help="Lean algorithm to run",
    )
    parser.add_argument(
        "--export-signals",
        action="store_true",
        help="Export trading journal signals before running backtest",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate results against thresholds",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline vs framework algorithms",
    )
    args = parser.parse_args()

    if args.export_signals:
        print("Exporting Trading-AI signals to Lean format...")
        bridge = LeanSignalBridge()
        out = bridge.export_to_lean_format("TradingAI-Lean/lean_data")
        print(f"Signals exported to {out}")

        bridge.export_lean_insights_json("TradingAI-Lean/lean_data/insights.json")
        summary = bridge.get_ticker_summary()
        for ticker, info in summary.items():
            print(f"  {ticker}: {info['total_signals']} signals "
                  f"(BUY={info['buy_count']}, SELL={info['sell_count']}, HOLD={info['hold_count']})")

        if not args.validate and not args.compare:
            return

    validator = LeanValidator()

    if args.compare:
        print("\nComparing algorithms...")
        results = validator.compare_algorithms([
            {"name": "Baseline Buy-Hold", "file": "main.py"},
            {"name": "TradingAI Framework", "file": "TradingAIFrameworkAlgorithm.py"},
        ])
        for name, metrics in results.items():
            if "error" not in metrics:
                print(f"\n{name}:")
                for k, v in metrics.items():
                    if v is not None:
                        print(f"  {k}: {v}")
    elif args.validate or args.export_signals:
        print(f"\nValidating {args.algorithm}...")
        passed = validator.validate_change(algorithm=args.algorithm)
        sys.exit(0 if passed else 1)
    else:
        print(f"\nRunning backtest with {args.algorithm}...")
        metrics = validator.run_backtest(algorithm=args.algorithm)
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            sys.exit(1)
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

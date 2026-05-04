"""
LeanValidator: Automated validation of Trading-AI changes via Lean backtest.

This module provides a CI/CD-style validation loop:
1. Export Trading-AI signals via LeanSignalBridge
2. Run a Lean backtest using those signals
3. Compare key metrics (Sharpe, MaxDD, Win Rate) against thresholds
4. Report pass/fail for each metric

Usage:
    python -c "from src.lean_validator import LeanValidator; v = LeanValidator(); v.validate_change()"
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LeanValidator:
    """Validates Trading-AI changes by running Lean backtests."""

    DEFAULT_THRESHOLDS = {
        "min_sharpe": 0.5,
        "max_drawdown": 0.25,
        "min_total_return": -0.10,
        "min_win_rate": 0.40,
    }

    def __init__(self, lean_project_path: str = "TradingAI-Lean"):
        self.lean_path = Path(lean_project_path)

    def _find_report(self, output_dir: Path) -> Optional[dict]:
        for candidate in [
            output_dir / "report.json",
            self.lean_path / output_dir.name / "report.json",
            self.lean_path / "backtest-results" / "report.json",
        ]:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f)

        for json_file in self.lean_path.rglob("*.json"):
            if "result" in json_file.name.lower() or "report" in json_file.name.lower():
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and ("statistics" in data or "Sharpe" in data):
                        return data
                except (json.JSONDecodeError, OSError):
                    continue
        return None

    def _extract_metrics(self, report: dict) -> dict:
        stats = report.get("statistics", report)

        def _get(keys):
            for key in keys:
                val = stats.get(key)
                if val is not None:
                    if isinstance(val, str):
                        cleaned = val.replace("%", "").replace("$", "").strip()
                        try:
                            return float(cleaned)
                        except ValueError:
                            return val
                    return val
            return None

        return {
            "sharpe": _get(["Sharpe Ratio", "sharpe", "Sharpe"]),
            "max_drawdown": _get(["Max Drawdown", "max_drawdown", "MaxDrawdown"]),
            "total_return": _get(["Total Return", "total_return", "TotalReturn"]),
            "win_rate": _get(["Win Rate", "win_rate", "WinRate", "Avg Win Rate"]),
            "total_trades": _get(["Total Trades", "total_trades", "TotalTrades"]),
            "sortino": _get(["Sortino Ratio", "sortino", "Sortino"]),
            "alpha": _get(["Alpha", "alpha"]),
            "beta": _get(["Beta", "beta"]),
        }

    def run_backtest(
        self,
        algorithm: str = "main.py",
        output_dir: str = "validation-results",
        timeout: int = 600,
    ) -> dict:
        cmd = [
            "lean", "backtest",
            "--algorithm", algorithm,
            "--output", output_dir,
        ]

        logger.info(f"Running Lean backtest: {' '.join(cmd)}")
        logger.info(f"Working directory: {self.lean_path}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.lean_path),
                timeout=timeout,
            )
        except FileNotFoundError:
            return {"error": "lean CLI not found. Install with: pip install lean"}
        except subprocess.TimeoutExpired:
            return {"error": f"Backtest timed out after {timeout}s"}

        report = self._find_report(Path(output_dir))
        if report:
            return self._extract_metrics(report)

        return {
            "error": "No report found",
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }

    def validate_change(
        self,
        algorithm: str = "TradingAIFrameworkAlgorithm.py",
        thresholds: dict = None,
    ) -> bool:
        thresholds = thresholds or self.DEFAULT_THRESHOLDS
        metrics = self.run_backtest(algorithm=algorithm)

        if "error" in metrics:
            logger.error(f"Backtest failed: {metrics['error']}")
            return False

        passed = True
        results = []

        sharpe_val = None
        if metrics.get("sharpe") is not None:
            try:
                sharpe_val = float(metrics["sharpe"])
            except (ValueError, TypeError):
                sharpe_val = None
        if sharpe_val is not None and sharpe_val < thresholds["min_sharpe"]:
            results.append(f"FAIL: Sharpe {sharpe_val:.2f} < {thresholds['min_sharpe']}")
            passed = False
        elif sharpe_val is not None:
            results.append(f"PASS: Sharpe {sharpe_val:.2f}")

        dd = None
        if metrics.get("max_drawdown") is not None:
            try:
                dd = float(metrics["max_drawdown"])
            except (ValueError, TypeError):
                dd = None
        if dd is not None and abs(dd) > thresholds["max_drawdown"]:
            results.append(f"FAIL: MaxDD {abs(dd):.2%} > {thresholds['max_drawdown']:.2%}")
            passed = False
        elif dd is not None:
            results.append(f"PASS: MaxDD {abs(dd):.2%}")

        tr = None
        if metrics.get("total_return") is not None:
            try:
                tr = float(metrics["total_return"])
            except (ValueError, TypeError):
                tr = None
        if tr is not None and tr < thresholds["min_total_return"]:
            results.append(f"FAIL: Total Return {tr:.2%} < {thresholds['min_total_return']:.2%}")
            passed = False
        elif tr is not None:
            results.append(f"PASS: Total Return {tr:.2%}")

        for line in results:
            logger.info(line)

        if passed:
            logger.info(
                f"VALIDATION PASSED: Sharpe={metrics.get('sharpe', 'N/A')}, "
                f"Return={metrics.get('total_return', 'N/A')}, "
                f"MaxDD={metrics.get('max_drawdown', 'N/A')}"
            )
        else:
            logger.warning("VALIDATION FAILED: One or more thresholds not met.")

        return passed

    def compare_algorithms(self, algorithms: list[dict]) -> dict:
        results = {}
        for algo in algorithms:
            name = algo.get("name", algo["file"])
            metrics = self.run_backtest(algorithm=algo["file"])
            results[name] = metrics
            logger.info(f"--- {name} ---")
            for k, v in metrics.items():
                if v is not None:
                    logger.info(f"  {k}: {v}")
        return results

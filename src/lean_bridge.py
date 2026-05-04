"""
LeanSignalBridge: Converts Trading-AI journal data into Lean-compatible formats.

This bridge parses trading_journal.csv and produces signal files that can be
consumed by the TradingAI-Lean backtesting algorithms. It handles both the
legacy journal format (Signal,Confidence,Risk,LLM_Analysis,Capital_T212) and
the current format (FINAL_SIGNAL,Confidence,Risk_Level,Risk_Adjusted,T212_Capital,
Model_classic,...,Model_vincent_ganne).
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SIGNAL_MAP = {
    "STRONG_BUY": 2,
    "BUY": 1,
    "HOLD": 0,
    "SELL": -1,
    "STRONG_SELL": -2,
}

TICKER_PROXY_MAP = {
    "SXRV.DE": "QQQ",
    "SXRV.FRK": "QQQ",
    "CRUDP.PA": "USO",
    "CRUDP": "USO",
    "^NDX": "QQQ",
    "CL=F": "USO",
    "QQQ": "QQQ",
    "USO": "USO",
}

DEFAULT_WEIGHTS = {
    "classic": 0.10,
    "llm_text": 0.15,
    "llm_visual": 0.10,
    "sentiment": 0.10,
    "timesfm": 0.20,
    "vincent_ganne": 0.15,
    "oil_bench": 0.10,
    "tensortrade": 0.10,
}


def _parse_confidence(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("%", "").replace("€", "").strip()
        try:
            val = float(cleaned)
            return val / 100.0 if val > 1.0 else val
        except ValueError:
            return 0.0
    return 0.0


def _parse_currency(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("€", "").replace(",", ".").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def _parse_model_signal(value: str) -> Optional[dict]:
    if not isinstance(value, str) or value.strip() in ("", "N/A"):
        return None
    match = re.match(r"^(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)\(([0-9.]+)\)$", value.strip())
    if match:
        return {"signal": match.group(1), "confidence": float(match.group(2))}
    signal_upper = value.strip().upper()
    if signal_upper in SIGNAL_MAP:
        return {"signal": signal_upper, "confidence": 0.0}
    return None


class LeanSignalBridge:
    """Converts Trading-AI journal CSV into Lean-compatible signal DataFrames."""

    def __init__(self, journal_path: str = "trading_journal.csv"):
        self.journal_path = Path(journal_path)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Signal": "FINAL_SIGNAL",
            "Risk": "Risk_Level",
            "Capital_T212": "T212_Capital",
        }
        return df.rename(columns=rename_map)

    def journal_to_signals(self) -> pd.DataFrame:
        if not self.journal_path.exists():
            raise FileNotFoundError(f"Journal not found: {self.journal_path}")

        with open(self.journal_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
        expected_cols = header_line.split(",")
        df = pd.read_csv(
            self.journal_path,
            header=None,
            names=expected_cols + [f"extra_{i}" for i in range(20)],
            skiprows=1,
            on_bad_lines="skip",
            encoding="utf-8",
        )
        df = df[[c for c in expected_cols if c in df.columns]]
        df = self._normalize_columns(df)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        records = []
        for _, row in df.iterrows():
            ticker = row.get("Ticker", "")
            lean_ticker = TICKER_PROXY_MAP.get(ticker, ticker.split(".")[0])
            confidence = _parse_confidence(row.get("Confidence", 0))
            signal_str = str(row.get("FINAL_SIGNAL", "HOLD")).upper()
            if signal_str not in SIGNAL_MAP:
                signal_str = "HOLD"

            record = {
                "date": row["Timestamp"],
                "ticker": ticker,
                "lean_ticker": lean_ticker,
                "signal": signal_str,
                "signal_value": SIGNAL_MAP[signal_str],
                "confidence": confidence,
                "risk_level": str(row.get("Risk_Level", "UNKNOWN")),
                "risk_adjusted": str(row.get("Risk_Adjusted", signal_str)),
                "t212_capital": _parse_currency(row.get("T212_Capital", 0)),
            }

            for col in df.columns:
                if col.startswith("Model_"):
                    model_name = col.replace("Model_", "")
                    parsed = _parse_model_signal(str(row.get(col, "")))
                    if parsed:
                        record[f"model_{model_name}_signal"] = parsed["signal"]
                        record[f"model_{model_name}_confidence"] = parsed["confidence"]
                        record[f"model_{model_name}_value"] = SIGNAL_MAP.get(
                            parsed["signal"], 0
                        )

            records.append(record)

        return pd.DataFrame(records)

    def export_to_lean_format(self, output_dir: str = "lean_data") -> Path:
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        signals = self.journal_to_signals()

        for ticker in signals["lean_ticker"].unique():
            ticker_signals = signals[signals["lean_ticker"] == ticker].sort_values("date")
            filename = f"{ticker}_signals.csv"
            ticker_signals.to_csv(out / filename, index=False)
            logger.info(f"Exported {len(ticker_signals)} signals for {ticker} -> {out / filename}")

        return out

    def compute_weighted_score(self, signals_df: pd.DataFrame) -> pd.Series:
        weighted = pd.Series(0.0, index=signals_df.index)
        for model_name, weight in DEFAULT_WEIGHTS.items():
            col_signal = f"model_{model_name}_value"
            col_conf = f"model_{model_name}_confidence"
            if col_signal in signals_df.columns and col_conf in signals_df.columns:
                weighted += signals_df[col_signal].fillna(0) * signals_df[col_conf].fillna(0) * weight
        return weighted

    def export_lean_insights_json(self, output_path: str = "lean_data/insights.json") -> Path:
        signals = self.journal_to_signals()
        weighted = self.compute_weighted_score(signals)
        signals["weighted_score"] = weighted

        insights = []
        for _, row in signals.iterrows():
            direction = "UP" if row["weighted_score"] > 0.05 else (
                "DOWN" if row["weighted_score"] < -0.05 else "FLAT"
            )
            magnitude = min(0.10, abs(row["weighted_score"]))
            insights.append({
                "date": row["date"].isoformat(),
                "ticker": row["lean_ticker"],
                "direction": direction,
                "magnitude": round(magnitude, 4),
                "confidence": round(row["confidence"], 4),
                "weighted_score": round(row["weighted_score"], 4),
            })

        out_path = Path(output_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2, default=str)
        logger.info(f"Exported {len(insights)} insights to {out_path}")
        return out_path

    def get_ticker_summary(self) -> dict:
        signals = self.journal_to_signals()
        summary = {}
        for ticker in signals["ticker"].unique():
            t_signals = signals[signals["ticker"] == ticker]
            summary[ticker] = {
                "total_signals": len(t_signals),
                "buy_count": (t_signals["signal"] == "BUY").sum()
                             + (t_signals["signal"] == "STRONG_BUY").sum(),
                "sell_count": (t_signals["signal"] == "SELL").sum()
                              + (t_signals["signal"] == "STRONG_SELL").sum(),
                "hold_count": (t_signals["signal"] == "HOLD").sum(),
                "avg_confidence": t_signals["confidence"].mean(),
                "lean_proxy": TICKER_PROXY_MAP.get(ticker, ticker),
            }
        return summary

"""
Shared helper functions for benchmark test files.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data import get_etf_data

T212_FEE_RATE = 0.001
BUDGET = 1000.0
START_OFFSET = 800
OUTPUT_DIR = Path("logs_prod")


def strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def truncate_10y(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=10)
    return df.loc[df.index >= cutoff]


def download(ticker: str) -> pd.DataFrame:
    try:
        df, _ = get_etf_data(ticker)
        return truncate_10y(strip_tz(df))
    except Exception:
        import yfinance as yf

        raw = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        return strip_tz(raw[["Close"]].dropna())


def simulate(dates, prices, signals, budget=BUDGET, partial_sizing=False):
    cash = budget
    position = 0.0
    entry_price = 0.0
    ec = []
    trades = []

    for dt, px, (sig, conf) in zip(dates, prices, signals):
        size = min(conf * 1.5, 1.0) if partial_sizing else 1.0
        size = max(size, 0.25)

        if sig in ("BUY", "STRONG_BUY") and position == 0 and cash > 0:
            invest = cash * size
            fee = invest * T212_FEE_RATE
            bought = (invest - fee) / px
            position = bought
            entry_price = px
            trades.append({"date": str(dt.date()), "type": "BUY", "price": round(px, 2), "size_pct": round(size * 100)})
            cash -= invest
        elif sig in ("SELL", "STRONG_SELL") and position > 0:
            sell_qty = position
            proceeds = sell_qty * px
            fee = proceeds * T212_FEE_RATE
            pnl = (px - entry_price) * sell_qty - fee
            cash += proceeds - fee
            trades.append({"date": str(dt.date()), "type": "SELL", "price": round(px, 2), "pnl": round(pnl, 2)})
            position = 0
            entry_price = 0

        equity = cash + (position * px if position > 0 else 0)
        ec.append({"date": dt, "equity": round(equity, 2)})

    return metrics(pd.DataFrame(ec), trades, budget)


def metrics(ec, trades, budget=BUDGET):
    empty = {
        "equity_curve": ec,
        "trades": trades,
        "total_return_pct": 0,
        "annualized_return_pct": 0,
        "max_drawdown_pct": 0,
        "sharpe_ratio": 0,
        "n_trades": len(trades),
        "win_rate_pct": 0,
        "final_equity": budget,
    }
    if len(ec) == 0:
        return empty
    last = ec["equity"].iloc[-1]
    ret = (last / budget) - 1
    n = len(ec)
    ann = (1 + ret) ** (252 / max(n, 1)) - 1
    closed = [t for t in trades if t["type"] == "SELL"]
    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
    wr = wins / len(closed) if closed else 0
    dr = ec["equity"].pct_change().dropna()
    sh = (dr.mean() / dr.std()) * np.sqrt(252) if len(dr) > 1 and dr.std() > 0 else 0.0
    dd = ((ec["equity"].cummax() - ec["equity"]) / ec["equity"].cummax()).max()
    return {
        "equity_curve": ec,
        "trades": trades,
        "final_equity": round(last, 2),
        "total_return_pct": round(ret * 100, 2),
        "annualized_return_pct": round(ann * 100, 2),
        "max_drawdown_pct": round(dd * 100, 2),
        "sharpe_ratio": round(sh, 2),
        "n_trades": len(trades),
        "win_rate_pct": round(wr * 100, 1),
    }

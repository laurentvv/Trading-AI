"""
Standalone Production Backtest Engine.

Replays actual Trading-AI signals from logs_prod/trading_journal.csv against
real price data from data_cache/ parquet files. Applies T212 fees (0.1%) and
simulates portfolio evolution per ticker.

Usage:
    uv run python backtest_prod.py
"""

import sqlite3
import json
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

T212_FEE_RATE = 0.001
TICKER_PRICE_FILES = {
    "SXRV.DE": "data_cache/SXRV_DE_max_with_vix.parquet",
    "CRUDP.PA": "data_cache/CRUDP_PA_max_with_vix.parquet",
}
SIGNAL_MAP = {"STRONG_BUY": 2, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG_SELL": -2}
BUDGET_PER_TICKER = 1000.0


def load_journal(path: str = "logs_prod/trading_journal.csv") -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    expected_cols = header_line.split(",")
    df = pd.read_csv(
        path,
        header=None,
        names=expected_cols + [f"extra_{i}" for i in range(10)],
        skiprows=1,
        on_bad_lines="skip",
        encoding="utf-8",
    )
    df = df[[c for c in expected_cols if c in df.columns]]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["date"] = df["Timestamp"].dt.date
    df["signal_value"] = df["FINAL_SIGNAL"].str.upper().map(SIGNAL_MAP).fillna(0).astype(int)

    def parse_conf(v):
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace("%", "").replace("\u20ac", "").strip()
        try:
            val = float(s)
            return val / 100.0 if val > 1.0 else val
        except ValueError:
            return 0.0

    df["confidence"] = df["Confidence"].apply(parse_conf)
    return df


def load_prices(ticker: str) -> pd.Series:
    path = TICKER_PRICE_FILES[ticker]
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df["Close"].sort_index()


def aggregate_daily_signals(journal: pd.DataFrame) -> pd.DataFrame:
    daily = []
    for (ticker, d), grp in journal.groupby(["Ticker", "date"]):
        avg_signal = grp["signal_value"].mean()
        if avg_signal > 0.5:
            final = "BUY"
        elif avg_signal < -0.5:
            final = "SELL"
        else:
            final = "HOLD"
        daily.append(
            {
                "ticker": ticker,
                "date": d,
                "signal": final,
                "signal_value": round(avg_signal, 2),
                "confidence": grp["confidence"].mean(),
                "n_signals": len(grp),
            }
        )
    return pd.DataFrame(daily)


def run_backtest(daily_signals: pd.DataFrame, prices: dict, initial_cash: float = BUDGET_PER_TICKER) -> dict:
    results = {}
    for ticker in daily_signals["ticker"].unique():
        ts = daily_signals[daily_signals["ticker"] == ticker].sort_values("date")
        if ticker not in prices:
            continue
        px = prices[ticker]

        cash = initial_cash
        position = 0.0
        entry_price = 0.0
        trades = []
        equity_curve = []
        max_equity = initial_cash
        max_drawdown = 0.0

        for _, row in ts.iterrows():
            d = row["date"]
            px_date = pd.to_datetime(d)
            if px_date not in px.index:
                nearby = px.index[px.index >= px_date]
                if len(nearby) == 0:
                    continue
                px_date = nearby[0]
            price = px.loc[px_date]

            signal = row["signal"]

            if signal in ("BUY", "STRONG_BUY") and position == 0 and cash > 0:
                fee = cash * T212_FEE_RATE
                invest = cash - fee
                qty = invest / price
                position = qty
                entry_price = price
                trades.append({"date": d, "type": "BUY", "price": price, "qty": qty, "fee": fee})
                cash = 0

            elif signal in ("SELL", "STRONG_SELL") and position > 0:
                proceeds = position * price
                fee = proceeds * T212_FEE_RATE
                pnl = (price - entry_price) * position - fee
                cash = proceeds - fee
                trades.append({"date": d, "type": "SELL", "price": price, "qty": position, "fee": fee, "pnl": pnl})
                position = 0
                entry_price = 0

            equity = cash + (position * price if position > 0 else 0)
            equity_curve.append({"date": d, "equity": equity, "price": price, "position": position})
            max_equity = max(max_equity, equity)
            dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        if len(equity_curve) == 0:
            continue
        ec = pd.DataFrame(equity_curve)
        last_equity = ec["equity"].iloc[-1]
        total_return = (last_equity / initial_cash) - 1
        n_buys = sum(1 for t in trades if t["type"] == "BUY")
        n_sells = sum(1 for t in trades if t["type"] == "SELL")
        total_fees = sum(t.get("fee", 0) for t in trades)
        closed_trades = [t for t in trades if t["type"] == "SELL"]
        wins = sum(1 for t in closed_trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(closed_trades) if closed_trades else 0

        daily_returns = ec["equity"].pct_change().dropna()
        sharpe = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        results[ticker] = {
            "initial_cash": initial_cash,
            "final_equity": round(last_equity, 2),
            "total_return": round(total_return * 100, 2),
            "total_return_pct": f"{total_return * 100:.2f}%",
            "max_drawdown": round(max_drawdown * 100, 2),
            "max_drawdown_pct": f"{max_drawdown * 100:.2f}%",
            "sharpe_ratio": round(sharpe, 2),
            "total_fees": round(total_fees, 2),
            "n_trades": len(trades),
            "n_buys": n_buys,
            "n_sells": n_sells,
            "n_closed": len(closed_trades),
            "win_rate": f"{win_rate * 100:.1f}%",
            "trades": trades,
            "equity_curve": ec,
        }
    return results


def run_baseline(prices: dict, start_date: date, end_date: date, initial_cash: float = BUDGET_PER_TICKER) -> dict:
    results = {}
    for ticker, px in prices.items():
        mask = (px.index >= pd.Timestamp(start_date)) & (px.index <= pd.Timestamp(end_date))
        px_period = px[mask]
        if len(px_period) == 0:
            continue
        entry = px_period.iloc[0]
        fee = initial_cash * T212_FEE_RATE
        shares = (initial_cash - fee) / entry
        final = px_period.iloc[-1]
        final_equity = shares * final
        total_return = (final_equity / initial_cash) - 1
        equity_curve = pd.DataFrame(
            {
                "date": px_period.index,
                "equity": shares * px_period.values,
            }
        )
        daily_returns = equity_curve["equity"].pct_change().dropna()
        sharpe = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        cummax = equity_curve["equity"].cummax()
        drawdown = (cummax - equity_curve["equity"]) / cummax
        max_dd = drawdown.max()

        results[ticker] = {
            "initial_cash": initial_cash,
            "final_equity": round(final_equity, 2),
            "total_return": round(total_return * 100, 2),
            "total_return_pct": f"{total_return * 100:.2f}%",
            "max_drawdown": round(max_dd * 100, 2),
            "max_drawdown_pct": f"{max_dd * 100:.2f}%",
            "sharpe_ratio": round(sharpe, 2),
            "entry_price": round(entry, 2),
            "exit_price": round(final, 2),
        }
    return results


def print_report(journal_stats: dict, backtest: dict, baseline: dict, t212_trades: list):
    print("=" * 70)
    print("  TRADING-AI PRODUCTION BACKTEST REPORT")
    print("=" * 70)

    print("\n--- SIGNAL SUMMARY (from logs_prod/trading_journal.csv) ---")
    for ticker, stats in journal_stats.items():
        print(
            f"  {ticker}: {stats['total']} signals | "
            f"BUY={stats['buy']} SELL={stats['sell']} HOLD={stats['hold']} | "
            f"Avg Conf={stats['avg_conf']:.1%}"
        )

    print("\n--- REAL T212 TRADES (from logs_prod/trading_history.db) ---")
    if t212_trades:
        for t in t212_trades:
            print(f"  {t[1]} {t[3]} {t[4]:.4f}x {t[5]:.2f} EUR (cost={t[6]:.2f}, src={t[7]})")
    else:
        print("  (none)")

    print("\n--- SIGNAL-FOLLOWING BACKTEST (daily aggregated) ---")
    print(f"  {'Ticker':<10} {'Return':>10} {'MaxDD':>10} {'Sharpe':>8} {'Fees':>8} {'Trades':>8} {'WinRate':>8}")
    print(f"  {'-' * 62}")
    for ticker, r in backtest.items():
        print(
            f"  {ticker:<10} {r['total_return_pct']:>10} {r['max_drawdown_pct']:>10} "
            f"{r['sharpe_ratio']:>8.2f} {r['total_fees']:>8.2f} "
            f"{r['n_buys']}B/{r['n_sells']}S {' ' * 4}{r['win_rate']:>8}"
        )

    print("\n--- BASELINE BUY & HOLD ---")
    print(f"  {'Ticker':<10} {'Return':>10} {'MaxDD':>10} {'Sharpe':>8} {'Entry':>10} {'Exit':>10}")
    print(f"  {'-' * 58}")
    for ticker, r in baseline.items():
        print(
            f"  {ticker:<10} {r['total_return_pct']:>10} {r['max_drawdown_pct']:>10} "
            f"{r['sharpe_ratio']:>8.2f} {r['entry_price']:>10.2f} {r['exit_price']:>10.2f}"
        )

    print("\n--- COMPARISON ---")
    for ticker in backtest:
        if ticker in baseline:
            alpha = backtest[ticker]["total_return"] - baseline[ticker]["total_return"]
            emoji = "+" if alpha > 0 else ""
            print(
                f"  {ticker}: Signal strategy {backtest[ticker]['total_return_pct']} "
                f"vs Buy&Hold {baseline[ticker]['total_return_pct']} "
                f"=> Alpha: {emoji}{alpha:.2f}%"
            )

    print("\n" + "=" * 70)


def main():
    journal = load_journal()
    print(f"Loaded {len(journal)} signals from journal")

    start_date = journal["date"].min()
    end_date = journal["date"].max()
    print(f"Period: {start_date} to {end_date}")

    journal_stats = {}
    for ticker in journal["Ticker"].unique():
        tj = journal[journal["Ticker"] == ticker]
        journal_stats[ticker] = {
            "total": len(tj),
            "buy": tj["FINAL_SIGNAL"].isin(["BUY", "STRONG_BUY"]).sum(),
            "sell": tj["FINAL_SIGNAL"].isin(["SELL", "STRONG_SELL"]).sum(),
            "hold": (tj["FINAL_SIGNAL"] == "HOLD").sum(),
            "avg_conf": tj["confidence"].mean(),
        }

    prices = {}
    for ticker in TICKER_PRICE_FILES:
        prices[ticker] = load_prices(ticker)
        print(f"Loaded {len(prices[ticker])} price bars for {ticker}")

    daily = aggregate_daily_signals(journal)
    print(f"Aggregated to {len(daily)} daily signals")

    backtest = run_backtest(daily, prices)
    baseline = run_baseline(prices, start_date, end_date)

    db_path = Path("logs_prod/trading_history.db")
    t212_trades = []
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT * FROM transactions ORDER BY date")
        t212_trades = cur.fetchall()
        conn.close()

    print_report(journal_stats, backtest, baseline, t212_trades)

    out_dir = Path("logs_prod")
    for ticker, r in backtest.items():
        if isinstance(r.get("equity_curve"), pd.DataFrame):
            r["equity_curve"].to_csv(out_dir / f"backtest_equity_{ticker.replace('.', '_')}.csv", index=False)

    report_data = {
        "period": {"start": str(start_date), "end": str(end_date)},
        "signal_strategy": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("trades", "equity_curve")} for k, v in backtest.items()
        },
        "baseline": baseline,
        "t212_real_trades": len(t212_trades),
    }
    with open(out_dir / "backtest_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\nSaved: {out_dir}/backtest_report.json + equity curves")


if __name__ == "__main__":
    main()

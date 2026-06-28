"""
Backtest comparant la QUALITE des decisions de trading entre deux modeles LLM.

Principe:
  - Charge les donnees historiques d'un ticker via data_cache/parquet
  - Calcule les indicateurs techniques (RSI, MACD, BB, Trends)
  - Pour chaque date de rebalancement (toutes les 2 semaines), envoie
    les donnees techniques au LLM et recupere un signal BUY/SELL/HOLD
  - Simule les trades avec frais T212 (0.1%)
  - Compare: rendement total, Sharpe, max drawdown, win rate, alpha vs Buy&Hold

Models:
  A) hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M  (production)
  B) gemma4:12b-it-qat                            (new QAT)

Usage:
    .venv\\Scripts\\python.exe -m tests.backtest_llm_quality
    .venv\\Scripts\\python.exe -m tests.backtest_llm_quality --ticker CRUDP.PA --months 6
    .venv\\Scripts\\python.exe -m tests.backtest_llm_quality --months 3 --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import get_etf_data
from src.features import create_technical_indicators, create_features
from src.llm_client import (
    OLLAMA_API_URL,
    OLLAMA_BASE_URL,
    SCHEMA_TRADING_DECISION,
    _query_ollama,
)

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

MODELS = {
    "A_Q4KM": "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M",
    "C_LFM": "lfm2.5:8b-a1b-q8_0",
}

LABELS = {
    MODELS["A_Q4KM"]: "A (Q4_K_M prod)",
    MODELS["C_LFM"]: "C (LFM 2.5 8B)",
}

T212_FEE_RATE = 0.001
BUDGET = 1000.0
KEEP_ALIVE = "30m"


SYSTEM_PROMPT = "<|think|> You are an expert financial analyst. Your task is to analyze market data and news to provide a trading decision in a valid JSON format. Output ONLY the JSON object requested — never add a 'thought' key."


def load_data(ticker: str, months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 2y of data for indicator warmup, return (full_df, backtest_period_df)."""
    raw, _ = get_etf_data(ticker, period="2y", force_refresh=False)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    for col in ("Open", "High", "Low"):
        if col not in raw.columns:
            raw[col] = raw["Close"]
    if "Volume" not in raw.columns:
        raw["Volume"] = 0
    raw = create_technical_indicators(raw)
    full = create_features(raw)
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months)
    bt_period = full.loc[full.index >= cutoff].copy()
    return full, bt_period


def get_rebalance_dates(df: pd.DataFrame, freq_days: int = 14) -> list:
    dates = []
    last = None
    for dt in df.index:
        if last is None or (dt - last).days >= freq_days:
            dates.append(dt)
            last = dt
    return dates


def build_backtest_prompt(row: pd.Series, ticker: str) -> str:
    rsi_label = "Overbought" if row.get("RSI", 50) > 70 else "Oversold" if row.get("RSI", 50) < 30 else "Neutral"
    trend_short = "Bullish" if row.get("Trend_Short", 0) == 1 else "Bearish" if row.get("Trend_Short", 0) == -1 else "Neutral"
    trend_long = "Bullish" if row.get("Trend_Long", 0) == 1 else "Bearish" if row.get("Trend_Long", 0) == -1 else "Neutral"
    macd_signal = "Bullish crossover" if row.get("MACD_Bull", 0) == 1 else "Bearish"
    rsi_oversold = "YES" if row.get("RSI_Oversold", 0) == 1 else "no"
    rsi_overbought = "YES" if row.get("RSI_Overbought", 0) == 1 else "no"

    return f"""You are analyzing {ticker} market data. Make a trading decision based ONLY on the technical indicators below.

IMPORTANT RULES:
1. You MUST be decisive. Do NOT default to HOLD unless indicators are genuinely mixed.
2. RSI < 30 (Oversold) = strong BUY signal. RSI > 70 (Overbought) = strong SELL signal.
3. Bullish trend + MACD bullish = BUY. Bearish trend + MACD bearish = SELL.
4. BB_Position < 0.2 (near lower band) = BUY opportunity. BB_Position > 0.8 (near upper band) = SELL opportunity.
5. If both short-term and long-term trends agree, trade WITH the trend.
6. Confidence should reflect signal strength: strong consensus = high confidence (>0.7), mixed = moderate (0.4-0.6).

**Technical Data for {ticker}:**
- Close Price: {row.get('Close', 0):.2f}
- RSI (14): {row.get('RSI', 50):.2f} ({rsi_label})
- Oversold: {rsi_oversold} | Overbought: {rsi_overbought}
- MACD: {row.get('MACD', 0):.4f} | Signal: {row.get('MACD_Signal', 0):.4f} ({macd_signal})
- BB Position: {row.get('BB_Position', 0.5):.2f} (0=Bottom band, 1=Top band)
- Short-term Trend (MA5 vs MA20): {trend_short}
- Long-term Trend (MA20 vs MA50): {trend_long}
- Volatility (20d): {row.get('Volatility', 0):.6f}

Provide your decision ONLY as a valid JSON object:
{{
  "signal": "BUY | SELL | HOLD",
  "confidence": <float 0.0 to 1.0>,
  "analysis": "1-sentence justification referencing specific indicator values."
}}"""


def query_llm_for_signal(model: str, row: pd.Series, ticker: str) -> dict:
    prompt = build_backtest_prompt(row, ticker)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "format": SCHEMA_TRADING_DECISION,
        "options": {"temperature": 0.4, "num_predict": 512},
        "system": SYSTEM_PROMPT,
    }
    result = _query_ollama(payload, expected_keys=["signal", "confidence", "analysis"])
    signal = result.get("signal", "HOLD").upper()
    if signal not in ("BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"):
        signal = "HOLD"
    confidence = result.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    analysis = result.get("analysis", "")
    return {"signal": signal, "confidence": confidence, "analysis": analysis}


def simulate_trades(dates, prices, signals, budget=BUDGET) -> dict:
    cash = budget
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = []

    for dt, px, sig_conf in zip(dates, prices, signals):
        sig = sig_conf["signal"]
        conf = sig_conf["confidence"]

        if sig in ("BUY", "STRONG_BUY") and position == 0 and cash > 0:
            size = min(conf * 1.5, 1.0)
            size = max(size, 0.25)
            invest = cash * size
            fee = invest * T212_FEE_RATE
            bought = (invest - fee) / px
            position = bought
            entry_price = px
            trades.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "type": "BUY",
                "price": round(px, 2),
                "size_pct": round(size * 100),
                "signal": sig,
                "confidence": round(conf, 2),
            })
            cash -= invest

        elif sig in ("SELL", "STRONG_SELL") and position > 0:
            proceeds = position * px
            fee = proceeds * T212_FEE_RATE
            pnl = (px - entry_price) * position - fee
            cash += proceeds - fee
            trades.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "type": "SELL",
                "price": round(px, 2),
                "pnl": round(pnl, 2),
                "signal": sig,
                "confidence": round(conf, 2),
            })
            position = 0
            entry_price = 0

        equity = cash + (position * px if position > 0 else 0)
        equity_curve.append({"date": dt, "equity": round(equity, 2)})

    ec = pd.DataFrame(equity_curve)
    return _calc_metrics(ec, trades, budget)


def simulate_buy_hold(dates, prices, budget=BUDGET) -> dict:
    if len(prices) == 0:
        return {"total_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0, "win_rate_pct": 0, "n_trades": 0, "final_equity": budget}
    entry_fee = budget * T212_FEE_RATE
    shares = (budget - entry_fee) / prices[0]
    equity_curve = pd.DataFrame({"date": dates, "equity": [round(shares * p, 2) for p in prices]})
    trades = [{"date": str(dates[0].date()) if hasattr(dates[0], "date") else str(dates[0]), "type": "BUY", "price": round(prices[0], 2)}]
    return _calc_metrics(equity_curve, trades, budget)


def _calc_metrics(ec: pd.DataFrame, trades: list, budget: float) -> dict:
    if len(ec) == 0:
        return {
            "total_return_pct": 0, "annualized_return_pct": 0,
            "max_drawdown_pct": 0, "sharpe_ratio": 0,
            "n_trades": len(trades), "win_rate_pct": 0,
            "final_equity": budget, "trades": trades,
            "equity_curve": ec,
        }
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
        "total_return_pct": round(ret * 100, 2),
        "annualized_return_pct": round(ann * 100, 2),
        "max_drawdown_pct": round(dd * 100, 2),
        "sharpe_ratio": round(sh, 2),
        "n_trades": len(trades),
        "n_closed": len(closed),
        "win_rate_pct": round(wr * 100, 1),
        "final_equity": round(last, 2),
        "trades": trades,
        "equity_curve": ec,
    }


def warmup_model(model: str) -> float:
    print(f"  {DIM}Warming up {model[:40]}...{RESET}", end="", flush=True)
    import requests as req
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "prompt": "Hi",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_predict": 3},
    }
    try:
        req.post(OLLAMA_API_URL, json=payload, timeout=300)
    except Exception as e:
        print(f" {RED}FAIL: {e}{RESET}")
        return -1
    elapsed = time.perf_counter() - t0
    print(f" {GREEN}OK{RESET} ({elapsed:.1f}s)")
    return elapsed


def unload_model(model: str):
    import requests as req
    try:
        req.post(OLLAMA_API_URL, json={"model": model, "prompt": "", "stream": False, "keep_alive": "0"}, timeout=30)
    except Exception:
        pass


def print_trade_log(trades: list):
    for t in trades:
        tp = t["type"]
        pnl_str = ""
        if "pnl" in t:
            pnl_str = f" pnl={t['pnl']:+.2f}"
        sig = t.get("signal", "")
        conf = t.get("confidence", "")
        print(f"    {tp:4s} {t['date']} @{t['price']:.2f} {sig:12s} conf={conf}{pnl_str}")


def run_backtest_for_model(model_key: str, model_id: str, indicator_data: pd.DataFrame,
                           rebalance_dates: list, ticker: str) -> dict:
    print(f"\n  {BOLD}--- Running backtest for {model_key} ({model_id[:50]}) ---{RESET}")

    warmup_model(model_id)

    signals = []
    prices = []
    actual_dates = []

    total = len(rebalance_dates)
    for i, dt in enumerate(rebalance_dates):
        if dt not in indicator_data.index:
            nearby = indicator_data.index[indicator_data.index >= dt]
            if len(nearby) == 0:
                continue
            dt = nearby[0]

        row = indicator_data.loc[dt]

        print(f"    {DIM}[{i+1}/{total}]{RESET} {dt.strftime('%Y-%m-%d')} "
              f"Close={row['Close']:.2f} RSI={row['RSI']:.1f} "
              f"Trend={'B' if row.get('Trend_Short', 0) == 1 else 'A' if row.get('Trend_Short', 0) == -1 else 'N'}... ", end="", flush=True)

        t0 = time.perf_counter()
        try:
            result = query_llm_for_signal(model_id, row, ticker)
        except Exception as e:
            print(f"{RED}ERR: {e}{RESET}")
            result = {"signal": "HOLD", "confidence": 0.0, "analysis": f"error: {e}"}

        elapsed = time.perf_counter() - t0
        sig = result["signal"]
        conf = result["confidence"]
        print(f"{sig:12s} conf={conf:.2f} ({elapsed:.1f}s)")

        signals.append(result)
        prices.append(row["Close"])
        actual_dates.append(dt)

    unload_model(model_id)

    sim = simulate_trades(actual_dates, prices, signals)

    from collections import Counter
    sig_dist = Counter(s["signal"] for s in signals)
    conf_vals = [s["confidence"] for s in signals]

    return {
        "model_key": model_key,
        "model_id": model_id,
        "metrics": sim,
        "signal_distribution": dict(sig_dist),
        "avg_confidence": round(sum(conf_vals) / max(len(conf_vals), 1), 2),
        "n_decisions": len(signals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SXRV.DE", help="Ticker (default: SXRV.DE)")
    parser.add_argument("--months", type=int, default=6, help="Backtest period in months (default: 6)")
    parser.add_argument("--freq", type=int, default=14, help="Rebalance frequency in days (default: 14)")
    parser.add_argument("--quick", action="store_true", help="Only 3 months, 21-day rebalance")
    args = parser.parse_args()

    if args.quick:
        args.months = min(args.months, 3)
        args.freq = max(args.freq, 21)

    print(f"{BOLD}=== LLM Quality Backtest ==={RESET}")
    print(f"  Ticker:    {args.ticker}")
    print(f"  Period:    {args.months} months")
    print(f"  Rebalance: every {args.freq} days")

    try:
        import requests as req
        health = req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        health.raise_for_status()
        available = [m.get("name") for m in health.json().get("models", [])]
        for mk, mid in MODELS.items():
            if mid not in available:
                print(f"{YELLOW}Model {mid} not found, pulling...{RESET}")
                req.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": mid}, timeout=600)
        print(f"{GREEN}Ollama OK{RESET}")
    except Exception as e:
        print(f"{RED}Ollama unavailable ({e}){RESET}")
        return 2

    print(f"\n{BOLD}Loading data for {args.ticker}...{RESET}")
    full_df, bt_df = load_data(args.ticker, args.months)
    print(f"  Full data: {len(full_df)} bars from {full_df.index[0].date()} to {full_df.index[-1].date()}")
    print(f"  Backtest period: {len(bt_df)} bars from {bt_df.index[0].date()} to {bt_df.index[-1].date()}")

    rebalance_dates = get_rebalance_dates(bt_df, freq_days=args.freq)
    print(f"  Rebalance points: {len(rebalance_dates)}")
    print(f"  Period: {rebalance_dates[0].date()} to {rebalance_dates[-1].date()}")

    results = {}
    for mk, mid in MODELS.items():
        r = run_backtest_for_model(mk, mid, full_df, rebalance_dates, args.ticker)
        results[mk] = r

    bh_prices = [full_df.loc[dt, "Close"] for dt in rebalance_dates if dt in full_df.index]
    bh_dates_filtered = [dt for dt in rebalance_dates if dt in full_df.index]
    buy_hold = simulate_buy_hold(bh_dates_filtered, bh_prices)

    print(f"\n{'='*80}")
    print(f"{BOLD} QUALITY BACKTEST RESULTS{RESET}")
    print(f"{'='*80}")
    print(f"  Ticker: {args.ticker} | Period: {args.months} months | Rebalance: {args.freq}d")

    print(f"\n{BOLD}--- Model Results ---{RESET}")
    for mk, r in results.items():
        m = r["metrics"]
        print(f"\n  {CYAN}{mk} ({r['model_id'][:50]}){RESET}")
        print(f"    Decisions:     {r['n_decisions']}")
        print(f"    Signals:       {r['signal_distribution']}")
        print(f"    Avg conf:      {r['avg_confidence']:.2f}")
        print(f"    Return:        {GREEN if m['total_return_pct'] > 0 else RED}{m['total_return_pct']:+.2f}%{RESET}")
        print(f"    Annualized:    {m['annualized_return_pct']:+.2f}%")
        print(f"    Max Drawdown:  {RED}{m['max_drawdown_pct']:.2f}%{RESET}")
        print(f"    Sharpe:        {m['sharpe_ratio']:.2f}")
        print(f"    Win rate:      {m['win_rate_pct']:.1f}% ({m['n_closed']} closed trades)")
        print(f"    Final equity:  {m['final_equity']:.2f}")
        if m["trades"]:
            print("    Trade log:")
            print_trade_log(m["trades"])

    print(f"\n  {YELLOW}--- Buy & Hold ---{RESET}")
    print(f"    Return:        {buy_hold['total_return_pct']:+.2f}%")
    print(f"    Max Drawdown:  {buy_hold['max_drawdown_pct']:.2f}%")
    print(f"    Sharpe:        {buy_hold['sharpe_ratio']:.2f}")

    print(f"\n{'='*80}")
    print(f"{BOLD} COMPARISON TABLE{RESET}")
    print(f"{'='*80}")

    m_a = results["A_Q4KM"]["metrics"]
    challenger_key = [k for k in results if k != "A_Q4KM"][0]
    m_b = results[challenger_key]["metrics"]
    challenger_label = LABELS.get(MODELS[challenger_key], challenger_key)
    bh_ret = buy_hold["total_return_pct"]

    alpha_a = m_a["total_return_pct"] - bh_ret
    alpha_b = m_b["total_return_pct"] - bh_ret

    print(f"\n| Metric | A (Q4_K_M) | {challenger_label} | Buy&Hold |")
    print("|--------|-----------|---------|----------|")
    print(f"| Return | {m_a['total_return_pct']:+.2f}% | {m_b['total_return_pct']:+.2f}% | {bh_ret:+.2f}% |")
    print(f"| Alpha  | {alpha_a:+.2f}% | {alpha_b:+.2f}% | 0.00% |")
    print(f"| MaxDD  | {m_a['max_drawdown_pct']:.2f}% | {m_b['max_drawdown_pct']:.2f}% | {buy_hold['max_drawdown_pct']:.2f}% |")
    print(f"| Sharpe | {m_a['sharpe_ratio']:.2f} | {m_b['sharpe_ratio']:.2f} | {buy_hold['sharpe_ratio']:.2f} |")
    print(f"| WinRate| {m_a['win_rate_pct']:.1f}% | {m_b['win_rate_pct']:.1f}% | - |")
    print(f"| Trades | {m_a['n_trades']} | {m_b['n_trades']} | 1 |")

    sig_a = results["A_Q4KM"]["signal_distribution"]
    sig_b = results[challenger_key]["signal_distribution"]
    print(f"| Signals| {sig_a} | {sig_b} | - |")
    print(f"| AvgConf| {results['A_Q4KM']['avg_confidence']:.2f} | {results[challenger_key]['avg_confidence']:.2f} | - |")

    print(f"\n{BOLD}QUALITY VERDICT:{RESET}")
    score_a = 0
    score_b = 0
    reasons = []

    if m_a["total_return_pct"] > m_b["total_return_pct"]:
        reasons.append(f"Return: A ({m_a['total_return_pct']:+.2f}%) > {challenger_label} ({m_b['total_return_pct']:+.2f}%)")
        score_a += 3
    elif m_b["total_return_pct"] > m_a["total_return_pct"]:
        reasons.append(f"Return: {challenger_label} ({m_b['total_return_pct']:+.2f}%) > A ({m_a['total_return_pct']:+.2f}%)")
        score_b += 3
    else:
        reasons.append(f"Return: tie ({m_a['total_return_pct']:+.2f}%)")

    if alpha_a > alpha_b:
        reasons.append(f"Alpha vs B&H: A ({alpha_a:+.2f}%) > {challenger_label} ({alpha_b:+.2f}%)")
        score_a += 2
    elif alpha_b > alpha_a:
        reasons.append(f"Alpha vs B&H: {challenger_label} ({alpha_b:+.2f}%) > A ({alpha_a:+.2f}%)")
        score_b += 2

    if m_a["sharpe_ratio"] > m_b["sharpe_ratio"]:
        reasons.append(f"Sharpe: A ({m_a['sharpe_ratio']:.2f}) > {challenger_label} ({m_b['sharpe_ratio']:.2f})")
        score_a += 2
    elif m_b["sharpe_ratio"] > m_a["sharpe_ratio"]:
        reasons.append(f"Sharpe: {challenger_label} ({m_b['sharpe_ratio']:.2f}) > A ({m_a['sharpe_ratio']:.2f})")
        score_b += 2

    if m_a["max_drawdown_pct"] < m_b["max_drawdown_pct"]:
        reasons.append(f"Risk: A DD ({m_a['max_drawdown_pct']:.2f}%) < {challenger_label} DD ({m_b['max_drawdown_pct']:.2f}%)")
        score_a += 1
    elif m_b["max_drawdown_pct"] < m_a["max_drawdown_pct"]:
        reasons.append(f"Risk: {challenger_label} DD ({m_b['max_drawdown_pct']:.2f}%) < A DD ({m_a['max_drawdown_pct']:.2f}%)")
        score_b += 1

    if m_a["win_rate_pct"] > m_b["win_rate_pct"]:
        reasons.append(f"Win rate: A ({m_a['win_rate_pct']:.1f}%) > {challenger_label} ({m_b['win_rate_pct']:.1f}%)")
        score_a += 1
    elif m_b["win_rate_pct"] > m_a["win_rate_pct"]:
        reasons.append(f"Win rate: {challenger_label} ({m_b['win_rate_pct']:.1f}%) > A ({m_a['win_rate_pct']:.1f}%)")
        score_b += 1

    for r in reasons:
        print(f"  - {r}")

    if score_b > score_a:
        print(f"\n  {GREEN}>>> WINNER: {MODELS[challenger_key]} ({challenger_label}) — better quality [{challenger_key}={score_b} A={score_a}]{RESET}")
    elif score_a > score_b:
        print(f"\n  {YELLOW}>>> WINNER: hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M (A) — better quality [A={score_a} {challenger_key}={score_b}]{RESET}")
    else:
        print(f"\n  {CYAN}>>> TIE — both models produce equivalent quality [A={score_a} {challenger_key}={score_b}]{RESET}")

    out_dir = Path("logs_prod")
    out_dir.mkdir(exist_ok=True)
    report = {
        "ticker": args.ticker,
        "period_months": args.months,
        "rebalance_freq_days": args.freq,
        "n_decisions": results["A_Q4KM"]["n_decisions"],
        "models": {},
        "buy_hold": {k: v for k, v in buy_hold.items() if k not in ("equity_curve", "trades")},
    }
    for mk, r in results.items():
        report["models"][mk] = {
            "return": r["metrics"]["total_return_pct"],
            "alpha": r["metrics"]["total_return_pct"] - bh_ret,
            "sharpe": r["metrics"]["sharpe_ratio"],
            "max_dd": r["metrics"]["max_drawdown_pct"],
            "win_rate": r["metrics"]["win_rate_pct"],
            "signals": r["signal_distribution"],
            "avg_confidence": r["avg_confidence"],
            "trades": r["metrics"]["trades"],
        }
    report_path = out_dir / "backtest_llm_quality.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

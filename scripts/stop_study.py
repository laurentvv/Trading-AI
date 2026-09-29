"""Étude des stops : que coûte / rapporte un stop large sur une position moyen-long terme ?

    .venv\\Scripts\\python.exe scripts/stop_study.py > docs/STOP_STUDY_2026-09-29.txt

Base = buy & hold ou tendance MA200 (hystérésis 2 %) ; on superpose UN stop (à partir du prix d'entrée
ou du sommet), avec ré-entrée après un délai. Même moteur, mêmes coûts que `src.backtest.report`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import data as bt_data  # noqa: E402
from src.backtest import strategies as st  # noqa: E402
from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402

NEVER = 9.0  # seuil jamais atteint
WARMUP = 200
COOLDOWN = 21


def _variants(prices, window):
    """Cibles sur la fenêtre. Les bases à moyenne mobile sont calculées sur l'historique complet puis
    tronquées ; les surcouches de stop démarrent à l'ouverture de la fenêtre (état d'entrée/sommet neuf),
    pour ne pas dépendre de l'historique antérieur à la période testée."""
    idx = window.index
    always = st.buy_and_hold(window)
    ma = st.ma_trend(prices, 200, band=0.02).loc[idx]
    out = {"Buy & hold (sans stop)": always, "MA200 hystérésis (sans stop)": ma}
    for x in (0.10, 0.15, 0.20, 0.25, 0.30):
        out[f"B&H + stop fixe -{x:.0%} depuis l'entrée"] = st.apply_current_exit_rules(
            window, always, take_profit=NEVER, trail=NEVER, time_stop_days=10**6, hard_stop=x, cooldown_days=COOLDOWN
        )
    for x in (0.10, 0.15, 0.20, 0.25):
        out[f"B&H + trailing -{x:.0%} depuis le sommet"] = st.apply_current_exit_rules(
            window, always, take_profit=NEVER, trail=x, trail_min_profit=-1.0, time_stop_days=10**6,
            hard_stop=NEVER, cooldown_days=COOLDOWN
        )
    out["MA200 hystérésis + stop fixe -20 %"] = st.apply_current_exit_rules(
        window, ma, take_profit=NEVER, trail=NEVER, time_stop_days=10**6, hard_stop=0.20, cooldown_days=COOLDOWN
    )
    return out


def main() -> None:
    for ticker in ("SXRV.DE", "CL=F"):
        prices = bt_data.live_segment(bt_data.load_prices(ticker))
        window = prices.iloc[WARMUP:]
        print(f"\n## {ticker} {window.index[0].date()} -> {window.index[-1].date()}")
        variants = _variants(prices, window)
        for bps in (10.0, 25.0):
            print(f"\ncoût {bps:g} pb par côté (brut avant impôt)\n")
            print("| Variante | CAGR | Sharpe | Drawdown max | Trades | Temps investi |")
            print("|---|---|---|---|---|---|")
            for name, target in variants.items():
                m = compute_metrics(run_backtest(window, target, cost_bps=bps))
                print(f"| {name} | {m['cagr']:.1%} | {m['sharpe']:.2f} | {m['max_drawdown']:.1%} | "
                      f"{m['n_trades']} | {m['time_invested']:.0%} |")


if __name__ == "__main__":
    main()

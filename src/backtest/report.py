"""Rapport des références : buy & hold, 50 % fixe, tendance MA, momentum, règles de sortie actuelles.

    .venv\\Scripts\\python.exe -m src.backtest.report --out docs/BACKTEST_BASELINES_2026-09-29.md

Toutes les stratégies sont évaluées sur la MÊME fenêtre (après `--warmup` séances, le temps de calculer
la MA200), mêmes coûts, mêmes capitaux. Le rapport ne contient que des chiffres générés : la lecture
et les décisions sont écrites à la main dans le document publié.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from . import data as bt_data
from .engine import run_backtest
from .metrics import compute_metrics, daily_returns, deflated_sharpe_ratio, paired_block_bootstrap_sharpe_diff
from . import strategies as st

# ticker -> (libellé, note)
ASSETS = {
    "SXRV.DE": ("SXRV.DE — ETF Nasdaq-100 en EUR (instrument tradé)", ""),
    "CL=F": (
        "CL=F — contrat WTI continu (PROXY du pétrole, non tradable tel quel)",
        "Contrat à terme continu non ajusté du roll : les sauts de roll faussent les rendements. À lire comme "
        "un ordre de grandeur, jamais comme une performance atteignable.",
    ),
}


def build_strategies(prices: pd.DataFrame) -> dict[str, tuple[pd.Series, dict]]:
    """nom -> (exposition cible calculée sur toute la série, options du moteur)."""
    always = st.buy_and_hold(prices)
    ma200 = st.ma_trend(prices, 200)
    return {
        "Buy & hold": (always, {}),
        "50 % fixe (rééquilibré / 21 séances)": (st.fixed_exposure(prices, 0.5), {"rebalance_days": 21}),
        "Tendance MA100": (st.ma_trend(prices, 100), {}),
        "Tendance MA200": (ma200, {}),
        "Tendance MA200 (hystérésis 2 %)": (st.ma_trend(prices, 200, band=0.02), {}),
        "Momentum 12 mois": (st.time_series_momentum(prices, 252), {}),
        "Sorties actuelles, entrée toujours haussière": (st.apply_current_exit_rules(prices, always), {}),
        "Tendance MA200 + sorties actuelles": (st.apply_current_exit_rules(prices, ma200), {}),
    }


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def evaluate_asset(prices: pd.DataFrame, *, warmup: int, capital: float, cost_bps: float, tax_rate: float) -> dict:
    """Exécute toutes les stratégies sur la fenêtre commune ; renvoie résultats bruts, après impôt et rendements."""
    strategies = build_strategies(prices)
    window = prices.iloc[warmup:]
    out: dict = {"window": (window.index[0].date().isoformat(), window.index[-1].date().isoformat(), len(window)),
                 "rows": {}}
    for name, (target, opts) in strategies.items():
        tgt = target.loc[window.index]
        gross = run_backtest(window, tgt, initial_capital=capital, cost_bps=cost_bps, **opts)
        taxed = run_backtest(window, tgt, initial_capital=capital, cost_bps=cost_bps, tax_rate=tax_rate,
                             liquidate_at_end=True, **opts)
        out["rows"][name] = {"gross": gross, "taxed": taxed, "metrics": compute_metrics(gross),
                             "metrics_taxed": compute_metrics(taxed)}
    return out


def render_asset(title: str, note: str, ev: dict, *, cost_bps: float, tax_rate: float) -> str:
    rows = ev["rows"]
    start, end, n = ev["window"]
    lines = [f"## {title}", "", f"Fenêtre commune : {start} → {end} ({n} séances). {note}".strip(), ""]

    lines += ["### Performance brute (coûts inclus, avant impôt)", ""]
    headers = ["Stratégie", "CAGR", "Sharpe", "Sortino", "Drawdown max", "Calmar", "Temps investi", "Turnover/an",
               "Trades", "Gain moy.", "Perte moy."]
    body = []
    for name, r in rows.items():
        m = r["metrics"]
        body.append([name, _pct(m["cagr"]), f"{m['sharpe']:.2f}", f"{m['sortino']:.2f}", _pct(m["max_drawdown"]),
                     f"{m['calmar']:.2f}", _pct(m["time_invested"], 0), f"{m['turnover_per_year']:.1f}×",
                     str(m["n_trades"]), _pct(m["avg_win"]) if m["n_trades"] else "—",
                     _pct(m["avg_loss"]) if m["n_trades"] else "—"])
    lines += [_table(headers, body), ""]

    lines += [f"### Après impôt ({tax_rate:.0%} sur les plus-values réalisées) et position soldée à la fin", ""]
    body = []
    for name, r in rows.items():
        m = r["metrics_taxed"]
        body.append([name, f"{m['final_equity']:,.0f} €".replace(",", " "), _pct(m["cagr"]),
                     f"{m['tax_paid']:,.0f} €".replace(",", " "), _pct(m["max_drawdown"])])
    lines += [_table(["Stratégie", "Capital final", "CAGR net", "Impôt payé", "Drawdown max"], body), ""]

    lines += ["### Rendement par année civile (brut)", ""]
    years = sorted({d.year for r in rows.values() for d in r["gross"].equity.index})
    body = []
    for name, r in rows.items():
        eq = r["gross"].equity
        cells = []
        for y in years:
            seg = eq[eq.index.year == y]
            prev = eq[eq.index.year < y]
            base = prev.iloc[-1] if len(prev) else seg.iloc[0]
            cells.append(_pct(seg.iloc[-1] / base - 1.0))
        body.append([name] + cells)
    lines += [_table(["Stratégie"] + [str(y) for y in years], body), ""]

    lines += ["### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)", ""]
    trial_sharpes = [r["metrics"]["sharpe"] for r in rows.values()]
    bh = daily_returns(rows["Buy & hold"]["gross"].equity)
    ma = daily_returns(rows["Tendance MA200"]["gross"].equity)
    body = []
    for name, r in rows.items():
        ret = daily_returns(r["gross"].equity)
        cells = [name]
        for ref_name, ref in (("B&H", bh), ("MA200", ma)):
            if name in ("Buy & hold",) and ref_name == "B&H" or name == "Tendance MA200" and ref_name == "MA200":
                cells.append("—")
                continue
            b = paired_block_bootstrap_sharpe_diff(ret, ref, n_boot=1000, seed=0)
            cells.append(f"{b['diff']:+.2f} [{b['ci_low']:+.2f} ; {b['ci_high']:+.2f}]")
        dsr = deflated_sharpe_ratio(ret, trial_sharpes)
        cells.append("—" if dsr != dsr else f"{dsr:.2f}")
        body.append(cells)
    lines += [_table(["Stratégie", "ΔSharpe vs B&H", "ΔSharpe vs MA200", "Sharpe déflaté (P>0)"], body), ""]
    lines += [f"Coûts : {cost_bps:g} points de base par côté ; exécution à l'ouverture de la séance suivante. "
              f"Le Sharpe déflaté ne compte que les {len(rows)} variantes de ce tableau : il SURESTIME la "
              "confiance puisque le projet en a essayé bien davantage.", ""]
    return "\n".join(lines)


COST_LEVELS_BPS = (0.0, 10.0, 25.0, 35.0)
COST_SENSITIVITY_STRATEGIES = ("Buy & hold", "Tendance MA200 (hystérésis 2 %)", "Sorties actuelles, entrée toujours haussière")


def render_cost_sensitivity(prices: pd.DataFrame, *, warmup: int, capital: float) -> str:
    """CAGR et Sharpe bruts pour plusieurs niveaux de coût : sépare l'effet des règles de celui du churn."""
    strategies = build_strategies(prices)
    window = prices.iloc[warmup:]
    body = []
    for name in COST_SENSITIVITY_STRATEGIES:
        target, opts = strategies[name]
        cells = [name]
        for bps in COST_LEVELS_BPS:
            m = compute_metrics(run_backtest(window, target.loc[window.index], initial_capital=capital, cost_bps=bps, **opts))
            cells.append(f"{_pct(m['cagr'])} / {m['sharpe']:.2f}")
        body.append(cells)
    headers = ["CAGR / Sharpe"] + [f"{b:g} pb" for b in COST_LEVELS_BPS]
    return "\n".join(["### Sensibilité aux coûts (par côté, brut avant impôt)", "", _table(headers, body), ""])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rapport des stratégies de référence (phase 1)")
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--out", default=f"docs/BACKTEST_BASELINES_{date.today().isoformat()}.md")
    ap.add_argument("--capital", type=float, default=30_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--tax", type=float, default=0.30, help="taux sur les plus-values réalisées (à vérifier)")
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args(argv)

    parts = [f"# Références de performance — généré le {date.today().isoformat()}", "",
             "Sortie de `python -m src.backtest.report` (données : `data_cache/`). Chiffres bruts sans interprétation.", ""]

    parts += ["## Qualité des séries", ""]
    body = []
    for ticker in ("SXRV.DE", "CRUDP.PA", "^NDX", "CL=F"):
        cov = bt_data.coverage(bt_data.load_prices(ticker, args.data_dir))
        body.append([ticker, str(cov["rows"]), f"{cov['start']} → {cov['end']}", _pct(cov["frozen_share"], 0),
                     str(cov["live_rows"]), cov["live_start"] or "—"])
    parts += [_table(["Série", "Lignes", "Période", "Lignes gelées", "Lignes vivantes finales", "Vivante depuis"], body), ""]
    parts += ["Une ligne « gelée » = volume nul et clôture recopiée de la veille (flux Yahoo factice). Une série "
              "majoritairement gelée ne peut pas être backtestée.", ""]

    for ticker, (title, note) in ASSETS.items():
        prices = bt_data.live_segment(bt_data.load_prices(ticker, args.data_dir))
        ev = evaluate_asset(prices, warmup=args.warmup, capital=args.capital, cost_bps=args.cost_bps, tax_rate=args.tax)
        parts.append(render_asset(title, note, ev, cost_bps=args.cost_bps, tax_rate=args.tax))
        parts.append(render_cost_sensitivity(prices, warmup=args.warmup, capital=args.capital))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Rapport écrit : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

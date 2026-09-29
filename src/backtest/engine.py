"""Moteur de backtest quotidien : exposition cible -> équité, avec coûts et impôt.

Convention (aucun regard vers le futur) : la cible du jour t est calculée avec les données <= clôture t
et exécutée à l'OUVERTURE de t+1 (décision après la clôture, exécution le lendemain). La cible de la
dernière ligne n'est donc jamais exécutée. L'équité est valorisée à la clôture.

Coûts : `cost_bps` par côté appliqué au prix d'exécution (demi-spread + glissement ; 0,2 à 0,35 % observés
en démo T212). Impôt : `tax_rate` sur les plus-values réalisées de l'année civile (moins-values de l'année
compensées, aucun report d'une année sur l'autre), prélevé à la vente ; ``liquidate_at_end`` solde la
position à la dernière clôture pour comparer honnêtement une stratégie active au buy & hold, dont
l'impôt est sinon différé indéfiniment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame
    params: dict = field(default_factory=dict)


def run_backtest(
    prices: pd.DataFrame,
    target: pd.Series,
    *,
    initial_capital: float = 30_000.0,
    cost_bps: float = 25.0,
    tax_rate: float = 0.0,
    rebalance_days: int | None = None,
    liquidate_at_end: bool = False,
) -> BacktestResult:
    """Rejoue `target` (0..1) sur `prices` (colonnes Open/Close). `rebalance_days` force un retour à la
    cible toutes les N séances (utile pour une exposition fixe) ; sinon on ne trade qu'au changement de cible."""
    if not {"Open", "Close"} <= set(prices.columns):
        raise ValueError("prices doit contenir les colonnes Open et Close")
    n = len(prices)
    if n < 2:
        raise ValueError("au moins deux séances sont nécessaires")
    tgt = target.reindex(prices.index).fillna(0.0).clip(0.0, 1.0).to_numpy()
    opens = prices["Open"].to_numpy(dtype=float)
    closes = prices["Close"].to_numpy(dtype=float)
    dates = prices.index
    c = cost_bps / 1e4

    cash, shares, avg_cost = float(initial_capital), 0.0, 0.0
    year, ytd_realized, ytd_tax_paid = dates[0].year, 0.0, 0.0
    last_target: float | None = None
    since_rebalance = 0
    trades: list[dict] = []
    equity = [float(initial_capital)]
    exposure = [0.0]

    def _sell(i: int, qty: float, price: float) -> None:
        nonlocal cash, shares, avg_cost, ytd_realized, ytd_tax_paid, year
        if dates[i].year != year:
            year, ytd_realized, ytd_tax_paid = dates[i].year, 0.0, 0.0
        fill = price * (1.0 - c)
        realized = qty * (fill - avg_cost)
        ytd_realized += realized
        tax_due = tax_rate * max(0.0, ytd_realized)
        tax = tax_due - ytd_tax_paid
        ytd_tax_paid = tax_due
        cash += qty * fill - tax
        shares -= qty
        trades.append(
            {"date": dates[i], "side": "SELL", "shares": qty, "price": fill, "notional": qty * fill,
             "realized_pnl": realized, "return_pct": fill / avg_cost - 1.0 if avg_cost > 0 else 0.0, "tax": tax}
        )
        if shares < 1e-12:
            shares, avg_cost = 0.0, 0.0

    for i in range(1, n):
        want = tgt[i - 1]
        since_rebalance += 1
        forced = rebalance_days is not None and since_rebalance >= rebalance_days
        if last_target is None or want != last_target or forced:
            equity_open = cash + shares * opens[i]
            delta_value = want * equity_open - shares * opens[i]
            if delta_value > 1e-9:
                buy_px = opens[i] * (1.0 + c)
                qty = min(delta_value, cash) / buy_px
                if qty > 0:
                    avg_cost = (avg_cost * shares + qty * buy_px) / (shares + qty)
                    shares += qty
                    cash -= qty * buy_px
                    trades.append(
                        {"date": dates[i], "side": "BUY", "shares": qty, "price": buy_px, "notional": qty * buy_px,
                         "realized_pnl": 0.0, "return_pct": 0.0, "tax": 0.0}
                    )
            elif delta_value < -1e-9 and shares > 0:
                _sell(i, min(shares, -delta_value / opens[i]), opens[i])
            last_target, since_rebalance = want, 0
        eq = cash + shares * closes[i]
        equity.append(eq)
        exposure.append(shares * closes[i] / eq if eq > 0 else 0.0)

    if liquidate_at_end and shares > 0:
        _sell(n - 1, shares, closes[n - 1])
        equity[-1] = cash
        exposure[-1] = 0.0

    trade_cols = ["date", "side", "shares", "price", "notional", "realized_pnl", "return_pct", "tax"]
    return BacktestResult(
        equity=pd.Series(equity, index=dates, name="equity"),
        exposure=pd.Series(exposure, index=dates, name="exposure"),
        trades=pd.DataFrame(trades, columns=trade_cols),
        params={"initial_capital": initial_capital, "cost_bps": cost_bps, "tax_rate": tax_rate,
                "rebalance_days": rebalance_days, "liquidate_at_end": liquidate_at_end},
    )

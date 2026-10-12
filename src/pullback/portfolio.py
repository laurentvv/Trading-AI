"""Moteur de portefeuille multi-actions, événementiel journalier (plan §2, contrat 14-15).

Convention : les ordres du jour t+1 sont décidés à la clôture de t (aucun regard vers le futur).

Exécution sur barres OHLC, toujours dans le sens défavorable quand l'ordre intrajournalier est inconnu :
- entrée : ordre d'achat stop au `trigger` (+ 1 cent) ; rempli à max(ouverture, trigger) si High >= trigger ;
- stop : gap sous le stop -> sortie à l'ouverture (la perte dépasse 1R) ; sinon sortie au stop si Low <= stop ;
- si l'entrée et le stop tombent le même jour, on suppose entrée PUIS stop ;
- stop et cible le même jour : le stop passe en premier ;
- cibles 3R / 5R : au niveau, ou à l'ouverture si elle les dépasse ; EMA 9 : vente à l'ouverture suivante.

Monnaie : liquidités en EUR, titres en USD convertis au change du jour (`fx` = USD par EUR). Frais : `cost_bps`
par sens sur le prix (change 0,15 % + spread/glissement). Impôt : `tax_rate` sur les plus-values nettes de
l'année civile (EUR), prélevé à la vente comme dans `src.backtest.engine`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.pullback import rules

TICK = 0.01


@dataclass
class PortfolioResult:
    equity: pd.Series  # EUR, valorisé à la clôture
    exposure: pd.Series
    trades: pd.DataFrame
    fees_eur: float
    tax_eur: float


def run_portfolio(
    dates: pd.DatetimeIndex,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
    ema9: np.ndarray,
    fx: np.ndarray,
    orders: dict[int, list[tuple[int, float, float, float]]],
    symbols: list[str],
    *,
    start: int,
    end: int,
    capital: float = 30_000.0,
    cost_bps: float = 25.0,
    tax_rate: float = 0.30,
    ema_mode: str = "always",
    p: rules.Params = rules.Params(),
    liquidate_at_end: bool = True,
    intraday: str = "worst",
) -> PortfolioResult:
    """Rejoue `orders` {t_exécution: [(j, trigger, stop, priorité)]} de la séance `start` à `end`.

    `ema_mode` : "always" (sortie sous l'EMA 9 sur toute la position restante) ou "after_3r" (seulement
    après la première cible).

    `intraday` : ordre supposé des événements quand une même barre touche plusieurs niveaux.
    "worst" (défaut, règle du plan) : entrée puis stop, et stop avant cible. "ohlc" : chemin O-L-H-C si la
    barre clôture au-dessus de l'ouverture, sinon O-H-L-C. "best" : si le stop est touché le jour du
    déclenchement l'entrée est annulée, et les cibles passent avant le stop. Seul "worst" sert au verdict ;
    les deux autres bornent l'effet de cette hypothèse.
    """
    if ema_mode not in ("always", "after_3r"):
        raise ValueError(f"ema_mode inconnu : {ema_mode}")
    if intraday not in ("worst", "ohlc", "best"):
        raise ValueError(f"intraday inconnu : {intraday}")
    c = cost_bps / 1e4
    cash = float(capital)
    pos: dict[int, dict] = {}
    last_close: dict[int, float] = {}
    year, ytd_real, ytd_paid = None, 0.0, 0.0
    fees = taxes = 0.0
    trades: list[dict] = []
    eq_hist: list[float] = []
    ex_hist: list[float] = []
    prev_equity = float(capital)

    def sell(j: int, qty: float, px: float, t: int) -> None:
        """Vend `qty` de la position j au prix brut `px` (USD) ; frais, impôt et clôture de trade inclus."""
        nonlocal cash, ytd_real, ytd_paid, fees, taxes
        P = pos[j]
        qty = min(qty, P["qty"])
        net = px * (1.0 - c)
        proceeds_eur = qty * net / fx[t]
        pnl_eur = proceeds_eur - qty * P["cost_eur"]
        ytd_real += pnl_eur
        due = tax_rate * max(0.0, ytd_real)
        tax = due - ytd_paid
        ytd_paid = due
        cash += proceeds_eur - tax
        fees += qty * px * c / fx[t]
        taxes += tax
        P["qty"] -= qty
        P["pnl_usd"] += qty * (net - P["entry_net"])
        if P["qty"] < 1e-9:
            r_usd = P["init_qty"] * (P["entry_gross"] - P["stop"])
            trades.append(
                {
                    "symbol": symbols[j],
                    "entry": dates[P["t_entry"]],
                    "exit": dates[t],
                    "bars": t - P["t_entry"],
                    "r": P["pnl_usd"] / r_usd if r_usd > 0 else 0.0,
                    "stage": P["stage"],
                }
            )
            del pos[j]

    def stop_first(o: float, cl: float) -> bool:
        if intraday == "worst" or math.isnan(cl):
            return True
        return False if intraday == "best" else cl >= o  # ohlc : barre haussière = le bas d'abord

    def manage(j: int, t: int, o: float, h: float, lo: float, cl: float, *, entered: bool) -> None:
        P = pos[j]
        if not entered:
            if P["pending_exit"]:
                sell(j, P["qty"], o, t)
                return
            if o <= P["stop"]:
                sell(j, P["qty"], o, t)
                return

        def hit_targets() -> None:
            for k, (price, _frac) in enumerate(P["targets"]):
                if j in pos and k == P["stage"] and h >= price:
                    sell(j, min(P["init_qty"] * p.target_fraction, P["qty"]), max(o, price) if not entered else price, t)
                    P["stage"] += 1

        if lo <= P["stop"]:
            if stop_first(o, cl):
                sell(j, P["qty"], P["stop"], t)
                return
            hit_targets()
            if j in pos:
                sell(j, pos[j]["qty"], P["stop"], t)
            return
        hit_targets()

    for t in range(start, end + 1):
        if dates[t].year != year:
            year, ytd_real, ytd_paid = dates[t].year, 0.0, 0.0
        # 1) gestion des positions ouvertes sur la barre du jour
        for j in list(pos):
            o, h, lo = float(O[t, j]), float(H[t, j]), float(L[t, j])
            if math.isnan(o):
                continue
            manage(j, t, o, h, lo, float(C[t, j]), entered=False)
        # 2) nouvelles entrées (ordres décidés à la clôture précédente)
        todo = sorted(orders.get(t, []), key=lambda x: -x[3])
        for j, trigger, stop, _prio in todo:
            if j in pos:
                continue
            o, h = float(O[t, j]), float(H[t, j])
            if math.isnan(o) or math.isnan(h):
                continue
            trig = trigger + TICK
            if h < trig:
                continue
            if intraday != "worst" and float(L[t, j]) <= stop and (intraday == "best" or float(C[t, j]) >= o):
                continue  # le stop a cédé avant le déclenchement : setup annulé
            fill = max(o, trig)
            fx_prev = fx[t - 1] if t > 0 else fx[t]
            equity_usd = prev_equity * fx_prev
            open_risk = sum(q["qty"] * max(0.0, last_close.get(k, q["entry_gross"]) - q["stop"]) for k, q in pos.items())
            open_exposure = sum(q["qty"] * last_close.get(k, q["entry_gross"]) for k, q in pos.items())
            qty = rules.position_size(equity_usd, trig, stop, open_risk=open_risk, open_exposure=open_exposure, p=p)
            buy_px = fill * (1.0 + c)
            max_cash_qty = cash * fx[t] / buy_px
            qty = math.floor(min(qty, max_cash_qty) / p.qty_step + 1e-9) * p.qty_step
            if qty <= 0:
                continue
            cost_eur_total = qty * buy_px / fx[t]
            cash -= cost_eur_total
            fees += qty * fill * c / fx[t]
            pos[j] = {
                "qty": qty,
                "init_qty": qty,
                "entry_gross": fill,
                "entry_net": buy_px,
                "cost_eur": buy_px / fx[t],
                "stop": float(stop),
                "targets": rules.exit_targets(fill, float(stop), p),
                "stage": 0,
                "pending_exit": False,
                "t_entry": t,
                "pnl_usd": 0.0,
            }
            manage(j, t, o, h, float(L[t, j]), float(C[t, j]), entered=True)
        # 3) clôture : valorisation, drapeaux de sortie sous l'EMA 9
        value_eur = 0.0
        for j, P in list(pos.items()):
            cl = float(C[t, j])
            if not math.isnan(cl):
                last_close[j] = cl
                e9 = float(ema9[t, j])
                armed = ema_mode == "always" or P["stage"] >= 1
                P["pending_exit"] = bool(armed and cl < e9)
            value_eur += P["qty"] * last_close.get(j, P["entry_gross"]) / fx[t]
        equity = cash + value_eur
        eq_hist.append(equity)
        ex_hist.append(value_eur / equity if equity > 0 else 0.0)
        prev_equity = equity

    if liquidate_at_end and pos:
        t = end
        for j in list(pos):
            sell(j, pos[j]["qty"], last_close.get(j, pos[j]["entry_gross"]), t)
        eq_hist[-1] = cash
        ex_hist[-1] = 0.0

    idx = dates[start : end + 1]
    cols = ["symbol", "entry", "exit", "bars", "r", "stage"]
    return PortfolioResult(
        equity=pd.Series(eq_hist, index=idx, name="equity"),
        exposure=pd.Series(ex_hist, index=idx, name="exposure"),
        trades=pd.DataFrame(trades, columns=cols),
        fees_eur=fees,
        tax_eur=taxes,
    )

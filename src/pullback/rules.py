"""Règles pullback en fonctions pures (plan §1, contrat 6-13).

Chaque fonction prend des barres journalières (Open/High/Low/Close/Volume, index croissant) et une date
`asof` : seules les barres <= asof sont lues (aucun regard vers le futur). Les paramètres par défaut sont
ceux du plan ; les variantes pré-enregistrées s'obtiennent en les surchargeant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HORIZONS = (21, 63, 126)


@dataclass(frozen=True)
class Params:
    min_return: float = 0.30
    min_rank_pct: float = 75.0
    consolidation_max_bars: int = 30
    consolidation_min_bars: int = 3
    atr_ratio_max: float = 0.8
    ema_slope_bars: int = 5
    ema21_close_floor: float = 0.98
    support_tolerance: float = 0.015
    max_stop_distance: float = 0.03
    risk_per_trade: float = 0.005
    max_line_pct: float = 0.25
    max_open_risk: float = 0.03
    max_exposure: float = 1.0
    qty_step: float = 0.01
    target_r: tuple[float, float] = (3.0, 5.0)
    target_fraction: float = 1 / 3


def _upto(bars: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    return bars.loc[:asof]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(bars: pd.DataFrame, n: int) -> pd.Series:
    prev = bars["Close"].shift(1)
    tr = pd.concat([bars["High"] - bars["Low"], (bars["High"] - prev).abs(), (bars["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# --- Étape 1 : screener momentum ---------------------------------------------------------------------------


def trailing_returns(close: pd.Series, asof: pd.Timestamp) -> dict[int, float]:
    """Rendement sur 21/63/126 séances au `asof` ; NaN si l'historique est trop court."""
    c = close.loc[:asof]
    return {h: (c.iloc[-1] / c.iloc[-1 - h] - 1.0) if len(c) > h else math.nan for h in HORIZONS}


def momentum_screen(closes: dict[str, pd.Series], asof: pd.Timestamp, p: Params = Params()) -> dict[str, dict]:
    """Actions fortes : au moins un horizon avec rendement >= min_return ET rang centile >= min_rank_pct.

    Le rang est calculé dans l'ensemble `closes` fourni (l'univers liquide du jour). Retourne
    symbole -> {'returns': {h: r}, 'ranks': {h: pct}, 'horizons': [h retenus]}.
    """
    rets = pd.DataFrame({s: trailing_returns(c, asof) for s, c in closes.items()}).T
    ranks = rets.rank(pct=True, na_option="keep") * 100.0
    out: dict[str, dict] = {}
    for s in rets.index:
        ok = [h for h in HORIZONS if rets.at[s, h] >= p.min_return and ranks.at[s, h] >= p.min_rank_pct]
        if ok:
            out[s] = {"returns": rets.loc[s].to_dict(), "ranks": ranks.loc[s].to_dict(), "horizons": ok}
    return out


# --- Étape 1 (suite) : consolidation -----------------------------------------------------------------------


def pivot_lows(bars: pd.DataFrame, k: int = 2) -> list[int]:
    """Positions des creux pivots (fractal à k barres de chaque côté, donc confirmés k barres plus tard)."""
    low = bars["Low"].to_numpy()
    return [
        i
        for i in range(k, len(low) - k)
        if all(low[i] < low[i - j] and low[i] < low[i + j] for j in range(1, k + 1))
    ]


def pivot_highs(bars: pd.DataFrame, k: int = 2) -> list[int]:
    high = bars["High"].to_numpy()
    return [
        i
        for i in range(k, len(high) - k)
        if all(high[i] > high[i - j] and high[i] > high[i + j] for j in range(1, k + 1))
    ]


@dataclass
class Consolidation:
    ok: bool
    anchor: int | None = None  # position du plus haut de la consolidation (ancre de l'AVWAP)
    reasons: list[str] = field(default_factory=list)


def detect_consolidation(bars: pd.DataFrame, asof: pd.Timestamp, p: Params = Params()) -> Consolidation:
    """Consolidation qui se renforce : creux ascendants, volatilité qui se resserre, EMA 9/21 qui tiennent."""
    b = _upto(bars, asof)
    if len(b) < 60:
        return Consolidation(False, reasons=["historique < 60 séances"])
    close = b["Close"]
    e9, e21, e50 = ema(close, 9), ema(close, 21), ema(close, 50)
    window = b.iloc[-p.consolidation_max_bars :]
    anchor = len(b) - len(window) + int(np.argmax(window["High"].to_numpy()))
    bars_since = len(b) - 1 - anchor
    reasons: list[str] = []
    if bars_since < p.consolidation_min_bars:
        reasons.append("plus haut trop récent (pas de consolidation)")
    lows = [i for i in pivot_lows(b) if i > anchor]
    if len(lows) < 2 or not b["Low"].iloc[lows[-1]] > b["Low"].iloc[lows[-2]]:
        reasons.append("creux pivots non ascendants")
    a5, a20 = atr(b, 5).iloc[-1], atr(b, 20).iloc[-1]
    if not (a20 > 0 and a5 / a20 < p.atr_ratio_max):
        reasons.append("volatilité non resserrée")
    s = p.ema_slope_bars
    if not (e9.iloc[-1] >= e9.iloc[-1 - s] and e21.iloc[-1] >= e21.iloc[-1 - s]):
        reasons.append("EMA 9/21 en baisse")
    if not e21.iloc[-1] > e50.iloc[-1]:
        reasons.append("EMA 21 <= EMA 50")
    if not close.iloc[-1] >= e21.iloc[-1] * p.ema21_close_floor:
        reasons.append("clôture sous EMA 21")
    return Consolidation(not reasons, anchor, reasons)


# --- Étape 2 : zones de support, AVWAP, signal -------------------------------------------------------------


def avwap(bars: pd.DataFrame, anchor: int) -> pd.Series:
    """VWAP ancré : Σ(prix typique × volume) / Σ volume depuis la position `anchor`."""
    b = bars.iloc[anchor:]
    typical = (b["High"] + b["Low"] + b["Close"]) / 3.0
    return (typical * b["Volume"]).cumsum() / b["Volume"].cumsum()


def unfilled_gap_levels(bars: pd.DataFrame, lookback: int = 60) -> list[float]:
    """Plus hauts de la veille des gaps haussiers non comblés (ouverture > plus haut veille, plus bas jamais revenu dessous)."""
    b = bars.iloc[-lookback:]
    levels = []
    for i in range(1, len(b)):
        prev_high = b["High"].iloc[i - 1]
        if b["Open"].iloc[i] > prev_high and b["Low"].iloc[i:].min() > prev_high:
            levels.append(float(prev_high))
    return levels


def support_levels(bars: pd.DataFrame, asof: pd.Timestamp, anchor: int) -> dict[str, float]:
    """Niveaux de support au `asof` : EMA 9/21/50, AVWAP, anciens plus hauts (avant l'ancre), gaps non comblés."""
    b = _upto(bars, asof)
    close = b["Close"]
    levels = {f"ema{n}": float(ema(close, n).iloc[-1]) for n in (9, 21, 50)}
    levels["avwap"] = float(avwap(b, anchor).iloc[-1])
    for n, i in enumerate(i for i in pivot_highs(b.iloc[:anchor]) if i >= anchor - 252):
        levels[f"old_high_{n}"] = float(b["High"].iloc[i])
    for n, lv in enumerate(unfilled_gap_levels(b)):
        levels[f"gap_{n}"] = lv
    return levels


@dataclass
class Setup:
    asof: pd.Timestamp
    trigger: float  # plus haut du jour de setup : ordre d'achat stop
    stop: float  # plus bas du jour de setup
    distance: float  # (trigger - stop) / trigger
    confluence: int  # nombre de niveaux dans la zone
    levels: list[str]


def find_setup(bars: pd.DataFrame, asof: pd.Timestamp, p: Params = Params()) -> Setup | None:
    """Jour de setup : consolidation valide, plus bas à <= tolérance d'un niveau, clôture repassée au-dessus,
    distance trigger-stop <= max_stop_distance."""
    cons = detect_consolidation(bars, asof, p)
    if not cons.ok or cons.anchor is None:
        return None
    b = _upto(bars, asof)
    day = b.iloc[-1]
    hits = [
        name
        for name, lv in support_levels(bars, asof, cons.anchor).items()
        if lv > 0 and abs(day["Low"] - lv) / lv <= p.support_tolerance and day["Close"] > lv
    ]
    if not hits:
        return None
    trigger, stop = float(day["High"]), float(day["Low"])
    distance = (trigger - stop) / trigger
    if not 0 < distance <= p.max_stop_distance:
        return None
    return Setup(asof, trigger, stop, distance, len(hits), hits)


# --- Étape 3 : taille ---------------------------------------------------------------------------------------


def position_size(
    equity: float,
    entry: float,
    stop: float,
    *,
    open_risk: float = 0.0,
    open_exposure: float = 0.0,
    p: Params = Params(),
) -> float:
    """Quantité = risque autorisé / risque par action, plafonnée par ligne (25 % de l'équité), par exposition
    totale (<= 100 %) et par risque ouvert total (<= 3 % de l'équité). Arrondie à l'inférieur au pas de quantité.

    `open_risk` et `open_exposure` sont en montant (devise du compte), pas en pourcentage.
    """
    per_share = entry - stop
    if equity <= 0 or entry <= 0 or per_share <= 0:
        return 0.0
    qty = min(
        equity * p.risk_per_trade / per_share,
        equity * p.max_line_pct / entry,
        max(0.0, equity * p.max_exposure - open_exposure) / entry,
        max(0.0, equity * p.max_open_risk - open_risk) / per_share,
    )
    return math.floor(qty / p.qty_step + 1e-9) * p.qty_step if qty > 0 else 0.0


# --- Étape 4 : sorties --------------------------------------------------------------------------------------


def exit_targets(entry: float, stop: float, p: Params = Params()) -> list[tuple[float, float]]:
    """[(prix, fraction de la position initiale)] à 3R et 5R ; R = entrée - stop initial."""
    r = entry - stop
    return [(entry + m * r, p.target_fraction) for m in p.target_r]


def below_ema9_exit(bars: pd.DataFrame, asof: pd.Timestamp) -> bool:
    """Le reste est vendu à l'ouverture suivante si la clôture du `asof` est sous l'EMA 9."""
    close = _upto(bars, asof)["Close"]
    return bool(close.iloc[-1] < ema(close, 9).iloc[-1])

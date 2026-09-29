"""Stratégies de référence : elles produisent une exposition cible 0..1 à partir des seules données <= t.

Un signal calculé sur la fenêtre [.., t] n'utilise jamais t+1 (le moteur exécute à l'ouverture de t+1).
"""

from __future__ import annotations

import pandas as pd


def buy_and_hold(prices: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=prices.index)


def fixed_exposure(prices: pd.DataFrame, weight: float = 0.5) -> pd.Series:
    return pd.Series(float(weight), index=prices.index)


def ma_trend(prices: pd.DataFrame, window: int = 200, band: float = 0.0) -> pd.Series:
    """1 quand la clôture est au-dessus de sa moyenne mobile, 0 sinon. `band` ajoute de l'hystérésis :
    entrée au-dessus de MA×(1+band), sortie sous MA×(1-band). Avant `window` séances : hors marché."""
    close = prices["Close"]
    ma = close.rolling(window).mean()
    out = pd.Series(0.0, index=prices.index)
    in_market = False
    for i, (px, m) in enumerate(zip(close.to_numpy(), ma.to_numpy())):
        if m == m:  # pas NaN
            if not in_market and px > m * (1 + band):
                in_market = True
            elif in_market and px < m * (1 - band):
                in_market = False
        out.iloc[i] = 1.0 if in_market else 0.0
    return out


def time_series_momentum(prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """1 si le rendement sur `lookback` séances est positif, sinon 0."""
    ret = prices["Close"].pct_change(lookback)
    return (ret > 0).astype(float)


def apply_current_exit_rules(
    prices: pd.DataFrame,
    base: pd.Series,
    *,
    take_profit: float = 0.08,
    trail: float = 0.03,
    trail_min_profit: float = 0.005,
    time_stop_days: int = 15,
    time_stop_soft_loss: float = 0.05,
    hard_stop: float = 0.10,
    cooldown_days: int = 1,
) -> pd.Series:
    """Superpose les règles de sortie actuelles de `t212_executor` à un signal d'entrée `base`.

    Ordre de priorité du code réel : hard-stop -10 %, take-profit +8 %, trailing -3 % depuis le sommet (si
    profit > 0,5 %), time-stop à 15 jours calendaires (si perte <= 5 %). Évaluées à la clôture (le code réel
    les évalue à chaque cycle de 30 min : approximation quotidienne). Après une sortie, ré-entrée possible
    après `cooldown_days` séances si `base` est toujours haussier. Sert de test de stress : avec `base` =
    toujours investi, on mesure ce que les règles de sortie coûtent à elles seules en tronquant les gains.
    """
    close = prices["Close"].to_numpy(dtype=float)
    opens = prices["Open"].to_numpy(dtype=float)
    dates = prices.index
    b = base.reindex(prices.index).fillna(0.0).to_numpy()
    out = pd.Series(0.0, index=prices.index)
    in_pos = False
    entry_px = peak = 0.0
    entry_date = None
    blocked_until = -1
    for t in range(len(close)):
        if not in_pos:
            if b[t] > 0 and t >= blocked_until:
                in_pos = True
                entry_px = opens[t + 1] if t + 1 < len(close) else close[t]
                entry_date = dates[t + 1] if t + 1 < len(close) else dates[t]
                peak = entry_px
                out.iloc[t] = 1.0
            continue
        peak = max(peak, close[t])
        pnl = close[t] / entry_px - 1.0
        drop = 1.0 - close[t] / peak
        age = (dates[t] - entry_date).days
        exit_now = (
            b[t] <= 0
            or pnl <= -hard_stop
            or pnl >= take_profit
            or (drop >= trail and pnl > trail_min_profit)
            or (age >= time_stop_days and pnl > -time_stop_soft_loss)
        )
        if exit_now:
            in_pos = False
            blocked_until = t + cooldown_days
            out.iloc[t] = 0.0
        else:
            out.iloc[t] = 1.0
    return out

"""Matrices de marché (jours x symboles) et pré-filtre vectorisé des règles pullback (plan §2, phase 3).

Le banc ne peut pas appeler `rules.find_setup` sur chaque couple (symbole, jour) : ~6 ms l'appel, des
dizaines de millions de couples. Le pré-filtre ici reproduit EXACTEMENT les conditions de
`rules.detect_consolidation` (équivalence testée) et le screener momentum ; seuls les candidats survivants
passent ensuite par `rules.find_setup`, qui reste la source de vérité du setup.

Mémoire : matrices float32 (la machine de dev a 8 Go). Les barres de chaque symbole sont rechargées à la
demande, jamais gardées toutes en mémoire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.pullback import rules, store, universe

logger = logging.getLogger(__name__)

CUT_DATE = "2013-06-01"  # début des barres utilisées (échauffement des EMA/ATR avant 2015)
MAX_HL_RATIO = 5.0  # une barre High/Low > 5 est une erreur de données, pas un marché
CALENDAR_SYMBOL = "AAPL"


def clean_for_bench(df: pd.DataFrame) -> pd.DataFrame:
    """Barres cohérentes (O/C dans [L, H], High/Low <= 5), sans ligne gelée (volume 0 et clôture recopiée)."""
    df = store.clean_bars(df)
    if df.empty:
        return df
    hi_ok = df["High"] >= df[["Open", "Close", "Low"]].max(axis=1) * 0.9999
    lo_ok = df["Low"] <= df[["Open", "Close", "High"]].min(axis=1) * 1.0001
    ratio_ok = df["High"] / df["Low"] <= MAX_HL_RATIO
    frozen = (df["Volume"] == 0) & (df["Close"] == df["Close"].shift(1))
    return df[hi_ok & lo_ok & ratio_ok & ~frozen]


def load_symbol(symbol: str, store_dir: Path = store.STORE_DIR, cut: str = CUT_DATE) -> pd.DataFrame | None:
    p = store.STORE_DIR if store_dir is None else store_dir
    f = Path(p) / f"{symbol}.parquet"
    if not f.exists():
        return None
    df = clean_for_bench(pd.read_parquet(f).loc[cut:])
    return df if len(df) >= 60 else None


def consolidation_prefilter(bars: pd.DataFrame, p: rules.Params = rules.Params()) -> tuple[np.ndarray, np.ndarray]:
    """(ok, atr_ratio) par séance : reproduit `rules.detect_consolidation` pour chaque `asof` en une passe."""
    n = len(bars)
    ok = np.zeros(n, dtype=bool)
    atr_ratio = np.full(n, np.nan, dtype=np.float32)
    if n < 60:
        return ok, atr_ratio
    h, lo, c = bars["High"].to_numpy(), bars["Low"].to_numpy(), bars["Close"].to_numpy()
    close = bars["Close"]
    e9, e21, e50 = (rules.ema(close, s).to_numpy() for s in (9, 21, 50))
    a5, a20 = rules.atr(bars, 5).to_numpy(), rules.atr(bars, 20).to_numpy()
    w = p.consolidation_max_bars
    anchor = np.full(n, -1, dtype=np.int64)
    anchor[w - 1 :] = np.arange(n - w + 1) + np.argmax(sliding_window_view(h, w), axis=1)
    # creux pivots (fractal à 2 barres) : confirmés 2 barres plus tard, donc dernier pivot <= t-2
    piv = np.zeros(n, dtype=bool)
    piv[2:-2] = (lo[2:-2] < lo[1:-3]) & (lo[2:-2] < lo[:-4]) & (lo[2:-2] < lo[3:-1]) & (lo[2:-2] < lo[4:])
    idx = np.arange(n)
    last_piv = np.maximum.accumulate(np.where(piv, idx, -1))
    t = idx
    l1 = np.full(n, -1, dtype=np.int64)
    l1[2:] = last_piv[: n - 2]  # dernier pivot <= t-2
    l2 = np.full(n, -1, dtype=np.int64)
    has1 = l1 >= 1
    l2[has1] = last_piv[l1[has1] - 1]
    rising = (l1 > anchor) & (l2 > anchor) & (l2 >= 0)
    low_up = np.zeros(n, dtype=bool)
    sel = rising
    low_up[sel] = lo[l1[sel]] > lo[l2[sel]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = a5 / a20
    atr_ok = (a20 > 0) & (ratio < p.atr_ratio_max)
    s = p.ema_slope_bars
    slope_ok = np.zeros(n, dtype=bool)
    slope_ok[s:] = (e9[s:] >= e9[:-s]) & (e21[s:] >= e21[:-s])
    ok = (
        (t >= 59)
        & (anchor >= 0)
        & ((t - anchor) >= p.consolidation_min_bars)
        & low_up
        & atr_ok
        & slope_ok
        & (e21 > e50)
        & (c >= e21 * p.ema21_close_floor)
    )
    atr_ratio = np.where(np.isfinite(ratio), ratio, np.nan).astype(np.float32)
    return ok, atr_ratio


@dataclass
class Market:
    dates: pd.DatetimeIndex
    symbols: list[str]
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    ema9: np.ndarray
    dv20: np.ndarray  # volume en dollars moyen 20 séances (NaN avant 20 barres)
    hist_ok: np.ndarray  # >= 252 barres d'historique propre
    cons: np.ndarray  # pré-filtre consolidation
    atr_ratio: np.ndarray
    ret: dict[int, np.ndarray]  # rendements 21/63/126 séances (propres au symbole)

    def liquid(self, min_dollar_volume: float = universe.MIN_DOLLAR_VOLUME, min_price: float = universe.MIN_PRICE):
        return self.hist_ok & (self.C >= min_price) & (self.dv20 >= min_dollar_volume)


def build_market(symbols: list[str], store_dir: Path = store.STORE_DIR, end: str | None = None) -> Market:
    cal = load_symbol(CALENDAR_SYMBOL, store_dir)
    if cal is None:
        raise RuntimeError(f"calendrier de référence {CALENDAR_SYMBOL} introuvable")
    dates = cal.index if end is None else cal.loc[:end].index
    T = len(dates)
    names = [s for s in symbols if (Path(store_dir) / f"{s}.parquet").exists()]
    N = len(names)
    f32 = np.float32
    shape = (T, N)
    mats = {k: np.full(shape, np.nan, dtype=f32) for k in ("O", "H", "L", "C", "ema9", "dv20", "atr_ratio")}
    rets = {h: np.full(shape, np.nan, dtype=f32) for h in rules.HORIZONS}
    hist_ok = np.zeros(shape, dtype=bool)
    cons = np.zeros(shape, dtype=bool)
    kept: list[str] = []
    for name in names:
        bars = load_symbol(name, store_dir)
        if bars is None:
            continue
        j = len(kept)
        kept.append(name)
        pos = dates.get_indexer(bars.index)  # -1 si la séance n'est pas dans le calendrier
        m = pos >= 0
        if not m.any():
            continue
        # Tout est calculé sur la série COMPLÈTE du symbole (comme `rules`), puis reporté sur le calendrier.
        close = bars["Close"]
        pm = pos[m]
        mats["O"][pm, j] = bars["Open"].to_numpy(f32)[m]
        mats["H"][pm, j] = bars["High"].to_numpy(f32)[m]
        mats["L"][pm, j] = bars["Low"].to_numpy(f32)[m]
        mats["C"][pm, j] = close.to_numpy(f32)[m]
        mats["ema9"][pm, j] = rules.ema(close, 9).to_numpy(f32)[m]
        mats["dv20"][pm, j] = (close * bars["Volume"]).rolling(20).mean().to_numpy(f32)[m]
        enough = np.arange(len(bars)) >= 251
        hist_ok[pm, j] = enough[m]
        for h in rules.HORIZONS:
            rets[h][pm, j] = (close / close.shift(h) - 1.0).to_numpy(f32)[m]
        ok, ar = consolidation_prefilter(bars)
        cons[pm, j] = ok[m]
        mats["atr_ratio"][pm, j] = ar[m]
    N2 = len(kept)
    sl = slice(0, N2)
    return Market(
        dates=dates,
        symbols=kept,
        O=mats["O"][:, sl],
        H=mats["H"][:, sl],
        L=mats["L"][:, sl],
        C=mats["C"][:, sl],
        ema9=mats["ema9"][:, sl],
        dv20=mats["dv20"][:, sl],
        hist_ok=hist_ok[:, sl],
        cons=cons[:, sl],
        atr_ratio=mats["atr_ratio"][:, sl],
        ret={h: rets[h][:, sl] for h in rules.HORIZONS},
    )


def momentum_masks(mkt: Market, liquid: np.ndarray, p: rules.Params = rules.Params()):
    """(sel, sel80, rank63) : sélection momentum au rang `p.min_rank_pct` et à 80, rang centile à 63 séances.

    Le rang est calculé parmi les titres liquides du jour (comme `rules.momentum_screen`).
    """
    sel = np.zeros(liquid.shape, dtype=bool)
    sel80 = np.zeros(liquid.shape, dtype=bool)
    rank63 = None
    for h in rules.HORIZONS:
        r = np.where(liquid, mkt.ret[h], np.nan)
        rank = pd.DataFrame(r).rank(axis=1, pct=True, na_option="keep").to_numpy(np.float32) * 100.0
        strong = r >= p.min_return
        sel |= strong & (rank >= p.min_rank_pct)
        sel80 |= strong & (rank >= 80.0)
        if h == 63:
            rank63 = rank
    return sel & liquid, sel80 & liquid, rank63


def candidate_setups(
    mkt: Market,
    liquid: np.ndarray,
    *,
    date_from: str | pd.Timestamp,
    date_to: str | pd.Timestamp | None = None,
    store_dir: Path = store.STORE_DIR,
    p_loose: rules.Params = rules.Params(max_stop_distance=0.04),
) -> pd.DataFrame:
    """Setups détectés par `rules.find_setup` (paramètres les plus larges) sur les candidats du pré-filtre.

    Colonnes : t (séance de décision), j, symbol, trigger, stop, distance, confluence, atr_ratio,
    pass80, rank63, priority. Les variantes plus strictes se déduisent par filtrage de ces colonnes.
    """
    sel, sel80, rank63 = momentum_masks(mkt, liquid)
    cand = sel & mkt.cons
    t0 = int(mkt.dates.searchsorted(pd.Timestamp(date_from)))
    t1 = len(mkt.dates) - 1 if date_to is None else int(mkt.dates.searchsorted(pd.Timestamp(date_to), side="right")) - 1
    cand[:t0] = False
    cand[t1 + 1 :] = False
    rows = []
    for j in np.flatnonzero(cand.any(axis=0)):
        bars = load_symbol(mkt.symbols[j], store_dir)
        if bars is None:
            continue
        for t in np.flatnonzero(cand[:, j]):
            d = mkt.dates[t]
            if d not in bars.index:
                continue
            s = rules.find_setup(bars, d, p_loose)
            if s is None:
                continue
            r63 = float(rank63[t, j]) if np.isfinite(rank63[t, j]) else 0.0
            rows.append(
                {
                    "t": int(t),
                    "j": int(j),
                    "symbol": mkt.symbols[j],
                    "trigger": s.trigger,
                    "stop": s.stop,
                    "distance": s.distance,
                    "confluence": s.confluence,
                    "atr_ratio": float(mkt.atr_ratio[t, j]),
                    "pass80": bool(sel80[t, j]),
                    "rank63": r63,
                    "priority": s.confluence + r63 / 1000.0,
                }
            )
    cols = ["t", "j", "symbol", "trigger", "stop", "distance", "confluence", "atr_ratio", "pass80", "rank63", "priority"]
    return pd.DataFrame(rows, columns=cols)

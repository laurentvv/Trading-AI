"""Stockage journalier en masse (Parquet, mise à jour incrémentale) de l'univers (plan §1.1, contrat 4-5).

Un fichier par symbole dans `data_cache/pullback/`. Seules les barres manquantes sont téléchargées.
Aucune donnée n'est fabriquée : un symbole sans données valides est absent du jour. Une série dont la
dernière barre est périmée (> 3 jours ouvrés) est refusée par `load_bars`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STORE_DIR = Path("data_cache") / "pullback"
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
MAX_STALE_BUSINESS_DAYS = 3
INITIAL_PERIOD = "max"
BATCH = 100

Downloader = Callable[[list[str], "str | None", "str | None"], "dict[str, pd.DataFrame]"]


def _path(symbol: str, store_dir: Path) -> Path:
    return store_dir / f"{symbol}.parquet"


def clean_bars(df: pd.DataFrame | None) -> pd.DataFrame:
    """Barres OHLCV valides (prix > 0, High >= Low), index croissant sans doublon ; jamais de comblement."""
    if df is None or df.empty or not set(COLUMNS) <= set(df.columns):
        return pd.DataFrame(columns=COLUMNS)
    out = df[COLUMNS].copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna()
    ok = (out[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (out["High"] >= out["Low"]) & (out["Volume"] >= 0)
    return out[ok]


def is_stale(last_bar: pd.Timestamp, asof: pd.Timestamp, max_days: int = MAX_STALE_BUSINESS_DAYS) -> bool:
    """Périmé si plus de `max_days` jours ouvrés séparent la dernière barre de `asof`."""
    return int(np.busday_count(last_bar.date(), asof.date())) > max_days


def load_bars(symbol: str, store_dir: Path = STORE_DIR, asof: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Série du symbole, ou None si absente, vide ou périmée par rapport à `asof` (défaut : aujourd'hui)."""
    p = _path(symbol, store_dir)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty:
        return None
    asof = (asof or pd.Timestamp.today()).normalize()
    if is_stale(df.index[-1], asof):
        logger.warning("%s refusé : dernière barre %s périmée", symbol, df.index[-1].date())
        return None
    return df


def update_symbols(
    symbols: list[str],
    download: Downloader,
    store_dir: Path = STORE_DIR,
    asof: pd.Timestamp | None = None,
) -> dict[str, str]:
    """Télécharge uniquement les barres manquantes. Retourne symbole -> 'new' | 'updated' | 'current' | 'failed'.

    `download(symbols, start, period)` renvoie {symbole: DataFrame OHLCV}. Les symboles dont la dernière
    barre date de la veille ouvrée de `asof` (ou plus récente) ne sont pas redemandés.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    asof = (asof or pd.Timestamp.today()).normalize()
    status: dict[str, str] = {}
    todo_new: list[str] = []
    todo_inc: dict[str, pd.Timestamp] = {}
    for s in symbols:
        p = _path(s, store_dir)
        existing = pd.read_parquet(p) if p.exists() else None
        if existing is None or existing.empty:
            todo_new.append(s)
        elif int(np.busday_count(existing.index[-1].date(), asof.date())) <= 1:
            status[s] = "current"
        else:
            todo_inc[s] = existing.index[-1]

    def _merge(sym: str, fresh: pd.DataFrame | None, kind: str) -> None:
        fresh = clean_bars(fresh)
        p = _path(sym, store_dir)
        if fresh.empty:  # rien de nouveau : on garde l'existant tel quel, on n'invente rien
            status[sym] = "current" if sym in todo_inc else "failed"
            return
        parts = [pd.read_parquet(p), fresh] if p.exists() else [fresh]
        clean_bars(pd.concat(parts)).to_parquet(p)
        status[sym] = kind

    for i in range(0, len(todo_new), BATCH):
        chunk = todo_new[i : i + BATCH]
        got = download(chunk, None, INITIAL_PERIOD)
        for s in chunk:
            _merge(s, got.get(s), "new")
    if todo_inc:
        start = (min(todo_inc.values()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        syms = list(todo_inc)
        for i in range(0, len(syms), BATCH):
            chunk = syms[i : i + BATCH]
            got = download(chunk, start, None)
            for s in chunk:
                _merge(s, got.get(s), "updated")
    return status


def yfinance_downloader(symbols: list[str], start: str | None, period: str | None) -> dict[str, pd.DataFrame]:
    """Téléchargement groupé yfinance (ajustement dividendes/splits laissé à Yahoo : auto_adjust)."""
    import yfinance as yf

    kwargs = {"start": start} if start else {"period": period or INITIAL_PERIOD}
    raw = yf.download(
        symbols, interval="1d", auto_adjust=True, group_by="ticker", progress=False, threads=True, timeout=30, **kwargs
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for s in symbols:
        try:
            out[s] = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
    return out

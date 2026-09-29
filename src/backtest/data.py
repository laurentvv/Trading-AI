"""Chargement des séries de prix du cache et détection des séries « gelées ».

Certains flux Yahoo renvoient des tronçons factices (Volume = 0, Close recopié de la ligne précédente) :
CRUDP.PA est gelé à ~82 % sur 2021-2026 (cf. AGENTS.md, « Feed-frozen rows »). Un backtest sur ces
lignes mesurerait du bruit : on ne garde que le tronçon vivant final et on signale la couverture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICE_FILES = {
    "SXRV.DE": "SXRV_DE_max_with_vix.parquet",
    "CRUDP.PA": "CRUDP_PA_max_with_vix.parquet",
    "^NDX": "^NDX_max_with_vix.parquet",
    "CL=F": "CL=F_max_with_vix.parquet",
}
OHLC = ["Open", "High", "Low", "Close"]


def load_prices(ticker: str, data_dir: str | Path = "data_cache") -> pd.DataFrame:
    """Lit le parquet du cache : index date croissant, sans doublon, barres OHLC valides uniquement."""
    path = Path(data_dir) / PRICE_FILES[ticker]
    df = pd.read_parquet(path).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Open", "Close"])
    return df[(df["Open"] > 0) & (df["Close"] > 0)]


def frozen_mask(df: pd.DataFrame) -> pd.Series:
    """Ligne gelée : aucun volume ET clôture identique à la veille (placeholder, pas une vraie séance)."""
    return (df["Volume"] == 0) & (df["Close"] == df["Close"].shift(1))


def live_segment(df: pd.DataFrame, min_run: int = 20) -> pd.DataFrame:
    """Tronçon vivant final : lignes après le dernier tronçon gelé d'au moins `min_run` séances.

    La première ligne vivante est conservée comme point de départ (son rendement vs la valeur périmée
    précédente est, lui, écarté : il vaut ici +36 % pour CRUDP.PA le 2026-01-08).
    """
    frozen = frozen_mask(df).to_numpy()
    end_of_last_long_run = -1
    run = 0
    for i, is_frozen in enumerate(frozen):
        run = run + 1 if is_frozen else 0
        if run >= min_run:
            end_of_last_long_run = i
    if end_of_last_long_run < 0:
        return df
    # Prolonge jusqu'à la fin du tronçon gelé en cours (le dernier run ne s'arrête qu'à la 1re ligne vivante).
    i = end_of_last_long_run
    while i + 1 < len(frozen) and frozen[i + 1]:
        i += 1
    return df.iloc[i + 1 :]


def coverage(df: pd.DataFrame) -> dict:
    """Résumé de qualité d'une série : lignes, part gelée, dates, tronçon vivant exploitable."""
    live = live_segment(df)
    return {
        "rows": len(df),
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "frozen_rows": int(frozen_mask(df).sum()),
        "frozen_share": float(frozen_mask(df).mean()),
        "live_rows": len(live),
        "live_start": live.index[0].date().isoformat() if len(live) else None,
    }

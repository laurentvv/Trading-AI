"""Univers U1 : actions US de Trading 212 -> symboles Yahoo, filtre de liquidité (plan §1.1).

Le symbole Yahoo vient de `shortName` (à jour après un changement de nom, contrairement au ticker T212 :
`BFT_US_EQ` a pour shortName PSFE) avec `.` -> `-` (BRK.B -> BRK-B). Aucune devinette : un nom qui ne
ressemble pas à un symbole US est journalisé et exclu.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

US_SUFFIX = "_US_EQ"
_YAHOO_SYMBOL = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 20_000_000.0
MIN_HISTORY = 252
DOLLAR_VOLUME_WINDOW = 20


def to_yahoo_symbol(short_name: str) -> str | None:
    """`BRK.B` -> `BRK-B` ; None si le nom n'est pas un symbole US plausible."""
    sym = (short_name or "").strip().upper().replace(".", "-")
    return sym if _YAHOO_SYMBOL.fullmatch(sym) else None


def build_us_universe(instruments: list[dict]) -> pd.DataFrame:
    """Actions `STOCK` en USD au suffixe `_US_EQ`. Colonnes : t212_ticker, symbol, isin, name, max_qty.

    En cas de symbole en double (ancien et nouveau listing), le plus récemment ajouté est gardé.
    """
    rows = []
    for ins in instruments:
        ticker = ins.get("ticker", "")
        if ins.get("type") != "STOCK" or not ticker.endswith(US_SUFFIX) or ins.get("currencyCode") != "USD":
            continue
        symbol = to_yahoo_symbol(ins.get("shortName", ""))
        if symbol is None:
            logger.info("Symbole non résolu, exclu : %s (%r)", ticker, ins.get("shortName"))
            continue
        rows.append(
            {
                "t212_ticker": ticker,
                "symbol": symbol,
                "isin": ins.get("isin", ""),
                "name": ins.get("name", ""),
                "max_qty": ins.get("maxOpenQuantity"),
                "added_on": ins.get("addedOn", ""),
            }
        )
    df = pd.DataFrame(rows, columns=["t212_ticker", "symbol", "isin", "name", "max_qty", "added_on"])
    df = df.sort_values("added_on").drop_duplicates("symbol", keep="last")
    return df.drop(columns="added_on").sort_values("symbol").reset_index(drop=True)


def is_liquid(
    bars: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    min_price: float = MIN_PRICE,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
    min_history: int = MIN_HISTORY,
) -> bool:
    """Filtre du jour `asof` : n'utilise que les barres <= asof (aucun regard vers le futur)."""
    hist = bars.loc[:asof]
    if len(hist) < max(min_history, DOLLAR_VOLUME_WINDOW) or hist.index[-1] != asof:
        return False
    last = hist.iloc[-1]
    if not last["Close"] >= min_price:
        return False
    recent = hist.iloc[-DOLLAR_VOLUME_WINDOW:]
    return bool((recent["Close"] * recent["Volume"]).mean() >= min_dollar_volume)

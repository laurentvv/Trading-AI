"""Univers U1 et stockage journalier incrémental (plan pullback, contrat 1-5)."""

import numpy as np
import pandas as pd

from src.pullback import store, universe


def _ins(ticker, short, typ="STOCK", cur="USD", added="2020-01-01", isin="US0000000001"):
    return {
        "ticker": ticker,
        "shortName": short,
        "type": typ,
        "currencyCode": cur,
        "addedOn": added,
        "isin": isin,
        "name": short,
        "maxOpenQuantity": 1000,
    }


def test_universe_keeps_only_us_usd_stocks():
    ins = [
        _ins("AAPL_US_EQ", "AAPL"),
        _ins("SXRVd_EQ", "SXRV", typ="ETF"),
        _ins("ZGYd_EQ", "ZGY", cur="EUR"),
        _ins("SPY_US_EQ", "SPY", typ="ETF"),
        _ins("X_US_EQ", "X", cur="EUR"),
    ]
    assert list(universe.build_us_universe(ins)["symbol"]) == ["AAPL"]


def test_symbol_mapping_is_deterministic():
    assert universe.to_yahoo_symbol("BRK.B") == "BRK-B"
    assert universe.to_yahoo_symbol("aapl") == "AAPL"
    assert universe.to_yahoo_symbol("MLAC1") is None
    assert universe.to_yahoo_symbol("") is None
    df = universe.build_us_universe([_ins("BRK_B_US_EQ", "BRK.B"), _ins("MLAC1_US_EQ", "MLAC1")])
    assert list(df["symbol"]) == ["BRK-B"]


def test_duplicate_symbol_keeps_latest_listing():
    df = universe.build_us_universe(
        [_ins("HSC_US_EQ", "HSC", added="2020-10-06"), _ins("NVRI_US_EQ", "HSC", added="2026-06-03")]
    )
    assert list(df["t212_ticker"]) == ["NVRI_US_EQ"]


def _bars(n=300, price=50.0, volume=1_000_000.0, end="2026-10-09"):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame(
        {"Open": price, "High": price * 1.01, "Low": price * 0.99, "Close": price, "Volume": volume}, index=idx
    )


def test_liquidity_filter_thresholds():
    b = _bars()
    asof = b.index[-1]
    assert universe.is_liquid(b, asof)
    assert not universe.is_liquid(_bars(price=4.0, volume=10_000_000), asof)  # cours < 5 $
    assert not universe.is_liquid(_bars(volume=100_000), asof)  # 5 M$ < 20 M$
    assert not universe.is_liquid(_bars(n=200), asof)  # historique < 252


def test_liquidity_filter_ignores_future_bars():
    b = _bars()
    asof = b.index[-30]
    before = universe.is_liquid(b, asof)
    b2 = b.copy()
    b2.iloc[-29:, b2.columns.get_loc("Volume")] = 0.0  # futur dégradé
    b2.iloc[-29:, b2.columns.get_loc("Close")] = 1.0
    assert universe.is_liquid(b2, asof) == before


def test_clean_bars_drops_invalid_and_never_fills():
    idx = pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-07", "2026-10-09"])
    df = pd.DataFrame(
        {
            "Open": [10, 0, 10, 11, np.nan],
            "High": [11, 11, 12, 12, 11],
            "Low": [9, 9, 9, 9, 9],
            "Close": [10, 10, 11, 11, 10],
            "Volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )
    out = store.clean_bars(df)
    assert list(out.index.strftime("%m-%d")) == ["10-05", "10-07"]  # 0 et NaN écartés, doublon -> dernier
    assert out.loc["2026-10-07", "Open"] == 11


def test_staleness_business_days():
    assert not store.is_stale(pd.Timestamp("2026-10-09"), pd.Timestamp("2026-10-12"))  # vendredi -> lundi
    assert store.is_stale(pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-12"))


def test_load_bars_refuses_stale(tmp_path):
    store.update_symbols(
        ["AAA"], lambda s, st, p: {"AAA": _bars(end="2026-10-01")}, tmp_path, asof=pd.Timestamp("2026-10-01")
    )
    assert store.load_bars("AAA", tmp_path, asof=pd.Timestamp("2026-10-02")) is not None
    assert store.load_bars("AAA", tmp_path, asof=pd.Timestamp("2026-10-12")) is None
    assert store.load_bars("ZZZ", tmp_path) is None


def test_update_is_incremental(tmp_path):
    calls = []

    def dl(symbols, start, period):
        calls.append((tuple(symbols), start, period))
        return {s: _bars(end="2026-10-08") if start is None else _bars(n=2, end="2026-10-12") for s in symbols}

    st1 = store.update_symbols(["AAA", "BBB"], dl, tmp_path, asof=pd.Timestamp("2026-10-08"))
    assert st1 == {"AAA": "new", "BBB": "new"} and calls[0][1:] == (None, "max")
    st2 = store.update_symbols(["AAA", "BBB"], dl, tmp_path, asof=pd.Timestamp("2026-10-09"))
    assert st2 == {"AAA": "current", "BBB": "current"} and len(calls) == 1  # rien redemandé
    st3 = store.update_symbols(["AAA"], dl, tmp_path, asof=pd.Timestamp("2026-10-12"))
    assert st3 == {"AAA": "updated"}
    assert calls[1][1] == "2026-10-09"  # seulement les barres manquantes
    assert store.load_bars("AAA", tmp_path, asof=pd.Timestamp("2026-10-12")).index[-1] == pd.Timestamp("2026-10-12")


def test_failed_download_creates_no_synthetic_data(tmp_path):
    st = store.update_symbols(["AAA"], lambda s, a, b: {}, tmp_path, asof=pd.Timestamp("2026-10-12"))
    assert st == {"AAA": "failed"}
    assert store.load_bars("AAA", tmp_path) is None

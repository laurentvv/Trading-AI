"""CLI : construit l'univers U1 depuis T212 (lecture seule) et met à jour les barres journalières.

    python -m src.pullback.cli universe            # écrit data_cache/pullback_universe.csv
    python -m src.pullback.cli update [--limit N]  # télécharge les barres manquantes
    python -m src.pullback.cli stats               # couverture et liquidité du jour
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.pullback import store, universe

UNIVERSE_CSV = Path("data_cache") / "pullback_universe.csv"
logger = logging.getLogger("pullback.cli")


def fetch_t212_instruments() -> list[dict]:
    """GET /equity/metadata/instruments (lecture seule, compte démo ou réel selon T212_ENV)."""
    import requests
    from dotenv import load_dotenv

    load_dotenv(".env.t212")
    from src.t212_executor import _get_t212_base_url, get_auth_header

    resp = requests.get(f"{_get_t212_base_url()}/equity/metadata/instruments", headers=get_auth_header(), timeout=60)
    resp.raise_for_status()
    return resp.json()


def cmd_universe() -> int:
    df = universe.build_us_universe(fetch_t212_instruments())
    UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIVERSE_CSV, index=False)
    print(f"{len(df)} symboles -> {UNIVERSE_CSV}")
    return 0


def cmd_update(limit: int | None) -> int:
    symbols = list(pd.read_csv(UNIVERSE_CSV)["symbol"])[:limit]
    status = store.update_symbols(symbols, store.yfinance_downloader)
    print(pd.Series(status).value_counts().to_string())
    return 0


def cmd_stats() -> int:
    symbols = list(pd.read_csv(UNIVERSE_CSV)["symbol"])
    asof = pd.Timestamp.today().normalize()
    loaded = liquid = 0
    for s in symbols:
        bars = store.load_bars(s, asof=asof)
        if bars is None:
            continue
        loaded += 1
        liquid += universe.is_liquid(bars, bars.index[-1])
    print(f"univers {len(symbols)} | barres valides et fraîches {loaded} | liquides {liquid}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("universe")
    up = sub.add_parser("update")
    up.add_argument("--limit", type=int, default=None)
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    if args.cmd == "universe":
        return cmd_universe()
    if args.cmd == "update":
        return cmd_update(args.limit)
    return cmd_stats()


if __name__ == "__main__":
    sys.exit(main())

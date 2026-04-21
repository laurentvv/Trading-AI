import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EIA_CACHE_DIR = Path("data_cache") / "eia"
EIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EIACacheEntry:
    data: pd.DataFrame
    fetched_at: datetime
    ttl_hours: int


class EIAClient:
    BASE_URL = "https://api.eia.gov/v2"

    TTL_HOURS = {
        "inventories": 6,
        "steo_wti": 24,
        "steo_brent": 24,
        "steo_prod": 24,
        "steo_demand": 24,
        "steo_stocks": 24,
        "steo_imports": 24,
    }

    def __init__(self):
        self.api_key: str = os.getenv("EIA_API_KEY", "")
        self._cache: dict[str, EIACacheEntry] = {}

        if not self.api_key:
            logger.warning("EIA_API_KEY not set — EIA module will return empty data")

    @staticmethod
    def is_oil_ticker(ticker: str) -> bool:
        if not ticker:
            return False
        return any(p in ticker for p in ("CL=F", "CRUDP", "BZ=F", "CL="))

    def get_fundamental_context(self) -> dict:
        if not self.api_key:
            logger.warning("EIA_API_KEY missing — skipping fundamental data fetch")
            return {}

        result: dict = {"as_of": datetime.now().isoformat()}

        inv_df = self.get_crude_inventories(weeks=8)
        if not inv_df.empty:
            result["inventories"] = self._parse_inventory(inv_df)

        # New: Crude Oil Imports (Monthly)
        imports_df = self.get_crude_imports(months=6)
        if not imports_df.empty:
            result["imports"] = self._parse_imports(imports_df)

        # New: Refinery Utilization (Weekly)
        refinery_df = self.get_refinery_utilization(weeks=4)
        if not refinery_df.empty:
            result["refinery"] = self._parse_refinery(refinery_df)

        # New: Brent Spot Price (Dated Brent)
        spot_df = self.get_brent_spot_price(days=30)
        if not spot_df.empty:
            result["brent_spot"] = self._parse_spot_price(spot_df)

        steo_data = {}
        for label, series_id in [
            ("wti_price", "WTIPUUS"),
            ("brent_price", "BREPUUS"),
            ("us_production", "COPRPUS"),
            ("world_demand", "PATC_WORLD"),
            ("us_stocks", "COSXPUS"),
            ("us_net_imports", "CONIPUS"),
        ]:
            df = self._get_steo_series(series_id, periods=3)
            if not df.empty:
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                unit = (
                    str(df.iloc[0].get("unit", ""))
                    if "unit" in df.columns or not df.empty
                    else ""
                )
                steo_data[label] = {
                    "latest_value": float(latest["value"]),
                    "latest_period": str(latest["period"]),
                    "previous_value": float(prev["value"]),
                    "unit": unit,
                }
        if steo_data:
            result["steo"] = steo_data

        return result

    def get_crude_inventories(self, weeks: int = 8) -> pd.DataFrame:
        cache_key = "inventories"
        cached = self._get_from_cache(cache_key, self.TTL_HOURS.get("inventories", 6))
        if cached is not None:
            return cached

        params = {
            "facets[duoarea][]": "NUS",
            "facets[product][]": "EPC0",
            "facets[process][]": "SAX",
            "frequency": "weekly",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": str(weeks),
            "data[]": "value",
        }
        data = self._make_request("/petroleum/stoc/wstk/data", params)
        if data:
            df = pd.DataFrame(data)
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["value"])
                df = df[df["value"] >= 0]
            df["period"] = pd.to_datetime(df["period"])
            df = df.sort_values("period").reset_index(drop=True)
            self._save_to_cache(cache_key, df, self.TTL_HOURS.get("inventories", 6))
            return df
        return self._load_disk_cache_fallback(cache_key)

    def get_crude_imports(self, months: int = 6) -> pd.DataFrame:
        """Fetches US Crude Oil Imports data from EIA v2."""
        cache_key = "crude_imports"
        cached = self._get_from_cache(cache_key, 24)  # 24h TTL for monthly data
        if cached is not None:
            return cached

        params = {
            "frequency": "monthly",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": str(months * 10),  # More rows because multiple origins
            "data[]": "quantity",
        }
        data = self._make_request("/crude-oil-imports/data", params)
        if data:
            df = pd.DataFrame(data)
            if "quantity" in df.columns:
                df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
            df["period"] = pd.to_datetime(df["period"])
            # Aggregate by period to get total US imports
            df_total = df.groupby("period")["quantity"].sum().reset_index()
            df_total = df_total.sort_values("period").reset_index(drop=True)
            self._save_to_cache(cache_key, df_total, 24)
            return df_total
        return self._load_disk_cache_fallback(cache_key)

    def get_refinery_utilization(self, weeks: int = 4) -> pd.DataFrame:
        """Fetches US Refinery Utilization percentage from EIA v2."""
        cache_key = "refinery_util"
        cached = self._get_from_cache(cache_key, self.TTL_HOURS.get("refinery", 6))
        if cached is not None:
            return cached

        params = {
            "facets[process][]": "YUP",
            "frequency": "weekly",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": str(weeks * 5),  # 5 PADDs
            "data[]": "value",
        }
        data = self._make_request("/petroleum/pnp/wiup/data", params)
        if data:
            df = pd.DataFrame(data)
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["value"])
            df["period"] = pd.to_datetime(df["period"])
            # Aggregate by period (mean across PADDs for US average)
            df_avg = df.groupby("period")["value"].mean().reset_index()
            df_avg = df_avg.sort_values("period").reset_index(drop=True)
            self._save_to_cache(cache_key, df_avg, self.TTL_HOURS.get("refinery", 6))
            return df_avg
        return self._load_disk_cache_fallback(cache_key)

    def get_brent_spot_price(self, days: int = 30) -> pd.DataFrame:
        """Fetches Europe Brent Spot Price FOB (Dated Brent) from EIA v2."""
        cache_key = "brent_spot"
        cached = self._get_from_cache(cache_key, 6)  # 6h TTL
        if cached is not None:
            return cached

        params = {
            "facets[series][]": "RBRTE",
            "frequency": "daily",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": str(days),
            "data[]": "value",
        }
        data = self._make_request("/petroleum/pri/spt/data", params)
        if data:
            df = pd.DataFrame(data)
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["period"] = pd.to_datetime(df["period"])
            df = df.sort_values("period").reset_index(drop=True)
            self._save_to_cache(cache_key, df, 6)
            return df
        return self._load_disk_cache_fallback(cache_key)

    def _get_steo_series(self, series_id: str, periods: int = 3) -> pd.DataFrame:
        cache_key = f"steo_{series_id}"
        ttl = self.TTL_HOURS.get(cache_key, 24)
        cached = self._get_from_cache(cache_key, ttl)
        if cached is not None:
            return cached

        params = {
            "facets[seriesId][]": series_id,
            "frequency": "monthly",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": str(periods),
            "data[]": "value",
        }
        data = self._make_request("/steo/data", params)
        if data:
            df = pd.DataFrame(data)
            if "value" in df.columns:
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["value"])
            df["period"] = pd.to_datetime(df["period"])
            df = df.sort_values("period").reset_index(drop=True)
            self._save_to_cache(cache_key, df, ttl)
            return df
        return self._load_disk_cache_fallback(cache_key)

    def format_for_llm(self, data: dict) -> str:
        if not data:
            return "No EIA fundamental data available."

        lines = [f"EIA Fundamental Data (as of {data.get('as_of', 'unknown')}):"]

        inv = data.get("inventories")
        if inv and inv.get("current") is not None:
            current = inv["current"]
            wow = inv.get("wow_change", 0)
            direction = "build (bearish)" if wow > 0 else "draw (bullish)"
            lines.append(
                f"- US Crude Inventories: {current:,.0f} thousand barrels "
                f"({wow:+,.0f} WoW, {direction})"
            )
            lines.append(
                f"  WoW Change: {wow:+,.0f} KB ({inv.get('wow_change_pct', 0):+.2f}%)"
            )
            trend = inv.get("trend_4w")
            if trend:
                lines.append(f"  4-Week Trend: {trend}")

        imp = data.get("imports")
        if imp and imp.get("latest_value") is not None:
            latest = imp["latest_value"]
            mom = imp.get("mom_change", 0)
            direction = (
                "increase (bearish supply)" if mom > 0 else "decrease (bullish supply)"
            )
            lines.append(
                f"- US Crude Imports: {latest:,.0f} thousand barrels "
                f"({mom:+,.0f} MoM, {direction})"
            )

        ref = data.get("refinery")
        if ref and ref.get("current") is not None:
            current = ref["current"]
            wow = ref.get("wow_change", 0)
            direction = (
                "increase (bullish for crude demand)"
                if wow > 0
                else "decrease (bearish for crude demand)"
            )
            lines.append(
                f"- US Refinery Utilization: {current:.1f}% "
                f"({wow:+.1f}% WoW, {direction})"
            )

        steo = data.get("steo", {})
        if steo:
            wti = steo.get("wti_price")
            if wti:
                lines.append(
                    f"- STEO WTI Price Forecast: ${wti['latest_value']:.2f}/bbl "
                    f"({wti['latest_period']})"
                )
            brent = steo.get("brent_price")
            if brent:
                lines.append(
                    f"- STEO Brent Price Forecast: ${brent['latest_value']:.2f}/bbl"
                )
            prod = steo.get("us_production")
            if prod:
                lines.append(
                    f"- STEO US Crude Production Forecast: {prod['latest_value']:.2f} M bbl/day"
                )
            demand = steo.get("world_demand")
            if demand:
                lines.append(
                    f"- STEO World Liquid Fuels Consumption: {demand['latest_value']:.2f} M bbl/day"
                )
            imports = steo.get("us_net_imports")
            if imports:
                lines.append(
                    f"- STEO US Crude Net Imports: {imports['latest_value']:.2f} M bbl/day"
                )

        spot = data.get("brent_spot")
        if spot and spot.get("current") is not None:
            current = spot["current"]
            wow = spot.get("wow_change", 0)
            lines.append(
                f"- Europe Brent Spot Price (Dated Brent): ${current:.2f}/bbl "
                f"({wow:+.2f} WoW)"
            )

        return "\n".join(lines)

    def _parse_inventory(self, df: pd.DataFrame) -> dict:
        if df.empty or "value" not in df.columns:
            return {}
        current = float(df.iloc[-1]["value"])
        previous = float(df.iloc[-2]["value"]) if len(df) > 1 else current
        wow_change = current - previous
        wow_pct = (wow_change / previous * 100) if previous != 0 else 0.0

        trend = None
        if len(df) >= 4:
            changes = df["value"].diff().dropna().tail(4)
            if not changes.empty:
                avg_change = changes.mean()
                trend = f"avg {avg_change:+,.0f} KB/week over 4 weeks"

        return {
            "current": current,
            "previous": previous,
            "wow_change": round(wow_change, 2),
            "wow_change_pct": round(wow_pct, 2),
            "history_periods": len(df),
            "trend_4w": trend,
        }

    def _parse_imports(self, df: pd.DataFrame) -> dict:
        if df.empty or "quantity" not in df.columns:
            return {}
        latest = float(df.iloc[-1]["quantity"])
        previous = float(df.iloc[-2]["quantity"]) if len(df) > 1 else latest
        mom_change = latest - previous
        mom_pct = (mom_change / previous * 100) if previous != 0 else 0.0

        return {
            "latest_value": latest,
            "previous_value": previous,
            "mom_change": round(mom_change, 2),
            "mom_change_pct": round(mom_pct, 2),
            "history_periods": len(df),
        }

    def _parse_refinery(self, df: pd.DataFrame) -> dict:
        if df.empty or "value" not in df.columns:
            return {}
        current = float(df.iloc[-1]["value"])
        previous = float(df.iloc[-2]["value"]) if len(df) > 1 else current
        wow_change = current - previous

        return {
            "current": round(current, 2),
            "previous": round(previous, 2),
            "wow_change": round(wow_change, 2),
            "history_periods": len(df),
        }

    def _parse_spot_price(self, df: pd.DataFrame) -> dict:
        if df.empty or "value" not in df.columns:
            return {}
        current = float(df.iloc[-1]["value"])
        previous = float(df.iloc[-2]["value"]) if len(df) > 1 else current
        wow_change = current - previous

        return {
            "current": round(current, 2),
            "previous": round(previous, 2),
            "wow_change": round(wow_change, 2),
            "history_periods": len(df),
        }

    def _make_request(self, endpoint: str, params: dict) -> list[dict] | None:
        if not self.api_key:
            return None

        url = f"{self.BASE_URL}{endpoint}"
        all_params = {"api_key": self.api_key, **params}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=all_params, timeout=15)
                if resp.status_code == 200:
                    body = resp.json()
                    data_rows = body.get("response", {}).get("data", [])
                    return data_rows
                elif resp.status_code in (401, 403):
                    logger.critical(
                        f"EIA API auth error ({resp.status_code}): key may be invalid"
                    )
                    return None
                elif resp.status_code == 429:
                    logger.warning("EIA API rate limited, falling back to cache")
                    return None
                elif resp.status_code >= 500:
                    logger.warning(
                        f"EIA API server error ({resp.status_code}), attempt {attempt + 1}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** (attempt + 1))
                    continue
                else:
                    logger.error(
                        f"EIA API error ({resp.status_code}): {resp.text[:200]}"
                    )
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"EIA API timeout, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
            except requests.exceptions.ConnectionError:
                logger.warning(f"EIA API connection error, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
            except Exception as e:
                logger.error(f"EIA API unexpected error: {e}")
                return None

        return None

    def _get_from_cache(self, key: str, ttl_hours: int) -> pd.DataFrame | None:
        now = datetime.now()
        entry = self._cache.get(key)
        if entry and (now - entry.fetched_at).total_seconds() < ttl_hours * 3600:
            logger.debug(f"EIA cache hit (memory): {key}")
            return entry.data

        disk_path = EIA_CACHE_DIR / f"eia_{key}.parquet"
        if disk_path.exists():
            try:
                df = pd.read_parquet(disk_path)
                mtime = datetime.fromtimestamp(disk_path.stat().st_mtime)
                if (now - mtime).total_seconds() < ttl_hours * 3600:
                    self._cache[key] = EIACacheEntry(
                        data=df, fetched_at=mtime, ttl_hours=ttl_hours
                    )
                    logger.debug(f"EIA cache hit (disk): {key}")
                    return df
                logger.debug(f"EIA cache expired (disk): {key}")
            except Exception as e:
                logger.warning(f"EIA cache read error for {key}: {e}")
        return None

    def _save_to_cache(self, key: str, data: pd.DataFrame, ttl_hours: int):
        self._cache[key] = EIACacheEntry(
            data=data, fetched_at=datetime.now(), ttl_hours=ttl_hours
        )
        disk_path = EIA_CACHE_DIR / f"eia_{key}.parquet"
        try:
            data.to_parquet(disk_path)
            logger.debug(f"EIA cached: {key}")
        except Exception as e:
            logger.warning(f"EIA cache write error for {key}: {e}")

    def _load_disk_cache_fallback(self, key: str) -> pd.DataFrame:
        disk_path = EIA_CACHE_DIR / f"eia_{key}.parquet"
        if disk_path.exists():
            try:
                df = pd.read_parquet(disk_path)
                logger.warning(f"Using expired EIA disk cache fallback for {key}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load expired EIA disk cache for {key}: {e}")
        return pd.DataFrame()

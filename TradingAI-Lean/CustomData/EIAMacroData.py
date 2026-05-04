"""
EIA Macro Data custom data feed for QuantConnect Lean.

Fetches petroleum stock data from the EIA API v2 and makes it available
as a custom data source within Lean backtests. This replaces the manual
EIA fetch in src/eia_client.py with a Lean-native data feed.
"""

from AlgorithmImports import *


class EIAMacroData(PythonData):
    """Custom data feed for EIA petroleum stock data."""

    def get_source(self, config, date, is_live):
        api_key = config.custom_properties.get("eia_api_key", "")
        if not api_key:
            return SubscriptionDataSource("", SubscriptionTransportMedium.REST_HEADER)

        url = (
            f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
            f"?api_key={api_key}"
            f"&frequency=weekly"
            f"&data[0]=value"
            f"&sort[0][column]=period"
            f"&sort[0][direction]=desc"
            f"&length=52"
        )
        return SubscriptionDataSource(url, SubscriptionTransportMedium.REST_HEADER)

    def reader(self, config, line, date, is_live):
        if not line.strip():
            return None

        data = EIAMacroData()
        data.symbol = config.symbol

        try:
            obj = json.loads(line)
            if "response" in obj and "data" in obj["response"]:
                items = obj["response"]["data"]
                if not items:
                    return None

                for item in items:
                    if "period" in item and "value" in item:
                        period_str = item.get("period", "")
                        try:
                            data.time = datetime.strptime(period_str, "%Y-%m-%d")
                        except ValueError:
                            continue

                        value_str = item.get("value", "")
                        if value_str is None or value_str == "":
                            continue

                        data.value = float(value_str)
                        data["area"] = item.get("area", "")
                        data["area_name"] = item.get("area-name", "")
                        data["product"] = item.get("product", "")
                        data["product_name"] = item.get("product-name", "")
                        data["series"] = item.get("series-description", "")
                        data["unit"] = item.get("units", "")
                        return data

        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return None

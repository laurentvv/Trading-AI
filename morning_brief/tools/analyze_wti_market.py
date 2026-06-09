from smolagents import Tool

from morning_brief.tools.rss_helpers import COMMON_FEEDS


def _compute_smas(close):
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    return sma20, sma50, sma200


def _compute_vwap(hist):
    if "Volume" not in hist.columns:
        return None
    vol = hist["Volume"]
    typical = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    mask = vol > 0
    if not mask.any():
        return None
    return float((typical[mask] * vol[mask]).sum() / vol[mask].sum())


def _compute_bollinger(close):
    if len(close) < 20:
        return None
    sma_bb = close.rolling(20).mean()
    std_bb = close.rolling(20).std()
    return {
        "upper": round(float(sma_bb.iloc[-1] + 2 * std_bb.iloc[-1]), 2),
        "middle": round(float(sma_bb.iloc[-1]), 2),
        "lower": round(float(sma_bb.iloc[-1] - 2 * std_bb.iloc[-1]), 2),
    }


def _format_technical_summary(latest, change_1d, rsi, sma20, sma50, sma200, vwap):
    if all(x is not None for x in [rsi, sma20, sma50, sma200, vwap]):
        return (
            f"WTI ${latest:.2f} ({change_1d:+.1f}%) | "
            f"RSI={rsi:.0f} | SMA20={sma20:.1f} SMA50={sma50:.1f} SMA200={sma200:.1f} | "
            f"VWAP={vwap:.1f}"
        )
    return f"WTI ${latest:.2f} ({change_1d:+.1f}%)"


_CRITICAL_KEYWORDS = [
    "strait of hormuz", "suez canal blocked", "panama canal blocked",
    "oil embargo", "oil crash", "oil plunge", "oil surge", "oil spike",
    "oil slump", "oil drops", "oil falls", "oil tumbles", "oil rallies",
    "oil soars", "oil disruption", "oil price war", "crude crash",
    "crude plunge", "crude surge", "crude spike", "crude drops",
    "crude falls", "crude tumbles", "crude rallies",
    "strategic petroleum reserve release", "supply disruption oil",
    "force majeure oil", "opec production cut", "opec production hike",
    "opec surprise", "saudi aramco attack", "saudi aramco fire",
    "saudi aramco explosion", "iran oil sanction", "iran attack oil",
    "hormuz blocked", "hormuz closure", "hormuz truce",
    "hormuz ceasefire", "hormuz reopen",
]

_CRITICAL_COMBO = [
    ("attack", "pipeline"), ("blaze", "refinery"), ("blaze", "pipeline"),
    ("blaze", "facility"), ("fire", "facility"), ("explosion", "facility"),
    ("outage", "refinery"), ("outage", "pipeline"), ("outage", "facility"),
    ("strike", "facility"), ("explosion", "pipeline"),
    ("pipeline", "rupture"), ("fire", "refinery"), ("explosion", "refinery"),
    ("strike", "refinery"), ("russia", "sanction", "oil"),
    ("iraq", "oil", "attack"), ("houthi", "attack", "ship"),
    ("red sea", "attack", "ship"), ("eia", "surprise"),
    ("api", "surprise"), ("inventories", "drop", "oil"),
    ("inventories", "surge", "oil"), ("demand", "collapse", "oil"),
    ("ban", "oil", "export"), ("oil", "nationaliz"),
    ("opec", "surprise", "cut"), ("opec", "surprise", "hike"),
    ("attack", "refinery"), ("drone", "attack", "oil"),
    ("hurricane", "oil"), ("hormuz", "reopen"),
    ("reserves", "release", "oil"), ("spr", "release"),
]

_OIL_CONTEXT_WORDS = [
    "oil", "crude", "petroleum", "opec", "hormuz", "wti", "brent",
    "barrel", "bpd", "refinery", "pipeline", "tanker", "bbl",
    "strait", "energy", "lng", "lpg", "fuel", "facility",
]


def _is_critical_headline(title):
    tl = title.lower()
    if any(kw in tl for kw in _CRITICAL_KEYWORDS):
        return True
    for combo in _CRITICAL_COMBO:
        if all(kw in tl for kw in combo):
            return True
    return False


def _is_oil_related(title):
    return any(w in title.lower() for w in _OIL_CONTEXT_WORDS)


class AnalyzeWtiMarketTool(Tool):
    name = "analyze_wti_market"
    description = (
        "WTI crude oil analysis: price, technicals (SMA, RSI, Bollinger, VWAP), "
        "EIA inventories, and critical news headlines from RSS keyword filtering. "
        "Returns a compact summary string. Full data saved to output/tools/."
    )
    inputs = {}
    output_type = "string"

    _RSS_FEEDS = COMMON_FEEDS + [
        ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CL=F&region=US&lang=en-US"),
        ("OilPrice.com", "https://oilprice.com/rss/main"),
        ("Google News - Oil Geopolitics", "https://news.google.com/rss/search?q=Iran+oil+OR+ceasefire+OR+%22strait+of+hormuz%22+OR+OPEC+when:1d&hl=en-US&gl=US&ceid=US:en"),
        ("Google News - Crude Oil", "https://news.google.com/rss/search?q=WTI+OR+%22crude+oil%22+OR+%22oil+price%22+when:1d&hl=en-US&gl=US&ceid=US:en"),
        ("Google News - Infrastructure", "https://news.google.com/rss/search?q=%28%22refinery%22+OR+%22pipeline%22+OR+%22oil+facility%22%29+AND+%28%22fire%22+OR+%22explosion%22+OR+%22blaze%22+OR+%22strike%22+OR+%22outage%22+OR+%22attack%22%29+when%3A1d&hl=en-US&gl=US&ceid=US:en"),
    ]

    def _fetch_technicals(self):
        import yfinance as yf
        from morning_brief.tools import save_tool_result

        try:
            ticker = yf.Ticker("CL=F")
            hist = ticker.history(period="1y")

            if hist.empty:
                full = {"price": None, "error": "No WTI data"}
                save_tool_result("wti_market", full)
                return None, "ERROR: No WTI data from yfinance"

            close = hist["Close"]
            latest = float(close.iloc[-1])
            change_1d = 0.0
            if len(close) >= 2:
                change_1d = (latest - float(close.iloc[-2])) / float(close.iloc[-2]) * 100

            sma20, sma50, sma200 = _compute_smas(close)
            vwap = _compute_vwap(hist)
            rsi = self._calc_rsi(close, 14)
            bb = _compute_bollinger(close)

            full = {
                "price": round(latest, 2),
                "change_1d": round(change_1d, 2),
                "sma20": round(sma20, 2) if sma20 else None,
                "sma50": round(sma50, 2) if sma50 else None,
                "sma200": round(sma200, 2) if sma200 else None,
                "vwap": round(vwap, 2) if vwap else None,
                "rsi": round(rsi, 1) if rsi else None,
                "bollinger": bb,
            }

            summary = _format_technical_summary(latest, change_1d, rsi, sma20, sma50, sma200, vwap)
            return full, summary
        except Exception as e:
            full = {"price": None, "error": str(e)}
            save_tool_result("wti_market", full)
            return None, f"ERROR: {e}"

    def _fetch_eia(self):
        import os
        import requests

        try:
            api_key = os.getenv("EIA_API_KEY", "")
            if not api_key:
                return None
            url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data"
            params = {
                "api_key": api_key,
                "facets[duoarea][]": "NUS", "facets[product][]": "EPC0",
                "facets[process][]": "SAX", "frequency": "weekly",
                "sort[0][column]": "period", "sort[0][direction]": "desc",
                "length": "4", "data[]": "value",
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            rows = resp.json().get("response", {}).get("data", [])
            if not rows:
                return None
            cur = float(rows[0]["value"])
            prev = float(rows[1]["value"]) if len(rows) > 1 else cur
            wow = cur - prev
            direction = "build(bearish)" if wow > 0 else "draw(bullish)"
            return {"current_kb": cur, "wow_change": round(wow, 0), "direction": direction, "wow_raw": wow}
        except Exception:
            return None

    def _fetch_headlines(self):
        from morning_brief.tools.rss_helpers import fetch_rss_entries

        try:
            headlines = []
            for source, url in self._RSS_FEEDS:
                for entry in fetch_rss_entries(url):
                    title = entry.get("title", "").strip()
                    if not title:
                        continue
                    is_crit = _is_critical_headline(title)
                    if not is_crit and not _is_oil_related(title):
                        continue
                    headlines.append({"title": title, "source": source, "critical": is_crit})
            return headlines
        except Exception:
            return []

    def forward(self) -> str:
        from morning_brief.tools import save_tool_result

        full, summary = self._fetch_technicals()
        if full is None:
            return summary

        eia = self._fetch_eia()
        if eia:
            wow_raw = eia.pop("wow_raw")
            full["eia"] = eia
            summary += f" | EIA: {eia['direction']} {wow_raw:+.0f}kb"

        headlines = self._fetch_headlines()
        headlines.sort(key=lambda h: h["critical"], reverse=True)
        full["critical_headlines"] = headlines[:10]
        crit_count = sum(1 for h in headlines if h["critical"])
        if headlines:
            summary += f" | {crit_count} critical headlines"

        save_tool_result("wti_market", full)
        return summary

    @staticmethod
    def _calc_rsi(series, period=14):
        from morning_brief.tools.indicators import calc_rsi
        return calc_rsi(series, period)

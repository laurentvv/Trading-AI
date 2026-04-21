import logging
from dataclasses import dataclass

import yfinance as yf

from eia_client import EIAClient
from llm_client import TEXT_LLM_MODEL, _query_ollama

logger = logging.getLogger(__name__)


@dataclass
class OilBenchConfig:
    wti_ticker: str = "CL=F"
    dxy_ticker: str = "DX-Y.NYB"
    brent_ticker: str = "BZ=F"
    allocation_strong_buy_threshold: float = 75.0
    allocation_buy_threshold: float = 55.0
    allocation_sell_threshold: float = 45.0
    allocation_strong_sell_threshold: float = 25.0
    lookback_days: int = 5


class OilBenchModel:
    def __init__(self, config: OilBenchConfig = None):
        self.config = config or OilBenchConfig()
        self.eia_client = EIAClient()

    def analyze(self, ticker: str, headlines: list = None) -> dict:
        if not EIAClient.is_oil_ticker(ticker):
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "analysis": "Non-oil ticker, OilBench skipped.",
            }

        try:
            price_data = self._fetch_price_data()
        except Exception as e:
            logger.error(f"OilBench price data fetch failed: {e}")
            price_data = {}

        try:
            eia_context = self.eia_client.get_fundamental_context()
            eia_text = self.eia_client.format_for_llm(eia_context)

            # Extract Brent Spot Price for spread calculation
            brent_spot = eia_context.get("brent_spot", {}).get("current")
            if brent_spot:
                price_data["brent_spot"] = brent_spot
        except Exception as e:
            logger.error(f"OilBench EIA context failed: {e}")
            eia_text = "EIA fundamental data unavailable."

        prompt = self._construct_prompt(price_data, eia_text, headlines)
        llm_response = self._query_llm(prompt)
        return self._translate_signal(llm_response)

    def _fetch_price_data(self) -> dict:
        result = {}
        tickers = {
            "wti": self.config.wti_ticker,
            "brent": self.config.brent_ticker,
            "dxy": self.config.dxy_ticker,
        }

        for name, ticker in tickers.items():
            try:
                data = yf.download(
                    ticker, period=f"{self.config.lookback_days}d", progress=False
                )
                if data.empty:
                    result[name] = {"price": None, "change_pct": None}
                    continue
                close = data["Close"]
                if hasattr(close, "iloc"):
                    close = close.iloc[:, 0] if close.ndim > 1 else close

                current = float(close.iloc[-1])
                start = float(close.iloc[0])
                change_pct = ((current - start) / start * 100) if start else 0
                result[name] = {"price": current, "change_pct": round(change_pct, 2)}
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({ticker}): {e}")
                result[name] = {"price": None, "change_pct": None}

        return result

    def _construct_prompt(
        self, price_data: dict, eia_text: str, headlines: list
    ) -> str:
        wti = price_data.get("wti", {})
        brent = price_data.get("brent", {})
        dxy = price_data.get("dxy", {})
        brent_spot_price = price_data.get("brent_spot")

        wti_price = wti.get("price")
        wti_change = wti.get("change_pct", 0)
        brent_price = brent.get("price")
        dxy_price = dxy.get("price")
        dxy_change = dxy.get("change_pct", 0)

        wti_str = f"${wti_price:.2f} ({wti_change:+.2f}% 5d)" if wti_price else "N/A"
        brent_str = f"${brent_price:.2f}" if brent_price else "N/A"

        # Dated Brent Spread Analysis
        spread_text = ""
        if wti_price and brent_price:
            spread_text += f" (WTI-Brent Spread: ${brent_price - wti_price:.2f})"

        if brent_spot_price and brent_price:
            dated_spread = brent_spot_price - brent_price
            spread_text += (
                f"\n- Dated Brent Spread (Spot vs Futures): ${dated_spread:.2f}"
            )
            if dated_spread > 10:
                spread_text += " [EXTREME PHYSICAL TENSION]"
            elif dated_spread < 3:
                spread_text += " [MARKET EASING/NORMAL]"

        dxy_str = f"{dxy_price:.2f} ({dxy_change:+.2f}% 5d)" if dxy_price else "N/A"

        headlines_text = "No recent oil-specific news available."
        if headlines:
            headlines_text = "\n".join(f"- {h}" for h in headlines[:10])

        prompt = f"""You are a senior commodity quantitative analyst specializing in WTI Crude Oil.
Analyze the following data to determine your recommended portfolio allocation (0-100%).

**Price Context:**
- WTI Spot: {wti_str}
- Brent Futures: {brent_str}{spread_text}
- Brent Spot (Dated): ${brent_spot_price:.2f} if available
- DXY: {dxy_str}

**EIA Fundamental Data:**
{eia_text}

**Recent Oil News:**
{headlines_text}

**Analysis Framework:**
1. Inventory Analysis: Compare current stocks to 5-year average. Builds > expected = bearish.
2. Supply/Demand Balance: Production trends, import levels, and REFINERY UTILIZATION.
3. Dated Brent Spread: Historical norm is $1-$2. High values (>$10) signal extreme physical scarcity. A NARROWING spread signals easing tension (Bullish for Stocks, Bearish for Oil).
4. DXY Impact: Stronger dollar = downward pressure on oil prices.
5. Brent-WTI Spread: Widening spread signals supply chain dynamics.

Return ONLY a JSON object:
{{"allocation": <float 0-100>, "reasoning": "<2-sentence analysis>"}}"""

        return prompt.strip()

    def _query_llm(self, prompt: str) -> dict:
        payload = {
            "model": TEXT_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "system": "You are a senior commodity quantitative analyst specializing in WTI Crude Oil. Return ONLY valid JSON.",
        }
        return _query_ollama(payload, expected_keys=["allocation", "reasoning"])

    def _translate_signal(self, llm_response: dict) -> dict:
        try:
            allocation = float(llm_response.get("allocation", 50.0))
        except (TypeError, ValueError):
            allocation = 50.0

        allocation = max(0.0, min(100.0, allocation))
        reasoning = str(llm_response.get("reasoning", "No reasoning provided"))

        if allocation >= self.config.allocation_strong_buy_threshold:
            signal = "STRONG_BUY"
            confidence = min(1.0, (allocation - 50) / 50)
        elif allocation >= self.config.allocation_buy_threshold:
            signal = "BUY"
            confidence = min(1.0, (allocation - 50) / 50)
        elif allocation <= self.config.allocation_strong_sell_threshold:
            signal = "STRONG_SELL"
            confidence = min(1.0, (50 - allocation) / 50)
        elif allocation <= self.config.allocation_sell_threshold:
            signal = "SELL"
            confidence = min(1.0, (50 - allocation) / 50)
        else:
            signal = "HOLD"
            # Add small jitter to 0.3 for visibility in dashboard
            import random

            confidence = 0.3 + (random.random() * 0.05)

        return {
            "signal": signal,
            "confidence": round(confidence, 3),
            "analysis": f"[OilBench] Alloc={allocation:.0f}% | {reasoning[:200]}",
        }

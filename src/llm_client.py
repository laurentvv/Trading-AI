import logging
import requests
import json
import pandas as pd
import base64
from pathlib import Path
import time

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_BASE_URL = "http://localhost:11434"
TEXT_LLM_MODEL = "gemma4:e4b"
VISUAL_LLM_MODEL = "gemma4:e4b"


def check_ollama_health(timeout: int = 5) -> bool:
    """Vérifie si Ollama est disponible en interrogeant /api/tags."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def construct_llm_prompt(
    latest_data: pd.DataFrame,
    headlines: list = None,
    web_context: str = None,
    vg_indicators: dict = None,
    ticker: str = "Unknown",
) -> str:
    """
    Constructs a detailed prompt for the LLM from the latest market data and news.
    """
    data = latest_data.iloc[0]
    news_text = (
        "\n".join([f"- {h}" for h in headlines[:15]])
        if headlines
        else "No recent news available."
    )
    web_text = (
        f"\n**Web Research / Macro Context:**\n{web_context}" if web_context else ""
    )

    # Déterminer le contexte de l'actif
    asset_type = (
        "OIL (WTI)"
        if "CRUD" in ticker.upper() or "CL=F" in ticker.upper()
        else "NASDAQ-100"
    )

    # Alternative Data / Speculative Sentiment (Hyperliquid)
    hl_text = ""
    if vg_indicators:
        hl_funding = vg_indicators.get("HL_OIL_funding")
        hl_oi = vg_indicators.get("HL_OIL_oi")
        if hl_funding is not None or hl_oi is not None:
            hl_text = f"\n**Speculative Sentiment (Hyperliquid {asset_type} Perps):**\n"
            if hl_funding is not None:
                hl_text += f"- Funding Rate: {hl_funding:.6f}% "
                hl_text += "(Positive = Longs dominant, Negative = Shorts dominant)\n"
            if hl_oi is not None:
                hl_text += f"- Open Interest: {hl_oi:.2f} (Trend strength indicator)\n"

    prompt = f"""
    Analyze the following market data and news for {ticker} ({asset_type}) to provide a highly accurate trading decision.
    Your priority is ACCURACY (justesse) over trading frequency.

    **Current Market Data for {ticker}:**
    - Close Price: {data["Close"]:.2f}
    - RSI (14): {data["RSI"]:.2f} ({"Overbought" if data["RSI"] > 70 else "Oversold" if data["RSI"] < 30 else "Neutral"})
    - MACD: {data["MACD"]:.4f} | Signal: {data["MACD_Signal"]:.4f}
    - Bollinger Bands Position: {data["BB_Position"]:.2f} (0=Bottom, 1=Top)
    - Short-term Trend: {"Bullish" if data["Trend_Short"] == 1 else "Bearish" if data["Trend_Short"] == -1 else "Neutral"}
    - Long-term Trend: {"Bullish" if data["Trend_Long"] == 1 else "Bearish" if data["Trend_Long"] == -1 else "Neutral"}
    {hl_text}
    **Recent News Headlines:**
    {news_text}{web_text}

    **Decision Rules for {asset_type}:**
    1. Priority: ACCURACY. If news contradict technicals or signals are weak/mixed, default to HOLD.
    2. Bullish trend + Positive news = High conviction BUY.
    3. Bearish trend + Negative news = High conviction SELL.
    4. {asset_type} Specific: Consider macroeconomic context (OPEC+ for Oil, Fed/Tech earnings for Nasdaq).
    5. Speculative Sentiment (HL): Extreme negative funding is often a contrarian BUY signal (bottoming).

    Provide your analysis ONLY as a valid JSON object.
    {{
      "signal": "BUY | SELL | HOLD",
      "confidence": <float 0.0 to 1.0>,
      "analysis": "A rigorous 2-sentence technical and fundamental justification."
    }}
    """
    return prompt.strip()


def get_llm_decision(
    latest_data: pd.DataFrame,
    headlines: list = None,
    web_context: str = None,
    vg_indicators: dict = None,
    ticker: str = "Unknown",
) -> dict:
    """
    Queries the textual LLM via Ollama to get a trading decision.
    """
    logger.info(f"Querying textual LLM for {ticker} decision...")
    prompt = construct_llm_prompt(
        latest_data, headlines, web_context, vg_indicators, ticker
    )
    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "system": "You are an expert financial analyst. Your task is to analyze market data and news to provide a trading decision in a valid JSON format.",
    }
    return _query_ollama(payload)


def get_visual_llm_decision(image_path: Path) -> dict:
    """
    Queries the visual LLM via Ollama with a chart image.
    """
    if not image_path.exists():
        logger.error(f"Chart image not found: {image_path}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": "Chart image missing."}

    logger.info(f"Querying visual LLM with image {image_path}...")

    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Could not read or encode image {image_path}: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "analysis": f"Error reading image: {e}",
        }

    prompt = """
    ACT AS A PROFESSIONAL CHART ANALYST. Analyze the attached price chart image.
    1. Patterns: Identify visible geometric patterns (Head & Shoulders, Triangles, Channels).
    2. Price Action: Note the recent candle behavior (rejection, momentum, gaps).
    3. Indicators: Look at the visual shape of indicators (RSI divergences, MACD crossovers).

    IMPORTANT: Your role is purely geometric and visual validation. 
    Output ONLY a valid JSON object exactly like this:
    {
      "signal": "BUY|SELL|HOLD",
      "confidence": <float 0.0-1.0>,
      "analysis": "2-3 sentence visual justification"
    }
    """

    payload = {
        "model": VISUAL_LLM_MODEL,
        "prompt": prompt.strip(),
        "images": [image_base64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
        "system": "You are a geometric chart analyst. Return ONLY JSON.",
    }
    return _query_ollama(payload)


def _query_ollama(
    payload: dict, max_retries: int = 3, expected_keys: list = None
) -> dict:
    """
    Enhanced helper function to send a request to the Ollama API.
    Includes robust JSON extraction and error handling.
    """
    if expected_keys is None:
        expected_keys = ["signal", "confidence", "analysis"]

    model_name = payload.get("model", "unknown")
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
            response.raise_for_status()

            response_data = response.json()
            raw_output = response_data.get("response", "").strip()

            if not raw_output or raw_output == "{}":
                logger.warning(
                    f"Attempt {attempt + 1}: Empty or trivial response from LLM."
                )
                continue

            # Robust JSON extraction (in case of markdown or preamble)
            json_str = raw_output
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            # Find first { and last } if still failing
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]

            try:
                llm_output = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(
                    f"Attempt {attempt + 1}: Could not parse JSON. Output: {raw_output[:100]}..."
                )
                continue

            # Normalize keys to lowercase for robustness
            llm_output = {k.lower(): v for k, v in llm_output.items()}

            if not all(key.lower() in llm_output for key in expected_keys):
                logger.warning(
                    f"Attempt {attempt + 1}: Missing keys in {list(llm_output.keys())}"
                )
                continue

            logger.info(f"LLM decision ({model_name}) received and validated.")
            return llm_output

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for LLM ({model_name}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                logger.error(
                    f"All {max_retries} attempts failed for LLM ({model_name})."
                )
                # Default response fallback
                return {
                    k: "HOLD" if k == "signal" else 0.0 if k == "confidence" else ""
                    for k in expected_keys
                }

    return {
        k: "HOLD" if k == "signal" else 0.0 if k == "confidence" else ""
        for k in expected_keys
    }

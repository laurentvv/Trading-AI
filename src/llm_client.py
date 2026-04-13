import logging
import requests
import json
import pandas as pd
import base64
from pathlib import Path
import time

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
TEXT_LLM_MODEL = "gemma4:e4b"
VISUAL_LLM_MODEL = "gemma4:e4b"

def construct_llm_prompt(latest_data: pd.DataFrame, headlines: list = None, web_context: str = None, vg_indicators: dict = None) -> str:
    """
    Constructs a detailed prompt for the LLM from the latest market data and news.
    """
    data = latest_data.iloc[0]
    news_text = "\n".join([f"- {h}" for h in headlines[:15]]) if headlines else "No recent news available."
    web_text = f"\n**Web Research / Macro Context:**\n{web_context}" if web_context else ""
    
    # Alternative Data / Speculative Sentiment (Hyperliquid)
    hl_text = ""
    if vg_indicators:
        hl_funding = vg_indicators.get('HL_OIL_funding')
        hl_oi = vg_indicators.get('HL_OIL_oi')
        if hl_funding is not None or hl_oi is not None:
            hl_text = "\n**Speculative Sentiment (Hyperliquid OIL Perps):**\n"
            if hl_funding is not None:
                hl_text += f"- Funding Rate: {hl_funding:.6f}% "
                hl_text += "(Positive = Longs dominant, Negative = Shorts dominant)\n"
            if hl_oi is not None:
                hl_text += f"- Open Interest: {hl_oi:.2f} (Trend strength indicator)\n"

    prompt = f"""
    Analyze the following market data and news for a NASDAQ-100 or OIL ETF to provide a highly accurate trading decision.
    Your priority is ACCURACY (justesse) over trading frequency.

    **Current Market Data:**
    - Close Price: {data['Close']:.2f}
    - RSI (14): {data['RSI']:.2f}
    - MACD: {data['MACD']:.4f} | Signal: {data['MACD_Signal']:.4f}
    - Bollinger Bands Position: {data['BB_Position']:.2f}
    - Short-term Trend: {'Bullish' if data['Trend_Short'] == 1 else 'Bearish' if data['Trend_Short'] == -1 else 'Neutral'}
    - Long-term Trend: {'Bullish' if data['Trend_Long'] == 1 else 'Bearish' if data['Trend_Long'] == -1 else 'Neutral'}
    {hl_text}
    **Recent News Headlines:**
    {news_text}{web_text}

    **Decision Rules:**
    1. Priority: ACCURACY. If news contradict technicals or signals are weak/mixed, default to HOLD.
    2. Bullish trend + Positive news = High conviction BUY.
    3. Bearish trend + Negative news = High conviction SELL.
    4. Speculative Sentiment (HL): Extreme funding can be a contrarian signal (e.g., very negative funding might signal a bottom/short squeeze).

    Provide your analysis ONLY as a valid JSON object.
    {{
      "signal": "BUY | SELL | HOLD",
      "confidence": <float 0.0 to 1.0>,
      "analysis": "A rigorous 2-sentence technical and fundamental justification."
    }}
    """
    return prompt.strip()

def get_llm_decision(latest_data: pd.DataFrame, headlines: list = None, web_context: str = None, vg_indicators: dict = None) -> dict:
    """
    Queries the textual LLM via Ollama to get a trading decision.
    """
    logger.info("Querying textual LLM for a trading decision...")
    prompt = construct_llm_prompt(latest_data, headlines, web_context, vg_indicators)
    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "system": "You are an expert financial analyst. Your task is to analyze market data and news to provide a trading decision in a valid JSON format."
    }
    return _query_ollama(payload)

def get_visual_llm_decision(image_path: Path) -> dict:
    """
    Queries the visual LLM via Ollama with a chart image.
    """
    logger.info(f"Querying visual LLM with image {image_path}...")

    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Could not read or encode image {image_path}: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Error reading image: {e}"}

    prompt = """
    Analyze the provided financial chart focusing ONLY on Price Action and geometric patterns.
    - Identify candlestick patterns (Hammers, Dojis, Engulfing).
    - Locate visual Support and Resistance zones.
    - Look for chart figures (Double Bottoms, Triangles, Channels).
    - Identify visual divergences between price and indicator shapes.
    
    IMPORTANT: Do not attempt to read the exact mathematical values of indicators; another model handles the numbers. Your role is purely geometric and visual validation.

    Provide your analysis in a valid JSON object:
    {
      "signal": "BUY|SELL|HOLD",
      "confidence": <float between 0.0 and 1.0>,
      "analysis": "<your 2-3 sentence analysis of the visual/geometric patterns identified>"
    }
    """

    payload = {
        "model": VISUAL_LLM_MODEL,
        "prompt": prompt.strip(),
        "images": [image_base64],
        "stream": False,
        "format": "json",
        "system": "You are an expert Price Action analyst and Geometric Chartist. You specialize in identifying visual patterns on financial charts without relying on raw numerical data."
    }
    return _query_ollama(payload)

def _query_ollama(payload: dict, max_retries: int = 3, expected_keys: list = None) -> dict:
    """
    Helper function to send a request to the Ollama API and handle the response, with retries.
    """
    if expected_keys is None:
        expected_keys = ["signal", "confidence", "analysis"]

    model_name = payload.get("model", "unknown")
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=600) # Extended timeout for CPU-based models (10 min)
            response.raise_for_status()

            response_data = response.json()
            llm_output_str = response_data.get('response', '{}')
            
            # Handle cases where the response is a string that needs to be parsed
            if isinstance(llm_output_str, str):
                llm_output = json.loads(llm_output_str)
            else:
                llm_output = llm_output_str

            if not all(key in llm_output for key in expected_keys):
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Response from LLM ({model_name}) is malformed: {llm_output}")
                raise ValueError("Invalid or malformed LLM response.")

            logger.info(f"LLM decision ({model_name}) received and validated.")
            return llm_output

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for LLM ({model_name}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1)) # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed for LLM ({model_name}).")
                if 'response' in locals() and hasattr(response, 'text'):
                    logger.error(f"Final raw response from LLM: {response.text}")

                # Default response fallback
                fallback = {k: "HOLD" if k == "signal" else 0.0 if k == "confidence" else f"LLM ({model_name}) failed after {max_retries} attempts: {e}" for k in expected_keys}
                return fallback
        except Exception as e:
            logger.error(f"Unexpected error when querying the LLM ({model_name}): {e}")
            fallback = {k: "HOLD" if k == "signal" else 0.0 if k == "confidence" else f"Unexpected error: {e}" for k in expected_keys}
            return fallback
    
    # This part should be unreachable, but as a fallback
    fallback = {k: "HOLD" if k == "signal" else 0.0 if k == "confidence" else "Fell through retry loop unexpectedly." for k in expected_keys}
    return fallback

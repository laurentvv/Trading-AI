import logging
import requests
import json
import pandas as pd
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
TEXT_LLM_MODEL = "gemma3:27b"
VISUAL_LLM_MODEL = "gemma3:27b"

def construct_llm_prompt(latest_data: pd.DataFrame) -> str:
    """
    Constructs a detailed prompt for the LLM from the latest market data.
    """
    data = latest_data.iloc[0]
    prompt = f"""
    Analyze the following market data and provide a trading decision in JSON format.

    **Current Market Data:**
    - Close Price: {data['Close']:.2f}
    - RSI (14): {data['RSI']:.2f}
    - MACD: {data['MACD']:.4f}
    - MACD Signal: {data['MACD_Signal']:.4f}
    - MACD Histogram: {data['MACD_Histogram']:.4f}
    - Bollinger Bands Position (0-1): {data['BB_Position']:.2f}
    - Short-term Trend (MA5 vs MA20): {'Bullish' if data['Trend_Short'] == 1 else 'Bearish' if data['Trend_Short'] == -1 else 'Neutral'}
    - Long-term Trend (MA20 vs MA50): {'Bullish' if data['Trend_Long'] == 1 else 'Bearish' if data['Trend_Long'] == -1 else 'Neutral'}

    **Instructions:**
    Respond with a single, valid JSON object with three keys: "signal" (string: "BUY", "SELL", or "HOLD"), "confidence" (float: 0.0 to 1.0), and "analysis" (string: a brief 2-3 sentence justification).
    """
    return prompt.strip()

def get_llm_decision(latest_data: pd.DataFrame) -> dict:
    """
    Queries the textual LLM via Ollama to get a trading decision.
    """
    logger.info("Querying textual LLM for a trading decision...")
    prompt = construct_llm_prompt(latest_data)
    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "system": "You are an expert financial analyst specializing in NASDAQ ETFs. Your task is to analyze market data and provide a trading decision in a valid JSON format."
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
    You are a technical analysis expert and a chartist. Analyze the provided financial chart.
    Identify key trends, chart patterns, and indicator dynamics (MAs, RSI, MACD, Volume).

    Provide your analysis in a valid JSON object with the following structure:
    {
      "signal": "BUY|SELL|HOLD",
      "confidence": <float between 0.0 and 1.0>,
      "analysis": "<your 2-3 sentence analysis of the visual patterns you identified>"
    }
    """

    payload = {
        "model": VISUAL_LLM_MODEL,
        "prompt": prompt.strip(),
        "images": [image_base64],
        "stream": False,
        "format": "json"
    }
    return _query_ollama(payload)

def _query_ollama(payload: dict) -> dict:
    """
    Helper function to send a request to the Ollama API and handle the response.
    """
    model_name = payload.get("model", "unknown")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120) # Longer timeout for visual models
        response.raise_for_status()

        response_data = response.json()
        llm_output = json.loads(response_data.get('response', '{}'))

        required_keys = ["signal", "confidence", "analysis"]
        if not all(key in llm_output for key in required_keys):
            logger.error(f"The response from the LLM ({model_name}) is malformed: {llm_output}")
            raise ValueError("Invalid or malformed LLM response.")

        logger.info(f"LLM decision ({model_name}) received and validated.")
        return llm_output

    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with the Ollama API ({model_name}): {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Error communicating with the Ollama API: {e}"}
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error decoding or validating the JSON response from the LLM ({model_name}): {e}")
        # Log the raw response text for debugging
        if 'response' in locals() and hasattr(response, 'text'):
            logger.error(f"Raw response from LLM: {response.text}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"The LLM response was not valid JSON or was malformed: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error when querying the LLM ({model_name}): {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Unexpected error: {e}"}

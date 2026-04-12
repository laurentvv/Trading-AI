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

def construct_llm_prompt(latest_data: pd.DataFrame, headlines: list = None, web_context: str = None) -> str:
    """
    Constructs a detailed prompt for the LLM from the latest market data and news.
    """
    data = latest_data.iloc[0]
    news_text = "\n".join([f"- {h}" for h in headlines[:15]]) if headlines else "No recent news available."
    web_context_text = web_context if web_context else "No web research context provided."
    
    # Specific hint for Oil (WTI)
    oil_hint = ""
    # Check if data has a Ticker column or if we can infer from news context
    is_oil = any(x in str(web_context_text).upper() for x in ["WTI", "CRUDE OIL", "CL=F"])
    if is_oil:
        oil_hint = """
        **CRITICAL OIL HINT (Hyperliquid):**
        Hyperliquid (DEX) trades 'flx:OIL' or 'OIL-USDH' 24/7. Use the web research context to identify if 
        crypto-traders are front-running a move. Positive funding or price divergence on Hyperliquid 
        during weekend/closed hours is a powerful leading indicator for WTI Crude Oil (CL=F).
        """

    prompt = f"""
    Analyze the following market data and news for a financial asset to provide a highly accurate trading decision.
    Your priority is ACCURACY (justesse) over trading frequency.

    **Current Market Data:**
    - Close Price: {data['Close']:.2f}
    - RSI (14): {data['RSI']:.2f}
    - MACD: {data['MACD']:.4f} | Signal: {data['MACD_Signal']:.4f}
    - Bollinger Bands Position: {data['BB_Position']:.2f}
    - Short-term Trend: {'Bullish' if data['Trend_Short'] == 1 else 'Bearish' if data['Trend_Short'] == -1 else 'Neutral'}
    - Long-term Trend: {'Bullish' if data['Trend_Long'] == 1 else 'Bearish' if data['Trend_Long'] == -1 else 'Neutral'}

    {oil_hint}

    **Recent News Headlines:**
    {news_text}

    **Deep Web Research Context:**
    {web_context_text}

    **Decision Rules:**
    1. Priority: ACCURACY. If news contradict technicals or signals are weak/mixed, default to HOLD.
    2. Bullish trend + Positive news = High conviction BUY.
    3. Bearish trend + Negative news = High conviction SELL.

    Provide your analysis ONLY as a valid JSON object.
    {{
      "signal": "BUY | SELL | HOLD",
      "confidence": <float 0.0 to 1.0>,
      "analysis": "A rigorous 2-sentence technical and fundamental justification."
    }}
    """
    return prompt.strip()

def get_llm_decision(latest_data: pd.DataFrame, headlines: list = None, web_context: str = None) -> dict:
    """
    Queries the textual LLM via Ollama to get a trading decision.
    """
    logger.info("Querying textual LLM for a trading decision...")
    prompt = construct_llm_prompt(latest_data, headlines, web_context)
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


def generate_search_query(ticker: str) -> str:
    """
    Uses the LLM to generate the most relevant web search query for a given ticker.
    """
    logger.info(f"Generating dynamic web search query for {ticker}...")

    prompt = f"""
    You are an expert macroeconomic analyst. I am preparing to analyze the financial asset '{ticker}'.
    To gather the most relevant long-term context, I need to search the web for recent macroeconomic forecasts,
    fundamental events, or policy decisions that directly impact this specific asset.

    Generate the single most effective web search query (in English, maximum 10 words) to find this information.
    For example, if the asset is a tech index, you might search for Federal Reserve rates or semiconductor supply chain.
    If it's oil, you might search for OPEC+ decisions and global demand.

    Provide your response ONLY as a valid JSON object:
    {{
      "query": "<your optimized search query>"
    }}
    """

    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt.strip(),
        "stream": False,
        "format": "json",
        "system": "You are a web research assistant. Output only a valid JSON object."
    }

    try:
        def validate_search(llm_output):
            if "query" not in llm_output:
                raise ValueError("Missing 'query' key in LLM response.")
            return True

        response = _query_ollama(payload, validation_func=validate_search)
        query = response.get("query")
        if query:
            logger.info(f"Generated search query: '{query}'")
            return query
    except Exception as e:
        logger.error(f"Failed to generate search query: {e}")

    # Fallback
    fallback_query = f"Macroeconomic forecast and market analysis for {ticker}"
    logger.warning(f"Using fallback search query: '{fallback_query}'")
    return fallback_query

def _query_ollama(payload: dict, max_retries: int = 3, validation_func=None) -> dict:
    """
    Helper function to send a request to the Ollama API and handle the response, with retries.
    """
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

            # Default validation if no function provided
            if validation_func is None:
                required_keys = ["signal", "confidence", "analysis"]
                if not all(key in llm_output for key in required_keys):
                    raise ValueError(f"Response from LLM ({model_name}) is missing required decision keys: {llm_output}")
            else:
                validation_func(llm_output)

            logger.info(f"LLM response ({model_name}) received and validated.")
            return llm_output

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for LLM ({model_name}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1)) # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed for LLM ({model_name}).")
                if 'response' in locals() and hasattr(response, 'text'):
                    logger.error(f"Final raw response from LLM: {response.text}")
                
                # If it was a decision, return a default HOLD
                if validation_func is None:
                    return {"signal": "HOLD", "confidence": 0.0, "analysis": f"LLM ({model_name}) failed: {e}"}
                raise e # For other types of queries, let the caller handle it
        except Exception as e:
            logger.error(f"Unexpected error when querying the LLM ({model_name}): {e}")
            if validation_func is None:
                return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Unexpected error: {e}"}
            raise e
    
    return {"signal": "HOLD", "confidence": 0.0, "analysis": "Fell through retry loop unexpectedly."}

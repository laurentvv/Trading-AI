# Ajout recherche WEB

Uitlisation de  **DuckDuckGo Search** via la librairie Python `ddgs` : src\web_researcher.py


Query LLM pour web search
def generate_search_query(ticker: str) -> str:
    """
    Uses the LLM to generate the most relevant web search query for a given ticker.
    """
    logger.info(f"Generating dynamic web search query for {ticker}...")

    # Specific instruction for Hyperliquid on Oil and NASDAQ
    extra_context = ""
    if any(x in ticker.upper() for x in ["CL=F", "OIL", "WTI", "CRUDP"]):
        extra_context = "Specifically check for 'flx:OIL' or 'OIL-USDH' price action and sentiment on Hyperliquid (DEX) as a leading indicator."
    elif any(x in ticker.upper() for x in ["^NDX", "NASDAQ", "QQQ", "SXRV"]):
        extra_context = "Specifically check for 'NDX' or tech index price action and sentiment on Hyperliquid (DEX) as a leading indicator for the US tech opening."

    prompt = f"""
    You are an expert macroeconomic analyst. I am preparing to analyze the financial asset '{ticker}'.
    To gather the most relevant long-term context, I need to search the web for recent macroeconomic forecasts,
    fundamental events, or policy decisions that directly impact this specific asset.
    {extra_context}

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

On ajoutera les infos de la rechere au prompt llm texte

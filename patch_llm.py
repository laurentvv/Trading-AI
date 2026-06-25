import re

with open('src/llm_client.py', 'r') as f:
    content = f.read()

import_statement = """from src.enhanced_decision_engine import ModelResult
try:
    from free_llm_api_keys import FreeLLMClient
    from free_llm_api_keys.exceptions import FreeLLMError
except ImportError:
    FreeLLMClient = None
    FreeLLMError = Exception
"""
content = content.replace('from src.enhanced_decision_engine import ModelResult', import_statement)

get_llm_decision_code = """
def get_llm_decision(
    latest_data: pd.DataFrame,
    headlines: list = None,
    web_context: str = None,
    vg_indicators: dict = None,
    ticker: str = "Unknown",
) -> ModelResult:
    \"\"\"
    Queries the textual LLM (via free-llm-api-keys first, with Ollama fallback) to get a trading decision.
    \"\"\"
    logger.info(f"Querying textual LLM for {ticker} decision...")
    prompt = construct_llm_prompt(latest_data, headlines, web_context, vg_indicators, ticker)

    system_prompt = "<|think|> You are an expert financial analyst. Your task is to analyze market data and news to provide a trading decision in a valid JSON format. Output ONLY the JSON object requested — never add a 'thought' key."

    if FreeLLMClient is not None:
        try:
            logger.info("Attempting to use FreeLLMClient for textual decision...")
            client = FreeLLMClient(type="texte")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response_text = client.chat(messages, temperature=0.4)

            raw_output = _strip_thinking_prefix(response_text)
            candidates = _extract_json_candidates(raw_output)

            expected_keys = ["signal", "confidence", "analysis"]
            llm_output = None
            for item in candidates:
                parsed = _find_dict_with_keys(item, expected_keys)
                if parsed is not None:
                    llm_output = parsed
                    break

            if llm_output is not None:
                logger.info("Successfully received textual decision from FreeLLMClient.")
                return ModelResult(
                    signal=llm_output.get("signal", "HOLD"),
                    confidence=float(llm_output.get("confidence", 0.0)),
                    reasoning=llm_output.get("analysis", "No analysis"),
                    metadata=llm_output,
                )
            else:
                logger.warning(f"FreeLLMClient returned invalid JSON: {response_text[:200]}")
        except Exception as e:
            logger.warning(f"FreeLLMClient failed for textual decision: {e}. Falling back to Ollama.")

    logger.info("Using Ollama fallback for textual decision...")
    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": SCHEMA_TRADING_DECISION,
        "options": {"temperature": 0.4, "num_predict": 1024},
        "system": system_prompt,
    }

    result_dict = _query_ollama(payload)
    return ModelResult(
        signal=result_dict.get("signal", "HOLD"),
        confidence=float(result_dict.get("confidence", 0.0)),
        reasoning=result_dict.get("analysis", "No analysis"),
        metadata=result_dict,
    )

def get_visual_llm_decision"""

content = re.sub(r'def get_llm_decision\(.*?def get_visual_llm_decision', get_llm_decision_code, content, flags=re.DOTALL)

with open('src/llm_client.py', 'w') as f:
    f.write(content)

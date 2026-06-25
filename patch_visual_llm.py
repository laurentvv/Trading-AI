import re

with open('src/llm_client.py', 'r') as f:
    content = f.read()

visual_llm_code = """
def get_visual_llm_decision(image_path: Path) -> ModelResult:
    \"\"\"
    Queries the visual LLM (via free-llm-api-keys first, with Ollama fallback) with a chart image.
    \"\"\"
    if not image_path.exists():
        logger.error(f"Chart image not found: {image_path}")
        return ModelResult("HOLD", 0.0, "Chart image missing.")

    logger.info(f"Querying visual LLM with image {image_path}...")

    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Could not read or encode image {image_path}: {e}")
        return ModelResult("HOLD", 0.0, f"Error reading image: {e}")

    prompt = \"\"\"
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
    \"\"\"
    system_prompt = "<|think|> You are a geometric chart analyst. Return ONLY the requested JSON object — never add a 'thought' key."

    if FreeLLMClient is not None:
        try:
            logger.info("Attempting to use FreeLLMClient for visual decision...")
            # We use a textual model but send the image in the prompt
            client = FreeLLMClient(type="texte")
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.strip()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            response_text = client.chat(messages, temperature=0.1)

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
                logger.info("Successfully received visual decision from FreeLLMClient.")
                return ModelResult(
                    signal=llm_output.get("signal", "HOLD"),
                    confidence=float(llm_output.get("confidence", 0.0)),
                    reasoning=llm_output.get("analysis", "No analysis"),
                    metadata=llm_output,
                )
            else:
                logger.warning(f"FreeLLMClient returned invalid JSON for visual: {response_text[:200]}")
        except Exception as e:
            logger.warning(f"FreeLLMClient failed for visual decision: {e}. Falling back to Ollama.")

    logger.info("Using Ollama fallback for visual decision...")
    payload = {
        "model": VISUAL_LLM_MODEL,
        "prompt": prompt.strip(),
        "images": [image_base64],
        "stream": False,
        "format": SCHEMA_TRADING_DECISION,
        "options": {"temperature": 0.1, "num_predict": 1024},
        "system": system_prompt,
    }

    result_dict = _query_ollama(payload)
    return ModelResult(
        signal=result_dict.get("signal", "HOLD"),
        confidence=float(result_dict.get("confidence", 0.0)),
        reasoning=result_dict.get("analysis", "No analysis"),
        metadata=result_dict,
    )
"""

content = re.sub(r'def get_visual_llm_decision.*?return ModelResult\([^)]+\)', visual_llm_code, content, flags=re.DOTALL)

with open('src/llm_client.py', 'w') as f:
    f.write(content)

import logging
import json
import re
import pandas as pd
from pathlib import Path
import time
import os
import asyncio
import concurrent.futures
from datetime import datetime
from src.enhanced_decision_engine import ModelResult
from nexusai_client import AIGateway

logger = logging.getLogger(__name__)

# JSON schemas for structured outputs
SCHEMA_TRADING_DECISION = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "confidence": {"type": "number"},
        "analysis": {"type": "string"},
    },
    "required": ["signal", "confidence", "analysis"],
    "additionalProperties": False,
}

SCHEMA_SEARCH_QUERY = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

SCHEMA_FINACUMEN_SOLVER = {
    "type": "object",
    "properties": {
        "python_code": {"type": "string"},
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD", "NONE"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["python_code", "action", "confidence", "reasoning"],
    "additionalProperties": False,
}

SCHEMA_FINACUMEN_ANNOTATOR = {
    "type": "object",
    "properties": {"directives": {"type": "array", "items": {"type": "string"}}},
    "required": ["directives"],
    "additionalProperties": False,
}

SCHEMA_OIL_ALLOCATION = {
    "type": "object",
    "properties": {
        "allocation": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["allocation", "reasoning"],
    "additionalProperties": False,
}

_THINKING_TOKENS = (
    "<channel|>",
    "<|channel|>",
    "<|thought|>",
    "<thought>",
    "</thought>",
    "thought|",
    "<|channel>thought",
    "<|channel>thought}",
    "<|channel>thought|>",
    "<|start|>",
    "<|end|>",
    "<|channel|response>",
)


def strip_thinking_debris(text: str) -> str:
    """Removes thinking channel debris from a prose string."""
    cleaned = text
    for tok in _THINKING_TOKENS:
        cleaned = cleaned.replace(tok, "")
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def _fallback_decision(expected_keys: list, *, reason: str = "all_retries_failed") -> dict:
    """Canonical HOLD fallback returned when LLM retries are exhausted."""
    out = {k: "HOLD" if k == "signal" else 0.0 if k == "confidence" else "" for k in expected_keys}
    out["failed"] = True
    out["failure_reason"] = reason
    return out


_LLM_DEBUG_FILE = Path("data_cache") / "llm_debug_fail.txt"
_LLM_DEBUG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def _dump_llm_failure(model_name: str, attempt: int, expected_keys: list, raw_output: str) -> None:
    """Appends a failure record to the debug file, with size cap."""
    if os.environ.get("TRADING_DEBUG_DUMP", "1") == "0":
        return
    try:
        _LLM_DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _LLM_DEBUG_FILE.exists() and _LLM_DEBUG_FILE.stat().st_size >= _LLM_DEBUG_MAX_BYTES:
            return
        with open(_LLM_DEBUG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- FAIL ATTEMPT {attempt} ({model_name}) ---\n")
            f.write(f"Expected keys: {expected_keys}\n")
            f.write(raw_output)
    except OSError as e:
        logger.warning(f"Could not write LLM debug dump: {e}")


def _run_sync(coro):
    """Executes an async coroutine synchronously from sync code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def check_ai_health(timeout: int = 5) -> bool:
    """Checks whether at least one AI provider is configured in the environment."""
    try:
        providers = AIGateway.get_configured_providers()
        return len(providers) > 0
    except Exception as e:
        logger.warning(f"Error checking AI health: {e}")
        return False


def check_ollama_health(timeout: int = 5) -> bool:
    """Backwards compatibility alias for check_ai_health."""
    return check_ai_health(timeout)


def get_morning_brief_context() -> str:
    """Reads the morning brief report if it was generated within the last 24 hours."""
    brief_path = Path("morning_brief/output/morning_market_brief.md")
    if brief_path.exists():
        if time.time() - brief_path.stat().st_mtime < 86400:
            try:
                with open(brief_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    return f"\n**Overnight AI Morning Brief (Extremely Important Context):**\n{content}\n"
            except Exception as e:
                logger.warning(f"Failed to read morning brief: {e}")
    return ""


COUNCIL_REPORTS_DIR = Path("docs/council_reports")
COUNCIL_STALENESS_SECONDS = 7 * 86400


def _find_latest_council_report() -> Path | None:
    """Returns the most recent council report by date embedded in its filename."""
    if not COUNCIL_REPORTS_DIR.exists():
        return None
    candidates = sorted(COUNCIL_REPORTS_DIR.glob("council_report_*.md"), reverse=True)
    return candidates[0] if candidates else None


def _extract_council_verdict(report_text: str) -> str:
    """Extracts the Judge's verdict section from a council report."""
    marker = "## Verdict du Juge"
    idx = report_text.find(marker)
    if idx == -1:
        return ""
    section = report_text[idx + len(marker):]

    annexe_idx = section.find("## Annexe")
    if annexe_idx != -1:
        section = section[:annexe_idx]

    return strip_thinking_debris(section)


def _load_fresh_council_report() -> tuple[float, str] | None:
    """Loads the latest council report if fresh, returning (age_days, text)."""
    report_path = _find_latest_council_report()
    if report_path is None:
        return None
    try:
        date_str = report_path.stem.rsplit("_", 1)[-1]
        report_date = datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, IndexError):
        logger.warning(f"Could not parse date from council report filename: {report_path.name}")
        return None
    age_seconds = (datetime.now() - report_date).total_seconds()
    if age_seconds < 0 or age_seconds > COUNCIL_STALENESS_SECONDS:
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.warning(f"Failed to read council report {report_path}: {e}")
        return None
    return (age_seconds / 86400.0, text)


def get_council_verdict_context() -> str:
    """Reads the most recent weekend-council verdict if still fresh (< 7 days)."""
    loaded = _load_fresh_council_report()
    if loaded is None:
        return ""
    _age_days, text = loaded
    verdict = _extract_council_verdict(text)
    if not verdict:
        return ""
    return f"\n**Weekend AI Council Verdict (Strategic Context):**\n{verdict}\n"


_TICKER_VERDICT_RE = re.compile(
    r"(?P<ticker>[\w.\-^=]+)\s*:\s*(?P<signal>BUY|SELL|HOLD)\s*\((?P<conf>[0-9]*[\.,]?[0-9]+)",
    re.IGNORECASE,
)


def get_council_ticker_stance(ticker: str) -> tuple[str | None, float]:
    """Returns the council's (signal, effective_confidence) for a ticker."""
    loaded = _load_fresh_council_report()
    if loaded is None:
        return (None, 0.0)
    age_days, report_text = loaded

    last_marker = report_text.rfind("VERDICT_TICKER")
    if last_marker == -1:
        logger.info("Council report has no VERDICT_TICKER block — skip vote.")
        return (None, 0.0)
    block = report_text[last_marker:]

    ticker_norm = ticker.strip().upper()
    for m in _TICKER_VERDICT_RE.finditer(block):
        if m.group("ticker").strip().upper() == ticker_norm:
            signal = m.group("signal").upper()
            try:
                confidence = float(m.group("conf").replace(",", "."))
            except ValueError:
                confidence = 0.0
            if confidence > 1.0:
                logger.warning(
                    f"Council ticker stance confidence {confidence} > 1.0 — "
                    f"interpreting as percent. Judge prompt may be ignored."
                )
                confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))
            decay = max(0.0, 1.0 - age_days / 7.0)
            return (signal, confidence * decay)

    logger.info(f"Council verdict found but no stance for ticker {ticker}")
    return (None, 0.0)


def construct_llm_prompt(
    latest_data: pd.DataFrame,
    headlines: list = None,
    web_context: str = None,
    vg_indicators: dict = None,
    ticker: str = "Unknown",
) -> str:
    """Constructs a detailed prompt for the LLM from the latest market data and news."""
    data = latest_data.iloc[0]
    news_text = "\n".join([f"- {h}" for h in headlines[:15]]) if headlines else "No recent news available."
    web_text = f"\n**Web Research / Macro Context:**\n{web_context}" if web_context else ""
    brief_text = get_morning_brief_context()
    council_text = get_council_verdict_context()

    asset_type = "OIL (WTI)" if "CRUD" in ticker.upper() or "CL=F" in ticker.upper() else "NASDAQ-100"

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
    {news_text}{web_text}{brief_text}{council_text}

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


def construct_visual_prompt() -> str:
    """Constructs the geometric visual chart analysis prompt."""
    prompt = """
    ACT AS A PROFESSIONAL CHART ANALYST. Analyze the attached price chart image.
    1. Patterns: Identify visible geometric patterns (Head & Shoulders, Triangles, Channels).
    2. Price Action: Note the recent candle behavior (rejection, momentum, gaps).
    3. Indicators: Look at the visual shape of indicators (RSI divergences, MACD crossovers).

    IMPORTANT: Your role is purely geometric and visual validation.
    - Output "BUY" when you see a recognizable uptrend (higher lows, breakout above resistance).
    - Output "SELL" when you see a recognizable downtrend (lower highs, breakdown below support).
    - Output "HOLD" ONLY when the chart is genuinely directionless (no clear trend or pattern).
    - Apply the SAME confidence standard to BUY and SELL (do not demand stronger evidence for one direction).

    Output ONLY a valid JSON object exactly like this:
    {
      "signal": "BUY|SELL|HOLD",
      "confidence": <float 0.0-1.0>,
      "analysis": "2-3 sentence visual justification"
    }
    """
    return prompt.strip()


def _extract_json_objects(text: str) -> list:
    """Extract all top-level JSON objects from a string."""
    objs = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                objs.append(obj)
            pos = start + end
        except (json.JSONDecodeError, ValueError):
            pos += 1
    return objs


def _find_dict_with_keys(node, expected_keys: list, _depth: int = 0):
    """Recursively search a parsed JSON node for a dict containing all expected keys."""
    if _depth > 6:
        return None

    if isinstance(node, dict):
        normalized = {str(k).lower(): v for k, v in node.items()}
        if all(k.lower() in normalized for k in expected_keys):
            return normalized
        for v in node.values():
            found = _find_dict_with_keys(v, expected_keys, _depth + 1)
            if found is not None:
                return found
        return None

    if isinstance(node, str):
        cleaned = strip_thinking_debris(node)
        for marker in ("```json", "```"):
            if marker in cleaned:
                for block in cleaned.split(marker)[1:]:
                    inner = block.split("```")[0]
                    for obj in _extract_json_objects(inner):
                        found = _find_dict_with_keys(obj, expected_keys, _depth + 1)
                        if found is not None:
                            return found
        for obj in _extract_json_objects(cleaned):
            found = _find_dict_with_keys(obj, expected_keys, _depth + 1)
            if found is not None:
                return found

    if isinstance(node, list):
        for v in node:
            found = _find_dict_with_keys(v, expected_keys, _depth + 1)
            if found is not None:
                return found

    return None


def _extract_json_candidates(raw_output: str) -> list:
    candidates = []
    if "```json" in raw_output:
        candidates.extend([b.split("```")[0].strip() for b in raw_output.split("```json")[1:]])
    elif "```" in raw_output:
        candidates.extend([b.split("```")[0].strip() for b in raw_output.split("```")[1:]])

    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(raw_output):
        try:
            start = raw_output.find("{", pos)
            if start == -1:
                break
            obj, end_idx = decoder.raw_decode(raw_output[start:])
            if isinstance(obj, dict):
                candidates.append(obj)
            pos = start + end_idx
        except (json.JSONDecodeError, ValueError):
            pos += 1

    if not candidates:
        candidates = [raw_output]
    return candidates


async def _async_query_nexus(
    prompt: str,
    system_prompt: str = None,
    expected_keys: list = None,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> dict:
    if expected_keys is None:
        expected_keys = ["signal", "confidence", "analysis"]

    for attempt in range(max_retries):
        try:
            async with AIGateway.auto_fallback() as client:
                resp = await client.generate_text(
                    prompt,
                    system_prompt=system_prompt or "You are an expert financial analyst. Return ONLY the requested JSON object.",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=True,
                )
                raw_output = resp.text.strip()
                if not raw_output or raw_output == "{}":
                    logger.warning(f"Attempt {attempt + 1}: Empty or trivial response from NexusAI [{resp.provider}].")
                    continue

                candidates = _extract_json_candidates(raw_output)
                llm_output = None
                for item in candidates:
                    parsed = _find_dict_with_keys(item, expected_keys)
                    if parsed is not None:
                        llm_output = parsed
                        break

                if llm_output is not None:
                    logger.info(f"NexusAI decision received via [{resp.provider} / {resp.model}].")
                    llm_output["_provider"] = resp.provider
                    llm_output["_model"] = resp.model
                    return llm_output

                logger.warning(f"Attempt {attempt + 1}: Could not find expected keys {expected_keys} in response from [{resp.provider}].")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with NexusAI: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))

    return _fallback_decision(expected_keys, reason="retries_exhausted_no_valid_json")


async def _async_query_nexus_vision(
    image_path: Path,
    prompt: str,
    system_prompt: str = None,
    expected_keys: list = None,
    temperature: float = 0.6,
    max_retries: int = 3,
) -> dict:
    if expected_keys is None:
        expected_keys = ["signal", "confidence", "analysis"]

    for attempt in range(max_retries):
        try:
            async with AIGateway.auto_fallback_vision() as client:
                resp = await client.analyze_image(
                    prompt,
                    image_path,
                    system_prompt=system_prompt or "You are an objective geometric chart analyst. Return ONLY the requested JSON object.",
                    temperature=temperature,
                    json_mode=True,
                )
                raw_output = resp.text.strip()
                candidates = _extract_json_candidates(raw_output)
                llm_output = None
                for item in candidates:
                    parsed = _find_dict_with_keys(item, expected_keys)
                    if parsed is not None:
                        llm_output = parsed
                        break

                if llm_output is not None:
                    logger.info(f"NexusAI Vision decision received via [{resp.provider} / {resp.model}].")
                    llm_output["_provider"] = resp.provider
                    llm_output["_model"] = resp.model
                    return llm_output

                logger.warning(f"Attempt {attempt + 1}: Could not find expected keys {expected_keys} in Vision response.")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with NexusAI Vision: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))

    return _fallback_decision(expected_keys, reason="vision_retries_exhausted")


def _query_nexus(
    prompt: str,
    system_prompt: str = None,
    expected_keys: list = None,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> dict:
    """Synchronous wrapper for NexusAI text query."""
    return _run_sync(
        _async_query_nexus(
            prompt,
            system_prompt=system_prompt,
            expected_keys=expected_keys,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    )


def _query_nexus_vision(
    image_path: Path,
    prompt: str,
    system_prompt: str = None,
    expected_keys: list = None,
    temperature: float = 0.6,
    max_retries: int = 3,
) -> dict:
    """Synchronous wrapper for NexusAI vision query."""
    return _run_sync(
        _async_query_nexus_vision(
            image_path,
            prompt,
            system_prompt=system_prompt,
            expected_keys=expected_keys,
            temperature=temperature,
            max_retries=max_retries,
        )
    )


def _query_ollama(payload: dict, max_retries: int = 3, expected_keys: list = None) -> dict:
    """Backwards compatibility alias delegating to NexusAI."""
    prompt = payload.get("prompt", "")
    system_prompt = payload.get("system")
    return _query_nexus(prompt, system_prompt=system_prompt, expected_keys=expected_keys, max_retries=max_retries)


def get_llm_decision(
    latest_data: pd.DataFrame,
    headlines: list = None,
    web_context: str = None,
    vg_indicators: dict = None,
    ticker: str = "Unknown",
) -> ModelResult:
    """Queries NexusAI Gateway across configured AI providers for a trading decision."""
    logger.info(f"Querying NexusAI textual LLM for {ticker} decision...")
    prompt = construct_llm_prompt(latest_data, headlines, web_context, vg_indicators, ticker)
    expected_keys = ["signal", "confidence", "analysis"]

    result_dict = _query_nexus(
        prompt,
        system_prompt="You are an expert financial analyst. Return ONLY the requested JSON object.",
        expected_keys=expected_keys,
        temperature=0.4,
        max_tokens=1024,
    )

    provider = result_dict.get("_provider", "nexusai")
    model = result_dict.get("_model", "auto")

    return ModelResult(
        signal=result_dict.get("signal", "HOLD"),
        confidence=result_dict.get("confidence", 0.0),
        reasoning=result_dict.get("analysis", "No analysis"),
        metadata={**result_dict, "backend": f"nexus_{provider}_{model}"},
    )


def get_visual_llm_decision(image_path: Path) -> ModelResult:
    """Queries NexusAI Vision Gateway with a chart image."""
    if not image_path.exists():
        logger.error(f"Chart image not found: {image_path}")
        return ModelResult("HOLD", 0.0, "Chart image missing.", {"backend": "none"})

    logger.info(f"Querying NexusAI visual LLM with image {image_path}...")
    prompt = construct_visual_prompt()
    expected_keys = ["signal", "confidence", "analysis"]

    result_dict = _query_nexus_vision(
        image_path,
        prompt,
        system_prompt="You are an objective geometric chart analyst. Return ONLY the requested JSON object.",
        expected_keys=expected_keys,
        temperature=0.6,
    )

    provider = result_dict.get("_provider", "nexusai_vision")
    model = result_dict.get("_model", "auto")

    return ModelResult(
        signal=result_dict.get("signal", "HOLD"),
        confidence=result_dict.get("confidence", 0.0),
        reasoning=result_dict.get("analysis", "No analysis"),
        metadata={**result_dict, "backend": f"nexus_vision_{provider}_{model}"},
    )

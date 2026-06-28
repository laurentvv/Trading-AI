import logging
import requests
import json
import re
import pandas as pd
import base64
from pathlib import Path
import time
import os
from datetime import datetime
from src.enhanced_decision_engine import ModelResult

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_BASE_URL = "http://localhost:11434"
TEXT_LLM_MODEL = "hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K"
VISUAL_LLM_MODEL = "hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K"

# JSON schemas used as Ollama `format` parameter. Using a strict schema
# (additionalProperties: false) physically prevents the Gemma thinking model
# from adding a "thought" key — the root cause of the JSON extraction failures.
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

# Thinking-mode debris tokens emitted by Gemma 4 when schema enforcement
# is bypassed or when the model leaks reasoning into JSON string values.
# Single source of truth — used by both _query_ollama (prefix strip) and
# _find_dict_with_keys (recursive string-value scrub).
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
    """Removes Gemma ``<|think|>`` channel debris from a prose string.

    In prose mode (no JSON schema), the think channel can leak leading
    ``<|channel>thought`` markers into the model's output. This helper strips
    every token in :data:`_THINKING_TOKENS` and collapses the resulting blank
    lines, returning clean text. Reused by the weekend council (prose mode)
    so its reports stay readable.
    """
    cleaned = text
    for tok in _THINKING_TOKENS:
        cleaned = cleaned.replace(tok, "")
    # Collapse 3+ consecutive newlines (left after stripping block tokens)
    # down to a single blank line.
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def _fallback_decision(expected_keys: list, *, reason: str = "all_retries_failed") -> dict:
    """Canonical HOLD fallback returned when LLM retries are exhausted.

    The ``failed`` flag and ``failure_reason`` are emitted for observability
    (logs, metrics, future filtering). They are NOT yet consumed by the
    consensus aggregator in ``enhanced_decision_engine.py`` — a downstream
    consumer must be added before this flag can be relied upon to exclude
    the vote from the weighted aggregation.
    """
    out = {k: "HOLD" if k == "signal" else 0.0 if k == "confidence" else "" for k in expected_keys}
    out["failed"] = True
    out["failure_reason"] = reason
    return out


# Cap the debug-fail dump file to 5 MB so it can never fill the disk.
# Disable entirely by setting TRADING_DEBUG_DUMP=0 in the environment.
_LLM_DEBUG_FILE = Path("data_cache") / "llm_debug_fail.txt"
_LLM_DEBUG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def _dump_llm_failure(model_name: str, attempt: int, expected_keys: list, raw_output: str) -> None:
    """Appends a failure record to the debug file, with size cap.

    Skipped entirely if env TRADING_DEBUG_DUMP=0 or if the file already exceeds cap.
    """
    if os.environ.get("TRADING_DEBUG_DUMP", "1") == "0":
        return
    try:
        _LLM_DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _LLM_DEBUG_FILE.exists() and _LLM_DEBUG_FILE.stat().st_size >= _LLM_DEBUG_MAX_BYTES:
            return  # Cap reached — silently drop further dumps
        with open(_LLM_DEBUG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- FAIL ATTEMPT {attempt} ({model_name}) ---\n")
            f.write(f"Expected keys: {expected_keys}\n")
            f.write(raw_output)
    except OSError as e:
        logger.warning(f"Could not write LLM debug dump: {e}")


def check_ollama_health(timeout: int = 5) -> bool:
    """Vérifie si Ollama est disponible en interrogeant /api/tags."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


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


# Council verdict is a weekly retrospective, not a daily artefact: a report
# produced on the weekend must stay relevant through the following trading
# week. The council runs ONCE per week on Saturday at 01:00, so a report
# produced Saturday must stay fresh until the next Saturday's run. 7 days
# covers the full week. Picked by filename date (robust to git-pull mtime
# resets), unlike the morning brief which uses mtime.
COUNCIL_REPORTS_DIR = Path("docs/council_reports")
COUNCIL_STALENESS_SECONDS = 7 * 86400


def _find_latest_council_report() -> Path | None:
    """Returns the most recent council report by date embedded in its filename.

    The filename format is ``council_report_YYYY-MM-DD.md``. Sorting on the
    filename date is more reliable than mtime, which a ``git pull`` or file
    copy can reset.
    """
    if not COUNCIL_REPORTS_DIR.exists():
        return None
    candidates = sorted(COUNCIL_REPORTS_DIR.glob("council_report_*.md"), reverse=True)
    return candidates[0] if candidates else None


def _extract_council_verdict(report_text: str) -> str:
    """Extracts the Judge's verdict section from a council report.

    The verdict runs from ``## Verdict du Juge`` up to the ``## Annexe``
    marker that precedes the full debate transcript. Only the verdict (the
    actionable synthesis) is returned — the transcript is deliberately omitted
    to keep the per-cycle prompt token cost bounded.

    Note: the verdict itself often contains internal ``---`` separators (e.g.
    between the intro and the recommendations), so we cannot cut at the first
    ``---``. We cut at the ``## Annexe`` boundary instead.

    The council runs in prose mode (no JSON schema), so Gemma's think channel
    may leak leading ``<|channel>thought`` debris into the verdict. We strip
    those tokens here so the injected strategic context stays clean.
    """
    marker = "## Verdict du Juge"
    idx = report_text.find(marker)
    if idx == -1:
        return ""
    section = report_text[idx + len(marker):]

    # Cut at the Annexe boundary (the transcript), not the first --- separator,
    # since the Judge's own verdict uses --- internally.
    annexe_idx = section.find("## Annexe")
    if annexe_idx != -1:
        section = section[:annexe_idx]

    return strip_thinking_debris(section)


def _load_fresh_council_report() -> tuple[float, str] | None:
    """Loads the latest council report if fresh, returning ``(age_days, text)``.

    Single source of truth for freshness + report loading, shared by both the
    text-injection path (:func:`get_council_verdict_context`) and the consensus
    vote path (:func:`get_council_ticker_stance`). Resolves the report ONCE so
    the two paths cannot disagree on freshness (audit finding: duplicated
    staleness logic that could drift) and avoids the TOCTOU race of calling
    ``_find_latest_council_report`` twice.
    """
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
    """Reads the most recent weekend-council verdict if still fresh (< 7 days).

    Mirror of :func:`get_morning_brief_context`: injects the council's strategic
    verdict into the per-cycle decision prompt so ``main.py`` benefits from the
    weekend retrospective automatically.
    """
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
    """Returns the council's (signal, effective_confidence) for a ticker.

    The council's Judge emits a parseable ``VERDICT_TICKER:`` block at the end
    of its report. This extracts the stance for the requested ticker and
    applies a linear age decay: a fresh verdict (day 0) keeps full confidence,
    decaying to 0 at day 7 (aligned with ``COUNCIL_STALENESS_SECONDS``).

    Returns ``(None, 0.0)`` if no report, ticker absent, or stale — the caller
    then omits the council vote (graceful, like any other optional model).

    Note: pass the TRADING ticker (e.g. ``SXRV.DE``), not the analysis ticker
    (``^NDX``) — the council's context and verdict use trading tickers.
    """
    loaded = _load_fresh_council_report()
    if loaded is None:
        return (None, 0.0)
    age_days, report_text = loaded

    # Isolate the LAST VERDICT_TICKER block (the Judge may reference it in prose
    # earlier; the real block is the final one). Everything after it is parsed.
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
            # Reject obvious misreads: the prompt asks for 0.0-1.0 but LLMs may
            # emit "85" (percent). Treat >1 as percent and rescale, with a warning.
            if confidence > 1.0:
                logger.warning(
                    f"Council ticker stance confidence {confidence} > 1.0 — "
                    f"interpreting as percent. Judge prompt may be ignored."
                )
                confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))
            # Linear decay: full at day 0 → 0 at day 7.
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
    """
    Constructs a detailed prompt for the LLM from the latest market data and news.
    """
    data = latest_data.iloc[0]
    news_text = "\n".join([f"- {h}" for h in headlines[:15]]) if headlines else "No recent news available."
    web_text = f"\n**Web Research / Macro Context:**\n{web_context}" if web_context else ""
    brief_text = get_morning_brief_context()
    council_text = get_council_verdict_context()

    # Déterminer le contexte de l'actif
    asset_type = "OIL (WTI)" if "CRUD" in ticker.upper() or "CL=F" in ticker.upper() else "NASDAQ-100"

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


def get_llm_decision(
    latest_data: pd.DataFrame,
    headlines: list = None,
    web_context: str = None,
    vg_indicators: dict = None,
    ticker: str = "Unknown",
) -> ModelResult:
    """
    Queries a textual LLM to get a trading decision.

    3-tier fallback chain: Gemini (Free Tier, if GEMINI_API_KEY is set) →
    FreeLLMClient (shared-key cloud proxy) → local Ollama. Each tier is
    optional and silently skipped on failure or absence; the pipeline never
    raises here. The winning backend is recorded in ``metadata["backend"]``
    for observability.
    """
    logger.info(f"Querying textual LLM for {ticker} decision...")
    prompt = construct_llm_prompt(latest_data, headlines, web_context, vg_indicators, ticker)

    # 0. Try Gemini reasoning tier (priority cloud tier if GEMINI_API_KEY is
    # configured). Internally this is a multi-model cascade (gemini-3.5-flash →
    # 3-flash-preview → 3.1-flash-lite → 2.5-flash → gemma-4-31b → gemma-4-26b);
    # Gemini 2.5 Pro is unavailable (0/0/0) on this key. See gemini_gateway.py.
    try:
        from src.gemini_gateway import GeminiGateway

        gateway = GeminiGateway()
        if gateway.enabled:
            logger.info("Trying Gemini reasoning tier for textual decision...")
            gemini_result = gateway.decide(prompt)
            if gemini_result is not None:
                # Belt-and-braces (AGENTS.md §2.1 Layer 2): re-extract through
                # the same JSON scrubber the cloud path uses, even though
                # Gemini's response_schema already enforces structure.
                parsed = _find_dict_with_keys(gemini_result, ["signal", "confidence", "analysis"]) or gemini_result
                logger.info("Successfully got textual decision from Gemini reasoning tier.")
                return ModelResult(
                    signal=parsed.get("signal", "HOLD"),
                    confidence=parsed.get("confidence", 0.0),
                    reasoning=parsed.get("analysis", "No analysis"),
                    metadata={**parsed, "backend": "gemini_reasoning"},
                )
            logger.info("Gemini reasoning tier unavailable/declined. Trying next backend.")
    except Exception as e:
        logger.warning(f"Gemini reasoning tier failed: {e}. Trying next backend.")

    # 1. Try FreeLLMClient
    try:
        from free_llm_api_keys import FreeLLMClient
        # Auto-rotates through available text models
        client = FreeLLMClient(type="texte")
        messages = [
            {"role": "system", "content": "You are an expert financial analyst. Your task is to analyze market data and news to provide a trading decision in a valid JSON format. Output ONLY the JSON object requested."},
            {"role": "user", "content": prompt}
        ]
        
        logger.info("Trying FreeLLMClient for textual decision...")
        response_text = client.chat(messages, temperature=0.4, max_tokens=1024)
        
        # Parse the JSON
        candidates = _extract_json_candidates(response_text)
        expected_keys = ["signal", "confidence", "analysis"]
        for item in candidates:
            parsed = _find_dict_with_keys(item, expected_keys)
            if parsed is not None:
                logger.info("Successfully got textual decision from FreeLLMClient.")
                return ModelResult(
                    signal=parsed.get("signal", "HOLD"),
                    confidence=parsed.get("confidence", 0.0),
                    reasoning=parsed.get("analysis", "No analysis"),
                    metadata={**parsed, "backend": "free_llm"},
                )
        logger.warning("FreeLLMClient returned invalid JSON. Falling back to Ollama.")
    except Exception as e:
        logger.warning(f"FreeLLMClient failed: {e}. Falling back to Ollama.")

    # 2. Fallback to Ollama
    logger.info("Using Ollama fallback for textual decision...")
    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": SCHEMA_TRADING_DECISION,
        "options": {"temperature": 0.4, "num_predict": 1024},
        "system": "<|think|> You are an expert financial analyst. Your task is to analyze market data and news to provide a trading decision in a valid JSON format. Output ONLY the JSON object requested — never add a 'thought' key.",
    }

    result_dict = _query_ollama(payload)
    return ModelResult(
        signal=result_dict.get("signal", "HOLD"),
        confidence=result_dict.get("confidence", 0.0),
        reasoning=result_dict.get("analysis", "No analysis"),
        metadata={**result_dict, "backend": "ollama"},
    )


def get_visual_llm_decision(image_path: Path) -> ModelResult:
    """
    Queries the visual LLM with a chart image.

    Vision tiers: Gemini vision cascade (if ``GEMINI_API_KEY`` is set) → local
    Ollama. The Gemini tier tries multiple vision-capable Flash models in turn
    (gemini-3.5-flash → 3-flash-preview → 2.5-flash → 2.0-flash); see
    gemini_gateway.VISION_CASCADE. Historically vision was Ollama-only because
    free proxies reject image payloads (AGENTS.md §6.1); a first-party Gemini
    key re-enables cloud vision. The winning backend is recorded in
    ``metadata["backend"]``.
    """
    if not image_path.exists():
        logger.error(f"Chart image not found: {image_path}")
        return ModelResult("HOLD", 0.0, "Chart image missing.", {"backend": "none"})

    logger.info(f"Querying visual LLM with image {image_path}...")

    # 0. Try Gemini vision tier first if enabled.
    # NOTE: read/encode the file lazily below only when we actually need the
    # Ollama path, so the Gemini path doesn't pay the base64 cost.
    try:
        from src.gemini_gateway import GeminiGateway

        gateway = GeminiGateway()
        if gateway.enabled:
            logger.info("Trying Gemini vision tier for visual decision...")
            gemini_result = gateway.analyze_chart(image_path)
            if gemini_result is not None:
                logger.info("Successfully got visual decision from Gemini vision tier.")
                return ModelResult(
                    signal=gemini_result.get("signal", "HOLD"),
                    confidence=gemini_result.get("confidence", 0.0),
                    reasoning=gemini_result.get("analysis", "No analysis"),
                    metadata={**gemini_result, "backend": "gemini_vision"},
                )
            logger.info("Gemini vision tier unavailable/declined. Falling back to Ollama.")
    except Exception as e:
        logger.warning(f"Gemini vision tier failed: {e}. Falling back to Ollama.")

    # 1. Fallback to Ollama
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Could not read or encode image {image_path}: {e}")
        return ModelResult("HOLD", 0.0, f"Error reading image: {e}", {"backend": "none"})

    prompt = """
    ACT AS A PROFESSIONAL CHART ANALYST. Analyze the attached price chart image.
    1. Patterns: Identify visible geometric patterns (Head & Shoulders, Triangles, Channels).
    2. Price Action: Note the recent candle behavior (rejection, momentum, gaps).
    3. Indicators: Look at the visual shape of indicators (RSI divergences, MACD crossovers).

    IMPORTANT: Your role is purely geometric and visual validation.
    - If the chart is ambiguous, mixed, or mostly sideways, you MUST output "HOLD" with low confidence (< 0.5).
    - Reserve "BUY" or "SELL" with high confidence (> 0.7) ONLY for textbook, unmistakable patterns.
    
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
        "format": SCHEMA_TRADING_DECISION,
        "options": {"temperature": 0.4, "num_predict": 1024},
        "system": "<|think|> You are an objective geometric chart analyst. Return ONLY the requested JSON object — never add a 'thought' key.",
    }

    result_dict = _query_ollama(payload)
    return ModelResult(
        signal=result_dict.get("signal", "HOLD"),
        confidence=result_dict.get("confidence", 0.0),
        reasoning=result_dict.get("analysis", "No analysis"),
        metadata={**result_dict, "backend": "ollama"},
    )


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
    """Recursively search a parsed JSON node for a dict containing all expected keys.

    The Gemma thinking model sometimes wraps its real answer inside a ``"thought"``
    string value (e.g. ``{"thought": "<channel|>```json{\\"query\\": \\"...\\"}```"} ``).
    This helper drills into string values, strips thinking/markdown debris, and
    looks for the first nested dict that has every required key.

    Returns the matched dict (with lowercased keys) or ``None``.
    """
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
        cleaned = node
        for tok in _THINKING_TOKENS:
            cleaned = cleaned.replace(tok, "")
        # Markdown JSON blocks first.
        for marker in ("```json", "```"):
            if marker in cleaned:
                for block in cleaned.split(marker)[1:]:
                    inner = block.split("```")[0]
                    for obj in _extract_json_objects(inner):
                        found = _find_dict_with_keys(obj, expected_keys, _depth + 1)
                        if found is not None:
                            return found
        # Bare JSON inside the string.
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


def _strip_thinking_prefix(raw_output: str) -> str:
    first_brace = raw_output.find("{")
    if first_brace > 0:
        prefix = raw_output[:first_brace]
        if any(tag in prefix for tag in _THINKING_TOKENS):
            return raw_output[first_brace:].strip()
    elif first_brace == -1:
        for tag in _THINKING_TOKENS:
            if tag in raw_output:
                return raw_output.split(tag)[-1].strip()
    return raw_output


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


def _query_ollama(payload: dict, max_retries: int = 3, expected_keys: list = None) -> dict:
    if expected_keys is None:
        expected_keys = ["signal", "confidence", "analysis"]

    model_name = payload.get("model", "unknown")
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
            response.raise_for_status()

            raw_output = response.json().get("response", "").strip()
            if not raw_output or raw_output == "{}":
                logger.warning(f"Attempt {attempt + 1}: Empty or trivial response from LLM.")
                continue

            raw_output = _strip_thinking_prefix(raw_output)
            candidates = _extract_json_candidates(raw_output)

            llm_output = None
            for item in candidates:
                parsed = _find_dict_with_keys(item, expected_keys)
                if parsed is not None:
                    llm_output = parsed
                    break

            if llm_output is None:
                logger.error(
                    f"Attempt {attempt + 1}: Could not find valid JSON with keys {expected_keys}. Raw (first 500 chars): {raw_output[:500]}"
                )
                if len(raw_output) > 500:
                    _dump_llm_failure(model_name, attempt + 1, expected_keys, raw_output)
                continue

            logger.info(f"LLM decision ({model_name}) received and validated.")
            return llm_output

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for LLM ({model_name}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                logger.error(f"All {max_retries} attempts failed for LLM ({model_name}).")
                return _fallback_decision(expected_keys, reason=f"exception: {type(e).__name__}")

    return _fallback_decision(expected_keys, reason="retries_exhausted_no_valid_json")

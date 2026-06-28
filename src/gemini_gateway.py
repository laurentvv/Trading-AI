"""Google Gemini gateway — the priority cloud backend in the 3-tier chain.

Fallback chain (per ``AGENTS.md`` §6.1 and the approved integration plan)::

    Gemini (GEMINI_API_KEY)  →  free-llm-api-keys  →  local Ollama

This module is the **Gemini** tier. Each public method returns either a result
or ``None``; ``None`` means "I could not serve this request — try the next
backend". The caller (``llm_client.py``) decides what the next backend is.
No exception ever escapes this module into the trading pipeline.

SDK note (June 2026)
--------------------
Uses the **unified ``google-genai`` SDK** (``from google import genai``), the
official successor to the deprecated ``google-generativeai`` package. The new
SDK is client-based: ``genai.Client(api_key=...)`` then
``client.models.generate_content(model=..., contents=..., config=...)``.

Model dispatch
--------------
Gemini 2.5 Pro returns a 0/0/0 quota on this key (unavailable), so the
reasoning tier runs on a Flash model. Dispatch follows the account's actual
Free Tier limits (confirmed empirically, not from public docs):

    - ``gemini-3.5-flash``       (5 RPM / 20 RPD)  → final trading decision
    - ``gemini-2.5-flash``       (5 RPM / 20 RPD)  → chart image analysis (vision)
    - ``gemini-3.1-flash-lite`` (15 RPM / 500 RPD) → web context summarization

Design notes
------------
* **Graceful degradation.** ``google-genai`` may be absent (pre-``uv sync``).
  The SDK is imported lazily and a missing package simply disables the gateway
  (``self.enabled = False``) instead of raising at import time.
* **Quota pre-flight.** Every call is gated by :class:`QuotaTracker` so we
  short-circuit *before* hitting the API once the local RPM/RPD ledger is full.
* **Invariant §2.1 (Dual-Layer JSON Defence).** Gemini has no ``<|think|>``
  channel, so structured output (``response_schema``) is the load-bearing JSON
  layer here. We still (a) append the ``"...never add a 'thought' key."``
  suffix for parity with the Ollama call sites, and (b) re-parse the response
  through ``_extract_json_candidates`` + ``_find_dict_with_keys`` (belt-and-
  braces) because Gemini occasionally wraps JSON in markdown fences or prefixes
  it with prose — a bug observed live and regression-tested.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.gemini_quota import QuotaTracker

logger = logging.getLogger(__name__)

load_dotenv()

# --- Lazy, fault-tolerant SDK import -------------------------------------- #
# The unified SDK (``google-genai``). If the wheel is absent (pre-``uv sync``)
# we degrade to ``enabled=False`` rather than crash at import time.
try:
    from google import genai
    from google.genai import types as genai_types  # noqa: F401  (used in _generate)
    from google.api_core.exceptions import ResourceExhausted
    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the wheel
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    ResourceExhausted = Exception  # type: ignore[misc,assignment]
    _SDK_AVAILABLE = False


# Free-Tier model identifiers (confirmed available to this key via models.list).
# Each use-case is an ORDERED CASCADE: tried first-to-last. When a model hits
# its local quota (RPM/RPD), returns 429, errors out, or is overloaded (503),
# the gateway transparently falls through to the next model in the chain.
# Quota is additive across the cascade — see gemini_quota.LIMITS.
#
# Ordering principle: most capable / newest first, deepest quota as the safety
# net. Gemma 4 (text-only) anchors the reasoning cascade so that even if every
# Flash model is exhausted, decisions still come from the cloud rather than
# falling all the way to local Ollama.

# Final trading decision (text reasoning). Adds up to ~2560 RPD before any
# fallback to free-llm-api-keys / Ollama.
REASONING_CASCADE = (
    "gemini-3.5-flash",        # newest Flash; small but scarce budget (20/d)
    "gemini-3-flash-preview",  # near-equivalent; (20/d)
    "gemini-3.1-flash-lite",   # lighter but generous (500/d)
    "gemini-2.5-flash",        # stable fallback (20/d)
    "gemma-4-31b",             # text-only safety net (1500/d)
    "gemma-4-26b",             # final text-only safety net (1500/d)
)

# Chart image analysis (vision). Gemma excluded — not image-capable via API.
VISION_CASCADE = (
    "gemini-3.5-flash",        # newest, vision-capable
    "gemini-3-flash-preview",  # vision-capable
    "gemini-2.5-flash",        # stable, vision-capable (validated live)
    "gemini-2.0-flash",        # oldest stable vision fallback
)

# Web context summarization (text, light). flash-lite anchors it (500/d).
WEB_SUMMARY_CASCADE = (
    "gemini-3.1-flash-lite",   # primary — huge RPD budget
    "gemini-2.5-flash-lite",   # fallback
    "gemini-2.0-flash-lite",   # final fallback
)

# JSON schema enforced server-side via ``response_schema``. This mirrors the
# Ollama ``SCHEMA_TRADING_DECISION`` (signal enum + confidence + analysis) so a
# Gemini decision is a drop-in replacement for an Ollama decision downstream.
_TRADING_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "confidence": {"type": "number"},
        "analysis": {"type": "string"},
    },
    "required": ["signal", "confidence", "analysis"],
}

# Parity suffix with the Ollama call sites (AGENTS.md §2.1, Layer 2).
_NO_THOUGHT_SUFFIX = "Output ONLY the JSON object requested — never add a 'thought' key."

# Prompt for the chart analyst. Verbatim copy of the Ollama visual prompt so the
# two backends reason from identical instructions (decision parity).
_CHART_ANALYST_PROMPT = """
ACT AS A PROFESSIONAL CHART ANALYST. Analyze the attached price chart image.
1. Patterns: Identify visible geometric patterns (Head & Shoulders, Triangles, Channels).
2. Price Action: Note the recent candle behavior (rejection, momentum, gaps).
3. Indicators: Look at the visual shape of indicators (RSI divergences, MACD crossovers).

IMPORTANT: Your role is purely geometric and visual validation.
- If the chart is ambiguous, mixed, or mostly sideways, you MUST output "HOLD" with low confidence (< 0.5).
- Reserve "BUY" or "SELL" with high confidence (> 0.7) ONLY for textbook, unmistakable patterns.
"""

# System instruction used for the heavy reasoning model (final decision).
_FINANCIAL_ANALYST_SYSTEM = (
    "You are an expert financial analyst. Your task is to analyze market data "
    "and news to provide a trading decision in a valid JSON format."
)


class GeminiGateway:
    """Priority cloud backend. Returns results or ``None``; never raises.

    The gateway holds one ``genai.Client`` (cheap, reusable) and a
    :class:`QuotaTracker`. Configuration is read once from the environment:

    * ``GEMINI_API_KEY`` — the Free Tier key. Absent ⇒ ``enabled=False``.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.enabled = bool(_SDK_AVAILABLE and self.api_key)
        self._client = None
        if not _SDK_AVAILABLE:
            logger.debug("Gemini disabled: google-genai not installed.")
        elif not self.api_key:
            logger.debug("Gemini disabled: GEMINI_API_KEY not set.")
        if self.enabled:
            try:
                # The new SDK is client-based: one Client per process/key.
                self._client = genai.Client(api_key=self.api_key)  # type: ignore[union-attr]
            except Exception as e:  # noqa: BLE001 - bad key must not crash the pipeline
                logger.warning("Gemini Client init failed (%s). Gateway disabled.", e)
                self.enabled = False
        self._quota = QuotaTracker()

    # ------------------------------------------------------------------ #
    # Internal call wrapper — the single place that talks to the SDK.
    # ------------------------------------------------------------------ #
    def _generate_one(
        self,
        model_name: str,
        contents: Any,
        *,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str,
        json_schema: Optional[dict] = None,
    ) -> Optional[str]:
        """Run ONE Gemini call against ``model_name``.

        Returns the raw text response, or ``None`` on quota exhaustion, 429,
        or any SDK/network failure. ``None`` is the contract that lets the
        caller try the next model in a cascade. Only a successful call
        consumes quota (recorded here).
        """
        if not self.enabled or self._client is None:
            return None

        # Pre-flight: skip this model if its local quota ledger is full.
        if not self._quota.check(model_name):
            return None

        try:
            config_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "system_instruction": system_instruction,
            }
            if json_schema is not None:
                # Structured output — the load-bearing JSON layer for Gemini.
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = json_schema
            config = genai_types.GenerateContentConfig(**config_kwargs)  # type: ignore[union-attr]

            response = self._client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            text = response.text
        except ResourceExhausted:
            logger.info(
                "Gemini %s: quota exhausted (429). Trying next model.", model_name
            )
            return None
        except Exception as e:  # noqa: BLE001 - try next model on any failure
            logger.info(
                "Gemini %s: call failed (%s). Trying next model.", model_name, e
            )
            return None

        if not text or not text.strip():
            logger.info("Gemini %s: empty response. Trying next model.", model_name)
            return None

        # Only a *successful* call consumes quota.
        self._quota.record(model_name)
        logger.debug("Gemini %s: success.", model_name)
        return text

    def _generate_cascaded(
        self,
        cascade: tuple,
        contents: Any,
        *,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str,
        json_schema: Optional[dict] = None,
    ) -> Optional[str]:
        """Try each model in ``cascade`` in order until one succeeds.

        This is the multi-model failover: quota (RPM/RPD), 429, 503 overload,
        and generic errors on one model all cause a transparent fall-through
        to the next. Returns ``None`` only if every model in the cascade is
        exhausted — at which point the outer caller falls back to free-llm /
        Ollama.
        """
        if not self.enabled:
            return None
        for model_name in cascade:
            text = self._generate_one(
                model_name,
                contents,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
                json_schema=json_schema,
            )
            if text is not None:
                return text
        logger.warning(
            "Gemini cascade exhausted (%d models tried, all failed/over-quota). "
            "Falling back to next backend.", len(cascade),
        )
        return None

    # ------------------------------------------------------------------ #
    # Public use-case methods
    # ------------------------------------------------------------------ #
    def summarize_web_context(self, text_content: str) -> Optional[str]:
        """Synthesize raw crawled web markdown into a trading-focused summary.

        Uses ``flash-lite`` (light, high RPD budget). Output is free-form text
        (no JSON schema) — this is a *summarizer*, not a decision-maker. The
        returned summary replaces the raw markdown fed into the decision
        prompt, reducing context noise.
        """
        if not self.enabled or not text_content or not text_content.strip():
            return None
        prompt = (
            "You are a macro news analyst for a trading desk. Read the web "
            "research below and produce a tight, fact-dense summary (max ~250 "
            "words) of the developments most likely to move markets: central "
            "bank action, macro data, earnings, commodities, geopolitics. Drop "
            "filler and repeats. Do NOT invent facts.\n\n"
            f"--- WEB RESEARCH ---\n{text_content}\n--- END ---"
        )
        return self._generate_cascaded(
            WEB_SUMMARY_CASCADE,
            prompt,
            temperature=0.2,
            max_output_tokens=700,
            system_instruction=(
                "You are a concise financial news summarizer for a trading system."
            ),
            json_schema=None,
        )

    def analyze_chart(self, image_path: Path) -> Optional[dict]:
        """Geometric chart analysis via ``flash`` (vision-capable).

        Returns a dict ``{signal, confidence, analysis}`` or ``None``. The
        prompt is a verbatim copy of the Ollama visual prompt so both
        backends share identical instructions.
        """
        if not self.enabled or not image_path.exists():
            return None
        try:
            # The new SDK accepts raw image bytes via Part.from_bytes — no PIL
            # round-trip needed, and no extra dependency at the call site.
            image_bytes = image_path.read_bytes()
            contents = [
                _CHART_ANALYST_PROMPT.strip(),
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),  # type: ignore[union-attr]
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning("Gemini vision could not load %s (%s).", image_path, e)
            return None

        text = self._generate_cascaded(
            VISION_CASCADE,
            contents,
            temperature=0.4,
            max_output_tokens=1024,
            system_instruction=(
                "You are an objective geometric chart analyst. "
                + _NO_THOUGHT_SUFFIX
            ),
            json_schema=_TRADING_DECISION_SCHEMA,
        )
        return self._parse_decision(text)

    def decide(self, prompt: str) -> Optional[dict]:
        """Final trading decision via ``gemini-3.5-flash`` (reasoning tier).

        Returns a dict ``{signal, confidence, analysis}`` or ``None``. The
        caller builds ``prompt`` via ``construct_llm_prompt`` (the same prompt
        the Ollama path uses), so the two paths reason over identical input.
        """
        if not self.enabled or not prompt or not prompt.strip():
            return None
        text = self._generate_cascaded(
            REASONING_CASCADE,
            prompt,
            temperature=0.4,
            max_output_tokens=1024,
            system_instruction=_FINANCIAL_ANALYST_SYSTEM + " " + _NO_THOUGHT_SUFFIX,
            json_schema=_TRADING_DECISION_SCHEMA,
        )
        return self._parse_decision(text)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_decision(text: Optional[str]) -> Optional[dict]:
        """Best-effort JSON parse of a structured Gemini response.

        Gemini's ``response_schema`` should already guarantee valid JSON, but
        the model sometimes wraps the object in markdown fences (````` ```json `````)
        or prefixes it with a short prose remark. We therefore reuse the
        production-grade extractor from ``llm_client`` (the same one that
        scrubs ``<|think|>`` debris and drills into nested structures for the
        Gemma path) rather than a naive ``json.loads``. Returns ``None`` if no
        usable object is found — the caller then falls back to the next backend.
        """
        if not text:
            return None

        # Fast path: the common case where Gemini returns bare JSON.
        try:
            import json
            data = json.loads(text)
            if isinstance(data, dict):
                return GeminiGateway._normalize_decision(data)
        except (ValueError, TypeError):
            pass

        # Robust path: handles ```json fences, leading prose, and nested dicts.
        # Lazy import to avoid a circular dependency at module load time
        # (llm_client imports nothing from this module, but we are imported by
        # it at call time, so deferring keeps the import graph acyclic).
        try:
            from src.llm_client import _extract_json_candidates, _find_dict_with_keys

            for candidate in _extract_json_candidates(text):
                parsed = _find_dict_with_keys(candidate, ["signal", "confidence", "analysis"])
                if parsed is not None:
                    return GeminiGateway._normalize_decision(parsed)
        except Exception:  # noqa: BLE001 - last-resort fallback, never raise
            pass

        logger.warning("Gemini JSON parse failed. Falling back. Head: %r", text[:160])
        return None

    @staticmethod
    def _normalize_decision(data: dict) -> dict:
        """Coerce a parsed decision dict into a stable, typed shape."""
        return {
            "signal": str(data.get("signal", "HOLD")).upper(),
            "confidence": float(data.get("confidence", 0.0)),
            "analysis": str(data.get("analysis", "")),
        }

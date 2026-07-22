"""Google Gemini gateway — priority cloud backend with two-tier keys.

Fallback chain (per ``AGENTS.md`` §6.1)::

    Gemini (paid key)  →  Gemini (free key)  →  free-llm-api-keys  →  local Ollama

This module is the **Gemini** portion. Each public method returns either a
result or ``None``; ``None`` means "I could not serve this request — try the
next backend". The caller (``llm_client.py``) decides what the next backend
is. No exception ever escapes this module into the trading pipeline.

Two API keys, two tiers
-----------------------
* ``GEMINI_API_KEY_PAY`` (paid Tier 1) — used for the high-value use-cases
  (decision, vision). Generous quotas; **Gemini 2.5 Pro** is available here
  and anchors the reasoning cascade. Bounded by a rolling 30-day cost budget
  (``GEMINI_PAY_MONTHLY_BUDGET_EUR``, the load-bearing billing guard) plus a
  daily call-cap backstop (``GEMINI_PAY_DAILY_CAP``, default 200).
* ``GEMINI_API_KEY`` (free Tier) — used for the low-value web-summary use-case,
  and as the fallback when the paid tier is exhausted. Pro is unavailable
  here (0/0/0); only Flash/Lite/Gemma models.

Use-case → tier routing
-----------------------
* ``decide()``          → PAID cascade (Pro → 3.5-flash → ...) → FREE cascade
* ``analyze_chart()``   → PAID cascade (2.5-flash → ...) → FREE cascade
* ``summarize_web_context()`` → FREE cascade only (trivial task)

When the paid tier's daily cap is reached (or every paid model 429s), the
decision/vision use-cases transparently fall back to the FREE cascade before
yielding ``None`` (which then falls through to free-llm / Ollama).

SDK note (June 2026)
--------------------
Uses the unified ``google-genai`` SDK (``from google import genai``), the
official successor to the deprecated ``google-generativeai`` package.

Design notes
------------
* **Graceful degradation.`` ``google-genai`` may be absent (pre-``uv sync``).
  The SDK is imported lazily and a missing package simply disables the
  gateway instead of raising at import time.
* **Quota pre-flight.** Every call is gated by :class:`QuotaTracker` so we
  short-circuit *before* hitting the API once the local ledger is full.
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
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from src.gemini_quota import (
    LIMITS_FREE,
    LIMITS_PAID,
    QuotaTracker,
    TIER_FREE,
    TIER_PAID,
    compute_call_cost_eur,
)

logger = logging.getLogger(__name__)

load_dotenv()

# --- Lazy, fault-tolerant SDK import -------------------------------------- #
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


# ========================================================================= #
# Cascades (ordered; most capable first, deepest quota as the safety net)
# ========================================================================= #

# --- PAID Tier 1 cascades -------------------------------------------------
# Decision: Gemini 2.5 Pro anchors it (only available on the paid key).
REASONING_CASCADE_PAID = (
    "gemini-2.5-pro",          # Pro: best reasoning, paid-only
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",  # 3.1 Pro: strong reasoning (preview suffix required)
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",   # huge RPD budget
    "gemini-2.5-flash",
    "gemma-4-31b-it",          # text-only safety net (-it suffix required)
    "gemma-4-26b-a4b-it",
)

# Vision: multimodal only (Gemma excluded).
VISION_CASCADE_PAID = (
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.0-flash",
)

# --- FREE Tier cascades (fallback) ---------------------------------------
# Pro is 0/0/0 here, so it's absent. Used when the paid cap is exhausted.
REASONING_CASCADE_FREE = (
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
)

VISION_CASCADE_FREE = (
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

# --- Web summary cascade (FREE only — trivial task, preserves paid budget) -
WEB_SUMMARY_CASCADE_FREE = (
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
)

# --- Weekend Council cascades (prose mode, no JSON schema) ----------------
# The council emits free-text analyses, not structured decisions. The Sceptique
# and Comportementaliste (the two members that previously shared a weak 1.2B
# LFM model and produced a broken "SELL|HOLD|BUY" placeholder) run on the FREE
# cascade — they fire many times per run, so the free tier preserves billing.
COUNCIL_MEMBER_CASCADE_FREE = (
    "gemini-2.5-flash",        # primary: fast, strong instruction-following
    "gemini-3.1-flash-lite",   # huge RPD budget, decent reasoning
    "gemini-2.5-flash-lite",
    "gemma-4-26b-a4b-it",      # text-only safety net (-a4b-it suffix required)
)
# The Judge fires ONCE per run but performs the hardest job (synthesising the
# full transcript into a structured verdict). It runs on the PAID cascade so
# Gemini 2.5 Pro can anchor it — but the QuotaTracker daily cap still protects
# billing (see DEFAULT_PAID_DAILY_CAP).
COUNCIL_JUDGE_CASCADE_PAID = (
    "gemini-2.5-pro",          # best synthesis, paid-only
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",  # (preview suffix required)
    "gemini-2.5-flash",        # safety net within paid tier
)
COUNCIL_JUDGE_CASCADE_FREE = (
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemma-4-26b-a4b-it",
)

# JSON schema enforced server-side via ``response_schema``. Mirrors the Ollama
# ``SCHEMA_TRADING_DECISION`` so a Gemini decision is a drop-in replacement.
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

_CHART_ANALYST_PROMPT = """
ACT AS A PROFESSIONAL CHART ANALYST. Analyze the attached price chart image.
1. Patterns: Identify visible geometric patterns (Head & Shoulders, Triangles, Channels).
2. Price Action: Note the recent candle behavior (rejection, momentum, gaps).
3. Indicators: Look at the visual shape of indicators (RSI divergences, MACD crossovers).

IMPORTANT: Your role is purely geometric and visual validation.
- If the chart is ambiguous, mixed, or mostly sideways, you MUST output "HOLD" with low confidence (< 0.5).
- Reserve "BUY" or "SELL" with high confidence (> 0.7) ONLY for textbook, unmistakable patterns.
"""

_FINANCIAL_ANALYST_SYSTEM = (
    "You are an expert financial analyst. Your task is to analyze market data "
    "and news to provide a trading decision in a valid JSON format."
)


class _TierBackend:
    """One Gemini client + its quota tier tag.

    Wraps a ``genai.Client`` for one key (paid or free) and routes quota
    accounting to the matching ledger in :class:`QuotaTracker`.
    """

    def __init__(self, client: Any, tier: str) -> None:
        self.client = client
        self.tier = tier

    def generate(
        self,
        model_name: str,
        contents: Any,
        *,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str,
        json_schema: Optional[dict] = None,
        quota: QuotaTracker,
    ) -> Optional[str]:
        """Run ONE call. Returns text or ``None`` (quota/429/error/empty)."""
        if not quota.check(model_name, self.tier):
            return None
        try:
            config_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "system_instruction": system_instruction,
            }
            if json_schema is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = json_schema
            config = genai_types.GenerateContentConfig(**config_kwargs)  # type: ignore[union-attr]
            response = self.client.models.generate_content(
                model=model_name, contents=contents, config=config,
            )
            text = response.text
        except ResourceExhausted:
            logger.info("Gemini %s/%s: 429 quota. Trying next.", self.tier, model_name)
            return None
        except Exception as e:  # noqa: BLE001 - try next model
            logger.info("Gemini %s/%s: failed (%s). Trying next.", self.tier, model_name, e)
            return None

        if not text or not text.strip():
            # Distinguish a true empty response from a token-limit truncation.
            # Pro is verbose under JSON schema; a MAX_TOKENS finish yields a
            # partial (unparseable) or empty body. Log the cause so it is not
            # mistaken for a safety filter.
            reason = GeminiGateway._finish_reason(response)
            logger.info(
                "Gemini %s/%s: empty response (finish_reason=%s). Trying next.",
                self.tier, model_name, reason,
            )
            return None
        # Bill the call against the ledger. Cost is computed from the actual
        # token usage reported by the API — but only on the paid tier (the
        # free tier is unmetered, so we skip the computation and record 0).
        cost_eur = (
            compute_call_cost_eur(
                model_name, GeminiGateway._usage_metadata(response)
            )
            if self.tier == TIER_PAID else 0.0
        )
        quota.record(model_name, self.tier, cost_eur=cost_eur)
        logger.debug(
            "Gemini %s/%s: success (cost %.5f EUR).", self.tier, model_name, cost_eur,
        )
        return text


class GeminiGateway:
    """Two-tier Gemini gateway. Returns results or ``None``; never raises.

    ``enabled`` is True if EITHER key is configured (so the gateway is useful
    even with only the free key). ``enabled_paid`` / ``enabled_free`` expose
    per-key availability for the routing logic.
    """

    def __init__(self) -> None:
        paid_key = os.getenv("GEMINI_API_KEY_PAY", "").strip()
        free_key = os.getenv("GEMINI_API_KEY", "").strip()

        self._quota = QuotaTracker()
        self._paid: Optional[_TierBackend] = self._make_backend(paid_key, TIER_PAID)
        self._free: Optional[_TierBackend] = self._make_backend(free_key, TIER_FREE)

        self.enabled_paid = self._paid is not None
        self.enabled_free = self._free is not None
        self.enabled = self.enabled_paid or self.enabled_free
        if not _SDK_AVAILABLE:
            logger.debug("Gemini disabled: google-genai not installed.")
            self.enabled = self.enabled_paid = self.enabled_free = False

    @staticmethod
    def _make_backend(api_key: str, tier: str) -> Optional[_TierBackend]:
        if not _SDK_AVAILABLE or not api_key:
            return None
        try:
            client = genai.Client(api_key=api_key)  # type: ignore[union-attr]
        except Exception as e:  # noqa: BLE001 - bad key must not crash pipeline
            logger.warning("Gemini %s Client init failed (%s). Tier disabled.", tier, e)
            return None
        return _TierBackend(client, tier)

    # ------------------------------------------------------------------ #
    # Cascade runner: try a tier's cascade, return (text, tier, model).
    # ------------------------------------------------------------------ #
    def _run_cascade(
        self,
        backend: Optional[_TierBackend],
        cascade: tuple,
        contents: Any,
        **gen_kwargs,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Try each model in ``cascade`` via ``backend``.

        Returns ``(text, model_name)`` on success, or ``(None, None)`` if the
        backend is missing or every model failed.
        """
        if backend is None:
            return None, None
        for model_name in cascade:
            text = backend.generate(
                model_name, contents, quota=self._quota, **gen_kwargs
            )
            if text is not None:
                return text, model_name
        return None, None

    def _run_tiered(
        self,
        contents: Any,
        paid_cascade: tuple,
        free_cascade: tuple,
        use_free_only: bool = False,
        **gen_kwargs,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Run paid cascade first, then free cascade as fallback.

        Returns ``(text, tier, model)`` — ``tier`` is ``"paid"`` / ``"free"`` /
        ``None``. When ``use_free_only`` is set (web summary), the paid tier is
        skipped entirely to preserve billing budget.
        """
        if not use_free_only and self._paid is not None:
            text, model = self._run_cascade(self._paid, paid_cascade, contents, **gen_kwargs)
            if text is not None:
                return text, TIER_PAID, model
            logger.info("Gemini paid tier exhausted. Trying free tier.")
        if self._free is not None:
            text, model = self._run_cascade(self._free, free_cascade, contents, **gen_kwargs)
            if text is not None:
                return text, TIER_FREE, model
        logger.warning(
            "Gemini tiered cascade exhausted (paid+free). Falling back to next backend."
        )
        return None, None, None

    # ------------------------------------------------------------------ #
    # Public use-case methods
    # ------------------------------------------------------------------ #
    def summarize_web_context(self, text_content: str) -> Optional[str]:
        """Synthesize raw crawled markdown into a trading-focused summary.

        FREE tier only (trivial task — preserves paid budget). Returns the
        summary text or ``None``.
        """
        if not self.enabled_free or not text_content or not text_content.strip():
            return None
        prompt = (
            "You are a macro news analyst for a trading desk. Read the web "
            "research below and produce a tight, fact-dense summary (max ~250 "
            "words) of the developments most likely to move markets: central "
            "bank action, macro data, earnings, commodities, geopolitics. Drop "
            "filler and repeats. Do NOT invent facts.\n\n"
            f"--- WEB RESEARCH ---\n{text_content}\n--- END ---"
        )
        text, _tier, _model = self._run_tiered(
            prompt,
            paid_cascade=WEB_SUMMARY_CASCADE_FREE,  # unused (free-only)
            free_cascade=WEB_SUMMARY_CASCADE_FREE,
            use_free_only=True,
            temperature=0.2,
            max_output_tokens=700,
            system_instruction="You are a concise financial news summarizer for a trading system.",
            json_schema=None,
        )
        return text

    def analyze_chart(self, image_path: Path) -> Optional[dict]:
        """Geometric chart analysis. PAID cascade → FREE cascade.

        Returns a dict ``{signal, confidence, analysis}`` or ``None``.
        """
        if not self.enabled or not image_path.exists():
            return None
        try:
            image_bytes = image_path.read_bytes()
            contents = [
                _CHART_ANALYST_PROMPT.strip(),
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),  # type: ignore[union-attr]
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning("Gemini vision could not load %s (%s).", image_path, e)
            return None

        text, _tier, _model = self._run_tiered(
            contents,
            paid_cascade=VISION_CASCADE_PAID,
            free_cascade=VISION_CASCADE_FREE,
            temperature=0.4,
            max_output_tokens=1024,
            system_instruction="You are an objective geometric chart analyst. " + _NO_THOUGHT_SUFFIX,
            json_schema=_TRADING_DECISION_SCHEMA,
        )
        return self._parse_decision(text)

    def decide(self, prompt: str) -> Optional[dict]:
        """Final trading decision. PAID cascade (Pro-led) → FREE cascade.

        Returns a dict ``{signal, confidence, analysis}`` or ``None``.
        """
        if not self.enabled or not prompt or not prompt.strip():
            return None
        text, _tier, _model = self._run_tiered(
            prompt,
            paid_cascade=REASONING_CASCADE_PAID,
            free_cascade=REASONING_CASCADE_FREE,
            temperature=0.4,
            # Reasoning models (esp. 2.5 Pro) are verbose under JSON schema
            # enforcement — 1024 truncated them (finish_reason=MAX_TOKENS →
            # empty/invalid JSON). 8192 is ample for a structured decision.
            max_output_tokens=8192,
            system_instruction=_FINANCIAL_ANALYST_SYSTEM + " " + _NO_THOUGHT_SUFFIX,
            json_schema=_TRADING_DECISION_SCHEMA,
        )
        return self._parse_decision(text)

    def deliberate(
        self,
        system_instruction: str,
        user_prompt: str,
        *,
        use_paid: bool = False,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
    ) -> Optional[str]:
        """Free-prose deliberation for the Weekend Council members and Judge.

        Unlike :meth:`decide` (structured JSON trading decision), this emits
        free text — council members write analyses, critiques, and a structured
        (but non-JSON) verdict. The caller (``weekend_council.ask_llm``) decides
        the tier: members fire many times per run so they use the FREE cascade
        (preserves billing); the Judge fires once but does the hardest job, so
        it uses the PAID cascade (Gemini 2.5 Pro anchors it).

        Returns the prose text, or ``None`` if every model in both cascades
        failed (the caller then falls back to local Ollama).

        No ``response_schema`` is enforced — the council's STANCE/VERDICT
        parsing is intentionally tolerant of prose variation (see
        ``weekend_council._parse_stance``).
        """
        if not self.enabled or not user_prompt or not user_prompt.strip():
            return None
        if use_paid:
            paid_cascade = COUNCIL_JUDGE_CASCADE_PAID
            free_cascade = COUNCIL_JUDGE_CASCADE_FREE
        else:
            paid_cascade = COUNCIL_MEMBER_CASCADE_FREE  # unused (free-only)
            free_cascade = COUNCIL_MEMBER_CASCADE_FREE
        # Members run free-only (use_free_only=True) to preserve the paid budget
        # for the Judge and the real-time decision/vision calls.
        text, _tier, _model = self._run_tiered(
            user_prompt,
            paid_cascade=paid_cascade,
            free_cascade=free_cascade,
            use_free_only=not use_paid,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            json_schema=None,
        )
        return text

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _finish_reason(response: Any) -> str:
        """Extract the finish_reason from a Gemini response, defensively.

        Returns a short string (``STOP``, ``MAX_TOKENS``, ``SAFETY``,
        ``OTHER``, ``?``) for logging. Never raises.
        """
        try:
            cands = getattr(response, "candidates", None)
            if not cands:
                return "no-candidates"
            reason = getattr(cands[0], "finish_reason", None)
            if reason is None:
                return "?"
            # The SDK enum stringifies to e.g. "FinishReason.MAX_TOKENS".
            name = getattr(reason, "name", str(reason))
            return name.replace("FinishReason.", "")
        except Exception:  # noqa: BLE001 - logging helper must never raise
            return "?"

    @staticmethod
    def _usage_metadata(response: Any) -> Optional[dict]:
        """Extract ``usage_metadata`` from a Gemini response as a plain dict.

        The SDK returns a protobuf-like object with ``prompt_token_count`` /
        ``candidates_token_count`` / ``total_token_count``. We coerce it to a
        plain dict for :func:`compute_call_cost_eur`. Returns ``None`` if the
        field is absent (older SDK, safety filter, etc.) — the cost helper
        then falls back to its conservative flat estimate. Never raises.
        """
        try:
            um = getattr(response, "usage_metadata", None)
            if um is None:
                return None
            # ``usage_metadata`` may be a proto message (use ``_unknown_class``
            # / attribute access) or already a dict. Handle both.
            if isinstance(um, dict):
                return um
            return {
                k: getattr(um, k)
                for k in ("prompt_token_count", "candidates_token_count", "total_token_count")
                if hasattr(um, k)
            }
        except Exception:  # noqa: BLE001 - cost helper must never raise
            return None

    @staticmethod
    def _parse_decision(text: Optional[str]) -> Optional[dict]:
        """Best-effort JSON parse of a structured Gemini response.

        Three strategies, tried in order:
        1. **Fast path** — ``json.loads`` (the common case: bare, valid JSON).
        2. **Robust path** — ``_extract_json_candidates`` + ``_find_dict_with_keys``
           from ``llm_client`` (handles ``\\`\\`\\`json`` fences, prose prefixes,
           and nested dicts — the production-grade path shared with Gemma).
        3. **Regex salvage** — if Gemini emitted a structurally broken object
           (e.g. unescaped quotes inside ``analysis``), extract ``signal`` and
           ``confidence`` directly. This trades the (often-broken) analysis
           text for a usable decision rather than discarding the whole call.

        Returns ``None`` only if even the regex salvage finds nothing usable.
        """
        if not text:
            return None

        # 1. Fast path.
        try:
            import json
            data = json.loads(text)
            if isinstance(data, dict):
                return GeminiGateway._normalize_decision(data)
        except (ValueError, TypeError):
            pass

        # 2. Robust path (markdown fences, prose prefix, nested dicts).
        # Lazy import to keep the import graph acyclic.
        try:
            from src.llm_client import _extract_json_candidates, _find_dict_with_keys

            for candidate in _extract_json_candidates(text):
                parsed = _find_dict_with_keys(candidate, ["signal", "confidence", "analysis"])
                if parsed is not None:
                    return GeminiGateway._normalize_decision(parsed)
        except Exception:  # noqa: BLE001 - fall through to regex salvage
            pass

        # 3. Regex salvage: the object is malformed, but signal/confidence
        # usually sit at the start and are themselves well-formed.
        salvaged = GeminiGateway._regex_salvage(text)
        if salvaged is not None:
            logger.info(
                "Gemini JSON was malformed; salvaged signal/confidence via regex."
            )
            return salvaged

        logger.warning("Gemini JSON parse failed. Falling back. Head: %r", text[:160])
        return None

    @staticmethod
    def _regex_salvage(text: str) -> Optional[dict]:
        """Extract signal/confidence from a malformed JSON object via regex.

        Used when Gemini produces broken JSON (typically unescaped quotes in
        the ``analysis`` field). We pull ``signal`` and ``confidence`` which
        are simple scalars at the object's head, and use a placeholder
        analysis so the decision is still usable.
        """
        import re
        sig_match = re.search(r'"signal"\s*:\s*"(BUY|SELL|HOLD)"', text, re.IGNORECASE)
        conf_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if not sig_match:
            return None
        return GeminiGateway._normalize_decision({
            "signal": sig_match.group(1),
            "confidence": conf_match.group(1) if conf_match else "0.0",
            "analysis": "(analysis unavailable — JSON was malformed)",
        })

    @staticmethod
    def _normalize_decision(data: dict) -> dict:
        """Coerce a parsed decision dict into a stable, typed shape.

        ``confidence`` is clamped to [0, 1] — some models (notably Gemini 2.5
        Pro) occasionally emit a confidence on a 0-10 scale despite the schema;
        clamping guards the downstream weighted aggregation.
        """
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        # If a model emitted a 0-10 scale, rescale anything > 1 down to [0,1].
        if confidence > 1.0:
            confidence = confidence / 10.0 if confidence <= 10.0 else 1.0
        return {
            "signal": str(data.get("signal", "HOLD")).upper(),
            "confidence": max(0.0, min(1.0, confidence)),
            "analysis": str(data.get("analysis", "")),
        }

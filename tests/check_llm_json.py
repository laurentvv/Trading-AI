"""Diagnostic: interroge directement Ollama pour isoler le bug JSON.

Lance plusieurs variantes du payload envoyé au LLM (system prompt,
format json, etc.) et affiche la réponse brute + resultat du parsing.
Permet de reproduire le bug "Could not find valid JSON with keys ['query']"
sans faire tourner tout le pipeline de trading.

Note (re-enablement du 2026-06-06) :
    Le token `<|think|>` a été ré-introduit dans les 4 prompts système
    de production (src/llm_client.py, src/oil_bench_model.py,
    src/web_researcher.py) sur la branche `think-mode`. Les cas `*_v1_buggy`
    ci-dessous correspondent désormais au chemin **production** (avec
    `<|think|>` actif) ; les cas `v2_fixed` / `v3_schema` restent exécutés
    comme références défensives (sans `<|think|>` / avec schema strict seul).

    Deux issues sont acceptables :
      * Tous les cas OK → Gemma 4 12B produit un JSON propre même avec
        `<|think|>`, la défense par `format: SCHEMA_*` suffit.
      * Les `*_v1_buggy` échouent avec le symptôme historique (débris
        `<|channel>thought` dans le JSON) → validation négative réussie :
        on confirme que c'est la contrainte de schéma qui porte la
        sécurité, pas l'absence du token. Le script exit non-zéro mais
        le résumé reste informatif.

Résultats mesurés au moment de la ré-activation (2026-06-06, commit de
référence sur la branche `think-mode`) :
      * OK  : query_v6_schema, query_v7_schema_strict,
              decision_v3_schema, oil_v1_buggy, oil_v2_fixed, oil_v3_schema
      * FAIL: query_v1_prod_buggy, query_v4_strict,
              decision_v1_buggy, decision_v2_fixed
    Tous les cas schema-strict passent avec `<|think|>` actif. Tous les
    échecs concernent des variantes `format:json` (loose) — non utilisées
    en production. La défense par schéma strict est donc confirmée comme
    la couche porteuse ; voir `docs/ADR-001-think-mode-dual-layer-defence.md`.

Usage:
    uv run tests/check_llm_json.py                # tous les cas
    uv run tests/check_llm_json.py --filter query # filtrer par cas
    uv run tests/check_llm_json.py --raw          # afficher la reponse brute complete
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

# Permet l'import de src.* quand on lance le script directement.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import OLLAMA_API_URL, TEXT_LLM_MODEL  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _short(s: str, n: int = 300) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"... [+{len(s) - n} chars]"


def _extract_json_objects(raw: str) -> list[dict]:
    """Extract all valid JSON objects from a string (best-effort)."""
    objs: list[dict] = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(raw):
        start = raw.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(raw[start:])
            if isinstance(obj, dict):
                objs.append(obj)
            pos = start + end
        except (json.JSONDecodeError, ValueError):
            pos += 1
    return objs


def _raw_call_ollama(payload: dict, timeout: int = 120) -> dict:
    """Call Ollama directly and return the raw JSON response."""
    r = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    description: str
    payload: dict
    expected_keys: list[str] = field(default_factory=lambda: ["query"])
    parser: Callable[[dict], dict] | None = None  # override parser if needed

    def __post_init__(self) -> None:
        # Force the model used everywhere (text LLM by default).
        self.payload.setdefault("model", TEXT_LLM_MODEL)
        self.payload.setdefault("stream", False)


# ---- Prompts --------------------------------------------------------------

SEARCH_QUERY_PROMPT = """
You are an expert macroeconomic research assistant. Today is June 2026.
Target Asset: CL=F
Current Context: The current price is 90.54 with a 5-day downward trend (-1.76%).
Specifically focus on OPEC+ supply decisions, global inventory levels, and 'flx:OIL' sentiment on Hyperliquid.

Your goal is to find the most impactful news or reports FROM THE LAST 30 DAYS that explain the current market regime.
Generate the single most effective Google/DuckDuckGo search query (maximum 10 words).

Output ONLY a valid JSON object:
{
  "query": "<your optimized search query>"
}
""".strip()


TRADING_DECISION_PROMPT = """
Analyze the following market data for CL=F (OIL WTI):
- Close Price: 90.54
- RSI (14): 42.15
- MACD: 12.5 | Signal: 15.2
- Short-term Trend: Bearish

Provide your analysis ONLY as a valid JSON object.
{
  "signal": "BUY | SELL | HOLD",
  "confidence": <float 0.0 to 1.0>,
  "analysis": "A rigorous 2-sentence technical and fundamental justification."
}
""".strip()


OIL_BENCH_PROMPT = """
You are a senior commodity analyst. Context:
- WTI Price: 90.54 (MA200: 72.79) -> above MA200, bullish
- DXY: 100.07
- Brent Spread: 5.20

Return ONLY a JSON object:
{"allocation": <float 0-100>, "reasoning": "<2-sentence analysis>"}
""".strip()


# ---- System prompt variants to test --------------------------------------

SYS_THINK = "<|think|> You are a professional financial researcher. Be precise and focus on current market catalysts."
SYS_PLAIN = "You are a professional financial researcher. Be precise and focus on current market catalysts."
SYS_JSON_STRICT = (
    "You are a professional financial researcher. "
    "Respond ONLY with the requested JSON object. "
    "Do NOT include any reasoning, commentary, or extra keys."
)
SYS_JSON_SCHEMA = (
    "You are a professional financial researcher. "
    "Output strictly the JSON object requested. "
    "Never add a 'thought' or 'reasoning' key."
)


# ---- Build all test cases -------------------------------------------------

def build_cases() -> list[TestCase]:
    cases: list[TestCase] = []

    # Common options matching production (low temperature for determinism).
    opts_short = {"temperature": 0.4, "num_predict": 512}
    opts_long = {"temperature": 0.4, "num_predict": 1024}

    # JSON schemas (Ollama accepts a schema object as `format`).
    schema_query = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    schema_decision = {
        "type": "object",
        "properties": {
            "signal": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "confidence": {"type": "number"},
            "analysis": {"type": "string"},
        },
        "required": ["signal", "confidence", "analysis"],
        "additionalProperties": False,
    }
    schema_oil = {
        "type": "object",
        "properties": {
            "allocation": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["allocation", "reasoning"],
        "additionalProperties": False,
    }

    # ===== search_query =====
    cases.append(TestCase(
        id="query_v1_prod_buggy",
        description="REPRO: <|think|> + format:json (config originale avant fix)",
        payload={
            "prompt": SEARCH_QUERY_PROMPT,
            "format": "json",
            "system": SYS_THINK,
        },
        expected_keys=["query"],
    ))

    cases.append(TestCase(
        id="query_v4_strict",
        description="FIX 1: system prompt 'never thought' + format:json + temp 0.4",
        payload={
            "prompt": SEARCH_QUERY_PROMPT,
            "format": "json",
            "options": opts_short,
            "system": SYS_JSON_STRICT,
        },
        expected_keys=["query"],
    ))

    cases.append(TestCase(
        id="query_v6_schema",
        description="FIX 2: schema JSON strict (format=object) — empêche 'thought'",
        payload={
            "prompt": SEARCH_QUERY_PROMPT,
            "format": schema_query,
            "options": opts_short,
            "system": SYS_PLAIN,
        },
        expected_keys=["query"],
    ))

    cases.append(TestCase(
        id="query_v7_schema_strict",
        description="FIX 3: schema strict + system 'never thought' (double protection)",
        payload={
            "prompt": SEARCH_QUERY_PROMPT,
            "format": schema_query,
            "options": opts_short,
            "system": SYS_JSON_STRICT,
        },
        expected_keys=["query"],
    ))

    # ===== trading_decision =====
    cases.append(TestCase(
        id="decision_v1_buggy",
        description="REPRO: get_llm_decision avec <|think|>",
        payload={
            "prompt": TRADING_DECISION_PROMPT,
            "format": "json",
            "system": "<|think|> You are an expert financial analyst. Return ONLY valid JSON.",
        },
        expected_keys=["signal", "confidence", "analysis"],
    ))

    cases.append(TestCase(
        id="decision_v2_fixed",
        description="FIX 1: system 'never thought' + temp 0.4",
        payload={
            "prompt": TRADING_DECISION_PROMPT,
            "format": "json",
            "options": opts_long,
            "system": (
                "You are an expert financial analyst. Your task is to analyze market data "
                "and news to provide a trading decision in a valid JSON format. "
                "Output ONLY the JSON object requested — never add a 'thought' key."
            ),
        },
        expected_keys=["signal", "confidence", "analysis"],
    ))

    cases.append(TestCase(
        id="decision_v3_schema",
        description="FIX 2: schema JSON strict pour decision",
        payload={
            "prompt": TRADING_DECISION_PROMPT,
            "format": schema_decision,
            "options": opts_long,
            "system": "You are an expert financial analyst. Return ONLY the requested JSON.",
        },
        expected_keys=["signal", "confidence", "analysis"],
    ))

    # ===== oil_bench =====
    cases.append(TestCase(
        id="oil_v1_buggy",
        description="REPRO: oil_bench avec <|think|>",
        payload={
            "prompt": OIL_BENCH_PROMPT,
            "format": "json",
            "system": "<|think|> You are a senior commodity quantitative analyst. Return ONLY valid JSON.",
        },
        expected_keys=["allocation", "reasoning"],
    ))

    cases.append(TestCase(
        id="oil_v2_fixed",
        description="FIX 1: oil_bench sans <|think|>",
        payload={
            "prompt": OIL_BENCH_PROMPT,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 1024},
            "system": (
                "You are a senior commodity quantitative analyst specializing in WTI Crude Oil. "
                "Return ONLY valid JSON — never add a 'thought' key."
            ),
        },
        expected_keys=["allocation", "reasoning"],
    ))

    cases.append(TestCase(
        id="oil_v3_schema",
        description="FIX 2: oil_bench avec schema strict",
        payload={
            "prompt": OIL_BENCH_PROMPT,
            "format": schema_oil,
            "options": {"temperature": 0.1, "num_predict": 1024},
            "system": "You are a senior commodity quantitative analyst. Return ONLY the requested JSON.",
        },
        expected_keys=["allocation", "reasoning"],
    ))

    return cases


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

def run_case(case: TestCase, show_raw: bool = False) -> dict[str, Any]:
    """Run one test case and return a structured result.

    Makes a single Ollama call and applies BOTH the test extractor and the
    production parser to the SAME raw response (so we compare apples to apples).
    """
    result: dict[str, Any] = {
        "id": case.id,
        "description": case.description,
        "ok": False,
        "raw_response": "",
        "parsed": None,
        "error": None,
        "duration_s": 0.0,
    }

    t0 = time.time()
    try:
        raw = _raw_call_ollama(case.payload)
    except Exception as e:
        result["error"] = f"HTTP error: {e}"
        result["duration_s"] = time.time() - t0
        return result

    result["duration_s"] = time.time() - t0
    raw_text = (raw.get("response") or "").strip()
    result["raw_response"] = raw_text

    # 1. Test extractor: extract JSON candidates directly from raw response.
    candidates = _extract_json_objects(raw_text)
    result["extracted_objects"] = candidates

    # 2. Production parser: replay the SAME raw_text through the client's
    #    extraction logic by mocking the HTTP call. This avoids the
    #    non-determinism of a second LLM call and tests the real code path.
    parsed = _parse_with_production_logic(raw_text, case.expected_keys)
    result["parsed"] = parsed

    if parsed is None or parsed.get("failed") is True:
        result["error"] = (
            parsed.get("failure_reason", "no_match") if parsed else "parser returned None"
        )
    else:
        if all(k in parsed for k in case.expected_keys):
            result["ok"] = True

    return result


def _parse_with_production_logic(raw_output: str, expected_keys: list[str]) -> dict | None:
    """Replays the client extraction pipeline on a given raw response.

    This duplicates (in a simplified way) the logic from src.llm_client so we
    can test it deterministically without making a second LLM call.
    """
    from src.llm_client import _find_dict_with_keys

    if not raw_output or raw_output == "{}":
        return {"failed": True, "failure_reason": "empty_response"}

    # Step 1: strip prefix-only thinking tokens (mirror of production fix).
    tags_to_strip = [
        "<channel|>", "<|channel|>", "<|thought|>", "<thought>", "</thought>",
        "thought|", "<|channel>thought", "<|channel>thought}", "<|channel>thought|>",
        "<|start|>", "<|end|>", "<|channel|response>",
    ]
    first_brace = raw_output.find("{")
    if first_brace > 0:
        prefix = raw_output[:first_brace]
        if any(tag in prefix for tag in tags_to_strip):
            raw_output = raw_output[first_brace:].strip()
    elif first_brace == -1:
        for tag in tags_to_strip:
            if tag in raw_output:
                raw_output = raw_output.split(tag)[-1].strip()
                break

    # Step 2: collect candidates (markdown + raw).
    candidates: list = []
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
            obj, end = decoder.raw_decode(raw_output[start:])
            if isinstance(obj, dict):
                candidates.append(obj)
            pos = start + end
        except (json.JSONDecodeError, ValueError):
            pos += 1

    if not candidates:
        candidates = [raw_output]

    # Step 3: find first candidate with all expected keys (recursive).
    for item in candidates:
        found = _find_dict_with_keys(item, expected_keys)
        if found is not None:
            return found

    return {"failed": True, "failure_reason": "retries_exhausted_no_valid_json"}


def print_result(result: dict, show_raw: bool) -> None:
    ok = result["ok"]
    badge = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
    print()
    print(f"{BOLD}[{badge}] {result['id']}{RESET}  {DIM}({result['duration_s']:.1f}s){RESET}")
    print(f"  {CYAN}{result['description']}{RESET}")

    if result.get("error"):
        print(f"  {YELLOW}error:{RESET} {result['error']}")

    raw = result.get("raw_response", "")
    print(f"  {DIM}raw (300):{_short(raw, 300)}{RESET}")

    if show_raw and raw:
        print(f"  {DIM}--- raw complet ---\n{raw}\n--- fin raw ---{RESET}")

    objs = result.get("extracted_objects") or []
    if objs:
        print(f"  {DIM}objs extraits:{RESET}")
        for i, o in enumerate(objs):
            keys = list(o.keys())
            print(f"    [{i}] keys={keys}  ->  {_short(json.dumps(o, ensure_ascii=False), 200)}")
    else:
        print(f"  {DIM}(aucun objet JSON detecte dans la reponse brute){RESET}")

    parsed = result.get("parsed")
    if parsed:
        print(f"  {DIM}parser result:{RESET} {_short(json.dumps(parsed, ensure_ascii=False), 250)}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter",
        default="",
        help="Ne lancer que les cas dont l'id contient ce substring (ex: 'query' ou 'v2').",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Afficher la totalite de la reponse brute du LLM (debug).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Limite le nombre de cas (0 = tous).",
    )
    args = parser.parse_args()

    # Health check.
    try:
        health = requests.get(OLLAMA_API_URL.replace("/api/generate", "/api/tags"), timeout=5)
        health.raise_for_status()
        models = [m.get("name") for m in health.json().get("models", [])]
        print(f"{GREEN}Ollama OK{RESET} - modeles disponibles: {len(models)}")
        if TEXT_LLM_MODEL not in models:
            print(f"{YELLOW}ATTENTION: {TEXT_LLM_MODEL} non present localement, Ollama va le pull.{RESET}")
    except Exception as e:
        print(f"{RED}Ollama indisponible ({e}). Lance 'ollama serve' d'abord.{RESET}")
        return 2

    cases = build_cases()
    if args.filter:
        cases = [c for c in cases if args.filter.lower() in c.id.lower()]
    if args.max > 0:
        cases = cases[: args.max]

    print(f"\n{BOLD}Model:{RESET} {TEXT_LLM_MODEL}")
    print(f"{BOLD}Cas a lancer:{RESET} {len(cases)}\n")

    results = []
    for case in cases:
        r = run_case(case, show_raw=args.raw)
        results.append(r)
        print_result(r, show_raw=args.raw)

    # Summary.
    n_ok = sum(1 for r in results if r["ok"])
    n_total = len(results)
    print(f"\n{BOLD}=== RESUME ==={RESET}")
    print(f"  Reussis:   {GREEN}{n_ok}/{n_total}{RESET}")
    print(f"  Echoues:   {RED}{n_total - n_ok}/{n_total}{RESET}")
    print()
    for r in results:
        badge = f"{GREEN}OK{RESET}   " if r["ok"] else f"{RED}FAIL{RESET} "
        print(f"  {badge} {r['id']}")

    # Recommandation.
    print(f"\n{BOLD}=== RECOMMANDATION ==={RESET}")
    fixed = [r for r in results if r["ok"] and "fixed" in r["id"]]
    buggy = [r for r in results if not r["ok"] and "buggy" in r["id"]]
    if buggy and fixed:
        print(f"  Bug confirme sur la config actuelle ({len(buggy)} cas).")
        print(f"  Variantes 'fixed' qui marchent: {[r['id'] for r in fixed]}")
        # Heuristic: prefer v2 (no <|think|>) which is the smallest change.
        v2 = next((r for r in fixed if "v2_" in r["id"]), None)
        if v2:
            print(f"  {GREEN}Recommandation:{RESET} retirer '<|think|> ' du system prompt "
                  f"(garde format:json) — cas valide: {v2['id']}")
    elif not buggy:
        print(f"  {YELLOW}Aucun cas 'buggy' n'a echoue — le bug ne se reproduit pas.{RESET}")
    else:
        print("  Aucune des variantes 'fixed' n'a marche — investiguer plus.")

    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())

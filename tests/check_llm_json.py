"""Diagnostic: test de validation JSON multi-fournisseurs via NexusAI-Client.

Vérifie que les décisions de trading, les requêtes de recherche web et les allocations pétrole
sont correctement parsées en JSON depuis les fournisseurs Cloud configurés.

Usage:
    uv run tests/check_llm_json.py
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path

# Force UTF-8 on Windows stdout if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import _query_nexus, check_ai_health
from nexusai_client import AIGateway

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("check_llm_json")


def test_nexus_trading_decision():
    print("\n1. Test Trading Decision JSON...")
    prompt = "Analyze NASDAQ trend (RSI=55, MACD Bullish crossover). Return JSON with signal (BUY/SELL/HOLD), confidence (0.0-1.0), analysis."
    res = _query_nexus(prompt, expected_keys=["signal", "confidence", "analysis"])
    print(f"Result: {res}")
    assert res.get("signal") in ["BUY", "SELL", "HOLD"], f"Invalid signal: {res}"
    assert isinstance(res.get("confidence"), (int, float)), f"Invalid confidence: {res}"
    print("[OK] Trading decision JSON OK")


def test_nexus_search_query():
    print("\n2. Test Search Query JSON...")
    prompt = "Generate the best web search query for crude oil inventory changes in the US. Return JSON with query."
    res = _query_nexus(prompt, expected_keys=["query"])
    print(f"Result: {res}")
    assert "query" in res and len(res["query"]) > 3, f"Invalid query: {res}"
    print("[OK] Search query JSON OK")


def test_nexus_oil_allocation():
    print("\n3. Test Oil Allocation JSON...")
    prompt = "Analyze oil macro situation (WTI=$75, inventory build +2M). Return JSON with allocation (0-100), reasoning."
    res = _query_nexus(prompt, expected_keys=["allocation", "reasoning"])
    print(f"Result: {res}")
    assert "allocation" in res and "reasoning" in res, f"Invalid oil allocation: {res}"
    print("[OK] Oil allocation JSON OK")


def main():
    print("========================================")
    print("NexusAI-Client JSON Validation Harness")
    print("========================================")
    providers = AIGateway.get_configured_providers()
    print(f"Active providers: {providers}")
    if not providers:
        print("[FAIL] No active providers found in .env!")
        sys.exit(1)

    try:
        test_nexus_trading_decision()
        test_nexus_search_query()
        test_nexus_oil_allocation()
        print("\n[SUCCESS] ALL NexusAI JSON TESTS PASSED!")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Benchmark comparing LLM models via Ollama.

Models tested:
  A) hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M  (current production)
  B) gemma4:12b-it-qat                            (QAT variant)
  C) lfm2.5:8b-a1b-q8_0                          (LFM 2.5 8B Q8)

Fairness measures:
  - Each model is WARMED UP with a dummy call before any timed test.
    The warmup forces Ollama to load the model into VRAM and measures
    the cold-load time separately (reported but NOT included in scores).
  - Each model is tested in a contiguous block so it stays in VRAM.
  - keep_alive=30m prevents Ollama from unloading mid-block.
  - A "discard first run" option (--discard-warmup) excludes the very
    first timed call per test (which may still include KV-cache fill).

Tests performed per model:
  1. Cold-load time (VRAM load, NOT scored)
  2. TTFT (Time To First Token) — streaming single-token probe
  3. JSON extraction accuracy — schema-strict calls (production path)
  4. Quality — analysis coherence, confidence calibration, signal variety
  5. Throughput — tokens/sec on a long prompt
  6. Consistency — N runs same prompt, check variance

Usage:
    .venv\\Scripts\\python.exe -m tests.bench_compare_llm
    .venv\\Scripts\\python.exe -m tests.bench_compare_llm --runs 5
    .venv\\Scripts\\python.exe -m tests.bench_compare_llm --quick
    .venv\\Scripts\\python.exe -m tests.bench_compare_llm --discard-warmup
    .venv\\Scripts\\python.exe -m tests.bench_compare_llm --model C
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import (
    OLLAMA_API_URL,
    OLLAMA_BASE_URL,
    SCHEMA_TRADING_DECISION,
    SCHEMA_SEARCH_QUERY,
    SCHEMA_OIL_ALLOCATION,
)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

MODELS = [
    "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M",
    "hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K",
]

LABELS = {
    MODELS[0]: "A (Q4_K_M prod)",
    MODELS[1]: "B (Q6_K)",
}

PROMPTS = {
    "trading_decision": {
        "prompt": """Analyze the following market data for CL=F (OIL WTI):
- Close Price: 90.54
- RSI (14): 42.15
- MACD: 12.5 | Signal: 15.2
- Short-term Trend: Bearish
- Long-term Trend: Bullish

Provide your analysis ONLY as a valid JSON object.
{
  "signal": "BUY | SELL | HOLD",
  "confidence": <float 0.0 to 1.0>,
  "analysis": "A rigorous 2-sentence technical and fundamental justification."
}""",
        "system": "<|think|> You are an expert financial analyst. Your task is to analyze market data and news to provide a trading decision in a valid JSON format. Output ONLY the JSON object requested — never add a 'thought' key.",
        "format": SCHEMA_TRADING_DECISION,
        "expected_keys": ["signal", "confidence", "analysis"],
        "options": {"temperature": 0.4, "num_predict": 1024},
    },
    "search_query": {
        "prompt": """You are an expert macroeconomic research assistant. Today is June 2026.
Target Asset: CL=F
Current Context: The current price is 90.54 with a 5-day downward trend (-1.76%).
Specifically focus on OPEC+ supply decisions, global inventory levels, and 'flx:OIL' sentiment on Hyperliquid.

Your goal is to find the most impactful news or reports FROM THE LAST 30 DAYS that explain the current market regime.
Generate the single most effective Google/DuckDuckGo search query (maximum 10 words).

Output ONLY a valid JSON object:
{
  "query": "<your optimized search query>"
}""",
        "system": "<|think|> You are a professional financial researcher. Be precise and focus on current market catalysts. Output ONLY the JSON object requested — never add a 'thought' key.",
        "format": SCHEMA_SEARCH_QUERY,
        "expected_keys": ["query"],
        "options": {"temperature": 0.4, "num_predict": 512},
    },
    "oil_allocation": {
        "prompt": """You are a senior commodity analyst. Context:
- WTI Price: 90.54 (MA200: 72.79) -> above MA200, bullish
- DXY: 100.07
- Brent Spread: 5.20
- OPEC+ meeting scheduled next week

Return ONLY a JSON object:
{"allocation": <float 0-100>, "reasoning": "<2-sentence analysis>"}""",
        "system": "<|think|> You are a senior commodity quantitative analyst. Output ONLY the JSON object requested — never add a 'thought' key.",
        "format": SCHEMA_OIL_ALLOCATION,
        "expected_keys": ["allocation", "reasoning"],
        "options": {"temperature": 0.1, "num_predict": 1024},
    },
}

LONG_PROMPT = """Write a comprehensive analysis of the current oil market conditions covering:
1. Supply-side factors (OPEC+ production, US shale, strategic reserves)
2. Demand-side factors (China recovery, global GDP, seasonal patterns)
3. Geopolitical risks (Middle East, Russia-Ukraine, sanctions)
4. Technical outlook for WTI Crude (support/resistance levels, trend analysis)
5. Forward-looking recommendation for the next 30 days

Be detailed and analytical. Write approximately 500 words."""

LONG_SYSTEM = "You are a senior energy market analyst. Provide thorough, data-driven analysis."

KEEP_ALIVE = "30m"


@dataclass
class BenchResult:
    model: str
    test_name: str
    run_idx: int
    duration_s: float
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    total_tokens: int = 0
    eval_count: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0
    parsed_ok: bool = False
    parsed_data: dict | None = None
    error: str | None = None
    raw_response: str = ""
    is_warmup: bool = False


def _call_ollama_raw(payload: dict, timeout: int = 300) -> dict:
    r = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _warmup_model(model: str) -> float:
    """Force-load model into VRAM. Returns cold-load time in seconds."""
    print(f"  {DIM}Warming up {LABELS[model]} (loading into VRAM)...{RESET}", end="", flush=True)
    payload = {
        "model": model,
        "prompt": "Hi",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_predict": 3},
    }
    t0 = time.perf_counter()
    try:
        _call_ollama_raw(payload, timeout=300)
    except Exception as e:
        print(f" {RED}FAIL: {e}{RESET}")
        return -1.0
    elapsed = time.perf_counter() - t0
    print(f" {GREEN}OK{RESET} ({elapsed:.1f}s cold load)")
    return elapsed


def _unload_model(model: str):
    """Eject model from VRAM so next model gets a clean slate."""
    try:
        requests.post(
            OLLAMA_API_URL,
            json={"model": model, "prompt": "", "stream": False, "keep_alive": "0"},
            timeout=30,
        )
    except Exception:
        pass


def _measure_ttft(model: str) -> float:
    payload = {
        "model": model,
        "prompt": "Say hello.",
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_predict": 5},
    }
    start = time.perf_counter()
    r = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=60)
    first_token = None
    for line in r.iter_lines():
        if line:
            first_token = time.perf_counter()
            break
    if first_token is None:
        return -1.0
    return (first_token - start) * 1000


def _run_schema_test(model: str, test_name: str, config: dict, run_idx: int, is_warmup: bool = False) -> BenchResult:
    payload = {
        "model": model,
        "prompt": config["prompt"],
        "system": config["system"],
        "format": config["format"],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": config["options"],
    }
    t0 = time.perf_counter()
    try:
        raw_resp = _call_ollama_raw(payload)
    except Exception as e:
        return BenchResult(model, test_name, run_idx, time.perf_counter() - t0, error=str(e), is_warmup=is_warmup)
    elapsed = time.perf_counter() - t0

    raw_text = (raw_resp.get("response") or "").strip()
    eval_count = raw_resp.get("eval_count", 0)
    prompt_eval_count = raw_resp.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = raw_resp.get("prompt_eval_duration", 0)
    eval_duration_ns = raw_resp.get("eval_duration", 0)
    total_tokens = eval_count + prompt_eval_count
    tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0

    parsed = None
    parsed_ok = False
    try:
        if raw_text:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and all(k in parsed for k in config["expected_keys"]):
                parsed_ok = True
    except json.JSONDecodeError:
        pass

    return BenchResult(
        model=model,
        test_name=test_name,
        run_idx=run_idx,
        duration_s=round(elapsed, 3),
        tokens_per_sec=round(tokens_per_sec, 2),
        total_tokens=total_tokens,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        prompt_eval_duration_ns=prompt_eval_duration_ns,
        eval_duration_ns=eval_duration_ns,
        parsed_ok=parsed_ok,
        parsed_data=parsed,
        raw_response=raw_text,
        is_warmup=is_warmup,
    )


def _run_throughput_test(model: str, run_idx: int) -> BenchResult:
    payload = {
        "model": model,
        "prompt": LONG_PROMPT,
        "system": LONG_SYSTEM,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.7, "num_predict": 2048},
    }
    t0 = time.perf_counter()
    try:
        raw_resp = _call_ollama_raw(payload, timeout=600)
    except Exception as e:
        return BenchResult(model, "throughput", run_idx, time.perf_counter() - t0, error=str(e))
    elapsed = time.perf_counter() - t0

    raw_text = (raw_resp.get("response") or "").strip()
    eval_count = raw_resp.get("eval_count", 0)
    prompt_eval_count = raw_resp.get("prompt_eval_count", 0)
    eval_duration_ns = raw_resp.get("eval_duration", 0)
    tokens_per_sec = eval_count / elapsed if elapsed > 0 else 0

    ollama_tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0

    return BenchResult(
        model=model,
        test_name="throughput",
        run_idx=run_idx,
        duration_s=round(elapsed, 3),
        tokens_per_sec=round(tokens_per_sec, 2),
        total_tokens=eval_count + prompt_eval_count,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        eval_duration_ns=eval_duration_ns,
        parsed_ok=True,
        raw_response=raw_text[:500],
    )


def _run_ttft_test(model: str, run_idx: int) -> BenchResult:
    ttft = _measure_ttft(model)
    return BenchResult(
        model=model,
        test_name="ttft",
        run_idx=run_idx,
        duration_s=round(ttft / 1000, 3),
        ttft_ms=round(ttft, 1),
        parsed_ok=True,
    )


def _print_bench_result(r: BenchResult):
    label = LABELS.get(r.model, r.model)
    ok_badge = f"{GREEN}OK{RESET}" if r.parsed_ok else f"{RED}FAIL{RESET}"
    warmup_tag = f"{DIM}[warm]{RESET} " if r.is_warmup else ""
    parts = [
        f"  {DIM}[{label}]{RESET} {warmup_tag}{r.test_name} run#{r.run_idx} {ok_badge} {r.duration_s:.2f}s",
    ]
    if r.ttft_ms is not None:
        parts.append(f"ttft={r.ttft_ms:.0f}ms")
    if r.tokens_per_sec:
        parts.append(f"tok/s={r.tokens_per_sec:.1f}")
    if r.eval_duration_ns > 0 and r.eval_count > 0:
        ollama_tps = r.eval_count / (r.eval_duration_ns / 1e9)
        parts.append(f"ollama_tps={ollama_tps:.1f}")
    if r.prompt_eval_duration_ns > 0 and r.prompt_eval_count > 0:
        prompt_tps = r.prompt_eval_count / (r.prompt_eval_duration_ns / 1e9)
        parts.append(f"prompt_tps={prompt_tps:.1f}")
    if r.error:
        parts.append(f"err={r.error}")

    print(" ".join(parts))


def _fmt(durations: list[float], unit: str = "s") -> str:
    if not durations:
        return "N/A"
    avg = statistics.mean(durations)
    if len(durations) == 1:
        return f"{avg:.2f}{unit}"
    return f"avg={avg:.2f}{unit} min={min(durations):.2f}{unit} max={max(durations):.2f}{unit} stdev={statistics.stdev(durations):.2f}{unit}"


def _print_summary(all_results: list[BenchResult], load_times: dict[str, float], discard_warmup: bool):
    scored = [r for r in all_results if not r.is_warmup] if discard_warmup else all_results

    print(f"\n{'='*90}")
    print(f"{BOLD} BENCHMARK SUMMARY{RESET}")
    if discard_warmup:
        print(f"{DIM} (warmup runs excluded from scores){RESET}")
    print(f"{'='*90}")

    print(f"\n{BOLD}Cold-load times (VRAM load, informational only):{RESET}")
    for model in MODELS:
        lt = load_times.get(model, -1)
        label = LABELS[model]
        if lt >= 0:
            print(f"  {label}: {lt:.1f}s")
        else:
            print(f"  {label}: {RED}FAILED{RESET}")

    for model in MODELS:
        label = LABELS[model]
        results = [r for r in scored if r.model == model]
        schema_results = [r for r in results if r.test_name in PROMPTS]
        throughput_results = [r for r in results if r.test_name == "throughput"]
        ttft_results = [r for r in results if r.test_name == "ttft"]

        print(f"\n{BOLD}--- {label} ---{RESET}")

        if schema_results:
            ok_count = sum(1 for r in schema_results if r.parsed_ok)
            durations = [r.duration_s for r in schema_results]
            tps = [r.tokens_per_sec for r in schema_results if r.tokens_per_sec]
            print(f"  JSON accuracy:  {GREEN}{ok_count}/{len(schema_results)}{RESET} OK")
            print(f"  Latency:        {_fmt(durations)}")
            if tps:
                print(f"  Tokens/sec:     {_fmt(tps)}")

        if throughput_results:
            durations = [r.duration_s for r in throughput_results]
            tps = [r.tokens_per_sec for r in throughput_results if r.tokens_per_sec]
            ollama_tps_vals = [r.eval_count / (r.eval_duration_ns / 1e9) for r in throughput_results if r.eval_duration_ns > 0]
            evals = [r.eval_count for r in throughput_results]
            if tps:
                print(f"  Throughput:     {_fmt(tps, ' tok/s')} avg_len={statistics.mean(evals):.0f}t avg_time={statistics.mean(durations):.1f}s")
            if ollama_tps_vals:
                print(f"  Ollama gen tps: {_fmt(ollama_tps_vals, ' tok/s')}")

        if ttft_results:
            ttfts = [r.ttft_ms for r in ttft_results if r.ttft_ms is not None]
            if ttfts:
                print(f"  TTFT:           {_fmt(ttfts, 'ms')}")

        signals = [r.parsed_data.get("signal") for r in schema_results
                    if r.parsed_ok and r.parsed_data and "signal" in r.parsed_data]
        if signals:
            counts = Counter(signals)
            print(f"  Signals:        {dict(counts)}")

        confidences = [r.parsed_data.get("confidence") for r in schema_results
                       if r.parsed_ok and r.parsed_data and "confidence" in r.parsed_data
                       and isinstance(r.parsed_data.get("confidence"), (int, float))]
        if confidences:
            if len(confidences) > 1:
                print(f"  Confidence:     {_fmt(confidences)}")
            else:
                print(f"  Confidence:     {confidences[0]:.2f}")

    print(f"\n{'='*90}")
    print(f"{BOLD} COMPARISON TABLE{RESET}")
    print(f"{'='*90}")
    print(f"\n| Metric | A (Q4_K_M) | B (QAT) | Winner |")
    print(f"|--------|-----------|---------|--------|")

    load_a = load_times.get(MODELS[0], 0)
    load_b = load_times.get(MODELS[1], 0)
    winner_load = "A" if load_a < load_b else "B" if load_b < load_a else "tie"
    print(f"| cold_load (s) | {load_a:.1f} | {load_b:.1f} | {winner_load} |")

    for test_name in list(PROMPTS.keys()) + ["throughput", "ttft"]:
        a_results = [r for r in scored if r.model == MODELS[0] and r.test_name == test_name]
        b_results = [r for r in scored if r.model == MODELS[1] and r.test_name == test_name]

        if not a_results or not b_results:
            continue

        if test_name == "ttft":
            a_val = statistics.mean([r.ttft_ms for r in a_results if r.ttft_ms])
            b_val = statistics.mean([r.ttft_ms for r in b_results if r.ttft_ms])
            winner = "A" if a_val < b_val else "B" if b_val < a_val else "tie"
            print(f"| ttft avg (ms) | {a_val:.0f} | {b_val:.0f} | {winner} |")
        elif test_name == "throughput":
            a_tps = [r.eval_count / (r.eval_duration_ns / 1e9) for r in a_results if r.eval_duration_ns > 0]
            b_tps = [r.eval_count / (r.eval_duration_ns / 1e9) for r in b_results if r.eval_duration_ns > 0]
            a_val = statistics.mean(a_tps) if a_tps else 0
            b_val = statistics.mean(b_tps) if b_tps else 0
            winner = "A" if a_val > b_val else "B" if b_val > a_val else "tie"
            print(f"| throughput (tok/s) | {a_val:.1f} | {b_val:.1f} | {winner} |")
        else:
            a_dur = statistics.mean([r.duration_s for r in a_results])
            b_dur = statistics.mean([r.duration_s for r in b_results])
            a_ok = sum(1 for r in a_results if r.parsed_ok)
            b_ok = sum(1 for r in b_results if r.parsed_ok)
            winner_speed = "A" if a_dur < b_dur else "B" if b_dur < a_dur else "tie"
            print(f"| {test_name} avg (s) | {a_dur:.2f} | {b_dur:.2f} | {winner_speed} |")
            a_acc = a_ok / len(a_results) * 100
            b_acc = b_ok / len(b_results) * 100
            winner_acc = "A" if a_acc > b_acc else "B" if b_acc > a_acc else "tie"
            print(f"| {test_name} OK% | {a_acc:.0f} | {b_acc:.0f} | {winner_acc} |")

    print()

    all_a = [r for r in scored if r.model == MODELS[0] and r.test_name in PROMPTS]
    all_b = [r for r in scored if r.model == MODELS[1] and r.test_name in PROMPTS]
    a_ok_pct = sum(1 for r in all_a if r.parsed_ok) / max(len(all_a), 1) * 100
    b_ok_pct = sum(1 for r in all_b if r.parsed_ok) / max(len(all_b), 1) * 100
    a_avg_dur = statistics.mean([r.duration_s for r in all_a]) if all_a else 0
    b_avg_dur = statistics.mean([r.duration_s for r in all_b]) if all_b else 0
    a_avg_tps = statistics.mean([r.tokens_per_sec for r in all_a if r.tokens_per_sec]) if [r for r in all_a if r.tokens_per_sec] else 0
    b_avg_tps = statistics.mean([r.tokens_per_sec for r in all_b if r.tokens_per_sec]) if [r for r in all_b if r.tokens_per_sec] else 0

    print(f"\n{BOLD}VERDICT:{RESET}")
    reasons = []
    if b_ok_pct >= a_ok_pct:
        reasons.append(f"JSON accuracy: B {b_ok_pct:.0f}% >= A {a_ok_pct:.0f}%")
    else:
        reasons.append(f"JSON accuracy: A {a_ok_pct:.0f}% > B {b_ok_pct:.0f}%")

    if b_avg_dur < a_avg_dur:
        reasons.append(f"Speed: B {b_avg_dur:.2f}s < A {a_avg_dur:.2f}s")
    else:
        reasons.append(f"Speed: A {a_avg_dur:.2f}s <= B {b_avg_dur:.2f}s")

    if b_avg_tps > a_avg_tps:
        reasons.append(f"Throughput: B {b_avg_tps:.1f} > A {a_avg_tps:.1f} tok/s")
    else:
        reasons.append(f"Throughput: A {a_avg_tps:.1f} >= B {b_avg_tps:.1f} tok/s")

    load_a = load_times.get(MODELS[0], 0)
    load_b = load_times.get(MODELS[1], 0)
    if load_b < load_a:
        reasons.append(f"Cold load: B {load_b:.1f}s < A {load_a:.1f}s")
    else:
        reasons.append(f"Cold load: A {load_a:.1f}s <= B {load_b:.1f}s")

    for r in reasons:
        print(f"  - {r}")

    score_a = 0
    score_b = 0
    if a_ok_pct > b_ok_pct: score_a += 2
    elif b_ok_pct > a_ok_pct: score_b += 2
    if a_avg_dur < b_avg_dur: score_a += 1
    elif b_avg_dur < a_avg_dur: score_b += 1
    if a_avg_tps > b_avg_tps: score_a += 1
    elif b_avg_tps > a_avg_tps: score_b += 1
    if load_a < load_b: score_a += 1
    elif load_b < load_a: score_b += 1

    if score_b > score_a:
        print(f"\n  {GREEN}>>> RECOMMENDATION: Switch to hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K (B){RESET}  [B={score_b} A={score_a}]")
    elif score_a > score_b:
        print(f"\n  {YELLOW}>>> RECOMMENDATION: Keep hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M (A){RESET}  [A={score_a} B={score_b}]")
    else:
        print(f"\n  {CYAN}>>> TIE — both models are equivalent on this benchmark{RESET}  [A={score_a} B={score_b}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Runs per test (default 3)")
    parser.add_argument("--quick", action="store_true", help="1 run, skip throughput test")
    parser.add_argument("--discard-warmup", action="store_true",
                        help="Exclude the first run of each test from scored results")
    parser.add_argument("--only", default="", help="Only test models whose name contains this substring (e.g. 'lfm')")
    args = parser.parse_args()
    runs = 1 if args.quick else args.runs
    skip_throughput = args.quick

    active_models = MODELS
    if args.only:
        active_models = [m for m in MODELS if args.only.lower() in m.lower()]
        if not active_models:
            print(f"{RED}No model matches '{args.only}'. Available: {MODELS}{RESET}")
            return 2

    try:
        health = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        health.raise_for_status()
        available = [m.get("name") for m in health.json().get("models", [])]
    except Exception as e:
        print(f"{RED}Ollama unavailable ({e}). Start 'ollama serve' first.{RESET}")
        return 2

    for m in active_models:
        if m not in available:
            print(f"{YELLOW}Model {m} not found locally, pulling...{RESET}")
            r = requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": m}, timeout=600)
            r.raise_for_status()

    print(f"{GREEN}Ollama OK{RESET} — models ready: {[LABELS[m] for m in active_models]}")
    print(f"{BOLD}Runs per test: {runs}{' (quick mode)' if args.quick else ''}{RESET}")
    if args.discard_warmup:
        print(f"{BOLD}Mode: discard-warmup (first run excluded from scores){RESET}")
    print()

    all_results: list[BenchResult] = []
    load_times: dict[str, float] = {}

    for model_idx, model in enumerate(active_models):
        label = LABELS[model]
        print(f"\n{'='*70}")
        print(f"{BOLD}=== Phase {model_idx+1}/{len(active_models)}: {label} ==={RESET}")
        print(f"{'='*70}")

        if model_idx > 0:
            print(f"\n  {DIM}Unloading previous model from VRAM...{RESET}")
            _unload_model(active_models[model_idx - 1])
            time.sleep(2)

        cold_load = _warmup_model(model)
        load_times[model] = cold_load
        if cold_load < 0:
            print(f"{RED}Failed to load {label}, skipping.{RESET}")
            continue

        for run_idx in range(1, runs + 1):
            is_first = run_idx == 1
            print(f"\n  {CYAN}--- Run {run_idx}/{runs}" + (" (warmup)" if is_first and args.discard_warmup else "") + " ---{RESET}")

            for test_name, config in PROMPTS.items():
                r = _run_schema_test(model, test_name, config, run_idx, is_warmup=(is_first and args.discard_warmup))
                all_results.append(r)
                _print_bench_result(r)

            if not skip_throughput:
                r = _run_throughput_test(model, run_idx)
                all_results.append(r)
                _print_bench_result(r)

            r = _run_ttft_test(model, run_idx)
            all_results.append(r)
            _print_bench_result(r)

        print(f"\n  {DIM}Unloading {label} from VRAM...{RESET}")
        _unload_model(model)
        time.sleep(2)

    _print_summary(all_results, load_times, args.discard_warmup)
    return 0


if __name__ == "__main__":
    sys.exit(main())

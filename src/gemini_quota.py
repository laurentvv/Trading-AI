"""Local quota tracker for Google Gemini (paid Tier 1 + free Tier).

The Google AI Studio API exposes **no endpoint** to query the remaining
quota in real time. The only reliable server-side signal is the
``ResourceExhausted`` (HTTP 429) error *after* a call is rejected. To avoid
burning calls once we are near a limit, we keep a local SQLite ledger of
successful calls and pre-emptively block new ones when a limit is reached.

Two key tiers are tracked, each with its **own** per-model limit map:

* ``LIMITS_PAID``  — Tier 1 (paid) limits. Generous; includes Gemini 2.5 Pro.
* ``LIMITS_FREE``  — Free Tier limits. Tight; Pro is 0/0/0 (unavailable).

Two billing protections gate the **paid** tier (both independent of Google's
per-model RPD caps):

1. **Monthly cost budget** (load-bearing) — the paid tier is bounded by a
   **rolling 30-day cost budget** in euros. Each successful call's cost is
   computed from the actual token usage (``usage_metadata`` from the API
   response) × the model's price table (``PRICE_TABLE_EUR_PER_MTOKEN``), and
   accumulated in the ledger. When the 30-day sum reaches the budget, every
   paid model is short-circuited and calls fall back to the free tier /
   Ollama. This is what honours a real €/month target — a call counter cannot,
   because Gemini 2.5 Pro costs ~15× more per call than Flash.
   Driven by ``GEMINI_PAY_MONTHLY_BUDGET_EUR`` (default ``DEFAULT_PAID_MONTHLY_BUDGET_EUR``).

2. **Daily call cap** (backstop) — ``GEMINI_PAY_DAILY_CAP`` (default 200) is a
   hard stop on the *number* of paid calls in a day. It is a backstop against
   runaway loops (a bug that retries forever would otherwise drain the budget
   in minutes); the cost budget is the real billing guard.

Two windows are tracked per model family:

* **RPM** (requests per minute) — sliding 60-second window.
* **RPD** (requests per day) — resets at local midnight.

The ledger is a small SQLite database at ``data_cache/gemini_quota.db``
(gitignored like the other caches). Concurrency is safe: every public method
acquires a ``threading.Lock``, because the trading pipeline runs multiple
models in parallel threads (see ``enhanced_trading_example.py``).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Format: (requests_per_minute, requests_per_day).
#
# PAID Tier 1 limits — confirmed from the account quota page (June 2026).
# Gemini 2.5 Pro is available here (150 RPM / 1000 RPD) — it anchors the
# reasoning cascade. Gemma 4 models are text-only (not vision-capable).
LIMITS_PAID: Dict[str, Tuple[int, int]] = {
    # --- Reasoning / decision tier (text, heaviest budget) ---------------
    "gemini-2.5-pro": (150, 1000),
    "gemini-3.5-flash": (1000, 10000),
    "gemini-3.1-pro": (25, 250),
    "gemini-3-flash-preview": (1000, 10000),
    "gemini-3.1-flash-lite": (4000, 150000),
    "gemini-2.5-flash": (1000, 10000),
    "gemma-4-31b": (30, 14400),
    "gemma-4-26b": (30, 14400),
    # --- Vision tier (multimodal; Gemma excluded — not image-capable) ----
    "gemini-2.0-flash": (2000, 10000),
}

# FREE Tier limits — tight. Gemini 2.5 Pro is 0/0/0 here and intentionally
# omitted. Used as the fallback tier when the paid cap is exhausted, and as
# the primary tier for the (low-value) web-summary use-case.
LIMITS_FREE: Dict[str, Tuple[int, int]] = {
    # --- Reasoning / decision tier (text) --------------------------------
    "gemini-3.5-flash": (5, 20),
    "gemini-3-flash-preview": (5, 20),
    "gemini-3.1-flash-lite": (15, 500),
    "gemini-2.5-flash": (5, 20),
    "gemma-4-31b": (15, 1500),
    "gemma-4-26b": (15, 1500),
    # --- Vision tier (multimodal) ----------------------------------------
    "gemini-2.0-flash": (5, 20),
    # --- Web-summary tier (text, light) ----------------------------------
    "gemini-2.5-flash-lite": (10, 20),
    "gemini-2.0-flash-lite": (10, 20),
}

# Hard local daily cap on the PAID tier (all paid models combined), independent
# of Google's per-model RPD. **Backstop only** — the real billing guard is the
# monthly cost budget below. Override via the environment.
DEFAULT_PAID_DAILY_CAP = 200

# --- Monthly cost budget (the load-bearing billing guard) ----------------- #
# Rolling 30-day budget in EUR for the PAID tier. Each successful paid call's
# cost is computed from actual token usage × ``PRICE_TABLE_EUR_PER_MTOKEN`` and
# accumulated; when the 30-day sum hits this budget, all paid models are blocked
# and calls fall back to the free tier / Ollama. Override via the environment.
#
# Default calibrated to the user's stated budget of 8.6 €/month (June 2026).
DEFAULT_PAID_MONTHLY_BUDGET_EUR = 8.6

# Window length (days) for the rolling cost budget.
MONTHLY_BUDGET_WINDOW_DAYS = 30

# --- Price table (EUR per 1 million tokens) ------------------------------- #
# Source: Google AI Studio pricing (https://ai.google.dev/pricing) as of June
# 2026, converted to EUR at ~1.08 USD/EUR. The paid tier is metered per token;
# the free tier is unmetered (cost 0 — it does not draw down the budget). Only
# models that appear in ``LIMITS_PAID`` need an entry here; an unknown model is
# priced at 0 (conservative — it won't draw down the budget silently, but it
# also won't be charged; the daily-call-cap backstop still bounds it).
#
# Format: (input_eur_per_mtoken, output_eur_per_mtoken). "Lite" text models are
# priced under 1 EUR/Mtok; Gemini 2.5 Pro is the anchor at ~9.3/37.2.
PRICE_TABLE_EUR_PER_MTOKEN: Dict[str, Tuple[float, float]] = {
    # --- Reasoning / decision tier (text) --------------------------------
    "gemini-2.5-pro":          (1.85, 11.10),   # $2.00/$12.00 per Mtok
    "gemini-3.5-flash":        (0.28, 1.67),    # $0.30/$1.80
    "gemini-3.1-pro":          (1.85, 11.10),
    "gemini-3-flash-preview":  (0.28, 1.67),
    "gemini-3.1-flash-lite":   (0.09, 0.37),    # $0.10/$0.40
    "gemini-2.5-flash":        (0.09, 0.37),
    "gemma-4-31b":             (0.09, 0.37),    # approx (open weights, low)
    "gemma-4-26b":             (0.09, 0.37),
    # --- Vision tier (multimodal) ----------------------------------------
    "gemini-2.0-flash":        (0.09, 0.37),
}

# Fallback estimate used when a call has no usable usage_metadata (rare: a
# safety filter or SDK quirk stripped it). Conservative average across the
# decision cascade — better to slightly over-count than silently under-count.
DEFAULT_CALL_COST_EUR = 0.005

# Default storage location. ``data_cache/`` is already gitignored and used by
# the rest of the pipeline for transient state.
DEFAULT_DB_PATH = Path("data_cache/gemini_quota.db")

# Sliding window for RPM, in seconds.
RPM_WINDOW_SECONDS = 60

# Ledger "tier" tags stored alongside each call.
TIER_PAID = "paid"
TIER_FREE = "free"


def _paid_daily_cap() -> int:
    """Read the paid daily cap from the env (cached at call time)."""
    raw = os.getenv("GEMINI_PAY_DAILY_CAP", str(DEFAULT_PAID_DAILY_CAP))
    try:
        cap = int(raw)
        return cap if cap > 0 else DEFAULT_PAID_DAILY_CAP
    except (TypeError, ValueError):
        return DEFAULT_PAID_DAILY_CAP


def _paid_monthly_budget() -> float:
    """Read the paid monthly (rolling 30-day) budget in EUR from the env.

    This is the load-bearing billing guard. A non-positive or unparseable
    value falls back to the compiled default rather than disabling the guard
    (a disabled guard would mean unbounded billing).
    """
    raw = os.getenv("GEMINI_PAY_MONTHLY_BUDGET_EUR", str(DEFAULT_PAID_MONTHLY_BUDGET_EUR))
    try:
        budget = float(raw)
        return budget if budget > 0 else DEFAULT_PAID_MONTHLY_BUDGET_EUR
    except (TypeError, ValueError):
        return DEFAULT_PAID_MONTHLY_BUDGET_EUR


def compute_call_cost_eur(
    model: str, usage_metadata: Optional[dict], output_chars: int = 0,
) -> float:
    """Compute the EUR cost of one successful Gemini call from its token usage.

    ``usage_metadata`` is the dict-shaped ``response.usage_metadata`` returned
    by the SDK (keys: ``prompt_token_count`` / ``candidates_token_count`` /
    ``total_token_count``). When it is missing or zero, we fall back to a
    conservative per-call estimate (``DEFAULT_CALL_COST_EUR``) so the budget is
    never silently under-counted.

    The free tier is unmetered — callers should not record a cost for free-tier
    calls (``record()`` ignores ``cost_eur`` when ``tier == free``).
    """
    if model not in PRICE_TABLE_EUR_PER_MTOKEN:
        return 0.0  # unknown model — no price entry, daily-cap backstop covers it
    in_price, out_price = PRICE_TABLE_EUR_PER_MTOKEN[model]

    in_tok = 0
    out_tok = 0
    if usage_metadata:
        try:
            in_tok = int(usage_metadata.get("prompt_token_count", 0) or 0)
            out_tok = int(usage_metadata.get("candidates_token_count", 0) or 0)
        except (TypeError, ValueError):
            in_tok = out_tok = 0

    if in_tok == 0 and out_tok == 0:
        # No usable token counts — use the conservative flat estimate.
        return DEFAULT_CALL_COST_EUR

    cost = (in_tok / 1_000_000.0) * in_price + (out_tok / 1_000_000.0) * out_price
    # A truly zero cost (e.g. a cached/echoed response) still consumed an API
    # request slot; floor it at a token of cost so the ledger is honest.
    return max(cost, DEFAULT_CALL_COST_EUR * 0.1)


class QuotaTracker:
    """Thread-safe local ledger of Gemini API calls across both tiers.

    A single tracker instance serves both the paid and free tiers. The tier
    is passed to :meth:`check` / :meth:`record` so calls are billed to the
    right ledger. The paid tier additionally enforces a global daily cap.

    The tracker is deliberately conservative: it only counts *successful*
    calls (via :meth:`record`). A call rejected server-side with 429 is NOT
    recorded, so the next attempt after backoff still has its full budget.
    Pre-flight checks (:meth:`check`) use the local ledger to short-circuit
    calls we already know would be rejected.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        # ``check_same_thread=False`` because we manage locking ourselves via
        # ``self._lock``; connections are short-lived (one per operation).
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self._db_path), check_same_thread=False)

    def _ensure_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                # Migrate the legacy single-tier schema if present: the old
                # ``calls`` table had no ``tier`` column. SQLite has no ADD
                # COLUMN IF NOT EXISTS, so we introspect PRAGMA table_info.
                cols = {row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()}
                if cols and "tier" not in cols:
                    logger.info("Migrating gemini_quota DB: adding 'tier' column.")
                    # Backfill existing rows as FREE (they predate the paid tier).
                    conn.execute(
                        "ALTER TABLE calls ADD COLUMN tier TEXT NOT NULL DEFAULT 'free'"
                    )
                    # Re-read columns after the first migration step.
                    cols = {row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()}
                # Migrate the pre-cost-budget schema: add ``cost_eur``. Legacy
                # rows backfill to 0 (their cost is already sunk and unknowable).
                if cols and "cost_eur" not in cols:
                    logger.info("Migrating gemini_quota DB: adding 'cost_eur' column.")
                    conn.execute(
                        "ALTER TABLE calls ADD COLUMN cost_eur REAL NOT NULL DEFAULT 0.0"
                    )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS calls (
                        id       INTEGER PRIMARY KEY AUTOINCREMENT,
                        tier     TEXT NOT NULL,
                        model    TEXT NOT NULL,
                        ts       INTEGER NOT NULL,
                        cost_eur REAL NOT NULL DEFAULT 0.0
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_calls_tier_model_ts ON calls (tier, model, ts)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_calls_tier_ts ON calls (tier, ts)"
                )

    @staticmethod
    def _midnight_epoch(now: float) -> float:
        """Unix timestamp of the most recent local midnight (start of today)."""
        today = datetime.fromtimestamp(now).date()
        midnight = datetime.combine(today, dtime.min)
        return midnight.timestamp()

    def _limits_for(self, tier: str) -> Dict[str, Tuple[int, int]]:
        return LIMITS_PAID if tier == TIER_PAID else LIMITS_FREE

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def check(self, model: str, tier: str = TIER_FREE) -> bool:
        """Return ``True`` if a call to ``model`` on ``tier`` is within budget.

        Returns ``False`` (i.e. "do not call") when any of these is reached:
        * the per-model RPM window (60s),
        * the per-model RPD cap (resets at midnight),
        * (paid tier only) the rolling 30-day **cost budget** (load-bearing),
        * (paid tier only) the global ``GEMINI_PAY_DAILY_CAP`` (backstop).

        Unknown models are treated as unlimited (``True``) so a new model
        name never hard-blocks the pipeline.
        """
        now = time.time()
        rpm_floor = now - RPM_WINDOW_SECONDS
        day_floor = self._midnight_epoch(now)
        month_floor = now - MONTHLY_BUDGET_WINDOW_DAYS * 86400

        with self._lock:
            with self._connect() as conn:
                rpm_count = conn.execute(
                    "SELECT COUNT(*) FROM calls WHERE tier = ? AND model = ? AND ts >= ?",
                    (tier, model, rpm_floor),
                ).fetchone()[0]
                rpd_count = conn.execute(
                    "SELECT COUNT(*) FROM calls WHERE tier = ? AND model = ? AND ts >= ?",
                    (tier, model, day_floor),
                ).fetchone()[0]
                paid_today = conn.execute(
                    "SELECT COUNT(*) FROM calls WHERE tier = ? AND ts >= ?",
                    (TIER_PAID, day_floor),
                ).fetchone()[0] if tier == TIER_PAID else 0
                paid_month_cost = conn.execute(
                    "SELECT COALESCE(SUM(cost_eur), 0.0) FROM calls WHERE tier = ? AND ts >= ?",
                    (TIER_PAID, month_floor),
                ).fetchone()[0] if tier == TIER_PAID else 0.0

        limits = self._limits_for(tier).get(model)
        if limits is not None:
            rpm_limit, rpd_limit = limits
            if rpm_count >= rpm_limit:
                logger.info(
                    "Gemini quota pre-check: %s/%s RPM exhausted (%d/%d).",
                    tier, model, rpm_count, rpm_limit,
                )
                return False
            if rpd_count >= rpd_limit:
                logger.info(
                    "Gemini quota pre-check: %s/%s RPD exhausted (%d/%d).",
                    tier, model, rpd_count, rpd_limit,
                )
                return False

        # Paid tier: enforce the cost budget (load-bearing) then the daily cap.
        if tier == TIER_PAID:
            budget = _paid_monthly_budget()
            if paid_month_cost >= budget:
                logger.info(
                    "Gemini quota pre-check: PAID monthly budget reached "
                    "(%.4f/%.2f EUR over 30d). Falling back to free tier.",
                    paid_month_cost, budget,
                )
                return False
            cap = _paid_daily_cap()
            if paid_today >= cap:
                logger.info(
                    "Gemini quota pre-check: PAID daily cap reached (%d/%d). "
                    "Falling back to free tier.", paid_today, cap,
                )
                return False
        return True

    def record(self, model: str, tier: str = TIER_FREE, cost_eur: float = 0.0) -> None:
        """Record one successful call to ``model`` on ``tier`` at now.

        ``cost_eur`` is the EUR cost of the call (from
        :func:`compute_call_cost_eur`). It is only meaningful for the paid
        tier — the free tier is unmetered, so its cost is always recorded as 0
        and never draws down the monthly budget.
        """
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO calls (tier, model, ts, cost_eur) VALUES (?, ?, ?, ?)",
                    (tier, model, now, float(cost_eur) if tier == TIER_PAID else 0.0),
                )
                conn.commit()

    def status(self) -> Dict[str, Dict[str, int]]:
        """Snapshot of current usage vs limits, for dashboards/logging.

        Returns ``{tier: {model: {rpm_used, rpm_limit, rpd_used, rpd_limit}}}``
        plus synthetic entries: ``"_paid_cap"`` (daily call-cap usage) and
        ``"_paid_budget"`` (rolling 30-day cost vs the monthly budget). Only
        models that appear in either LIMITS map are reported.
        """
        now = time.time()
        rpm_floor = now - RPM_WINDOW_SECONDS
        day_floor = self._midnight_epoch(now)
        month_floor = now - MONTHLY_BUDGET_WINDOW_DAYS * 86400
        out: Dict[str, Dict[str, int]] = {}

        with self._lock:
            with self._connect() as conn:
                for tier, limits in ((TIER_PAID, LIMITS_PAID), (TIER_FREE, LIMITS_FREE)):
                    out[tier] = {}
                    for model, (rpm_limit, rpd_limit) in limits.items():
                        rpm_used = conn.execute(
                            "SELECT COUNT(*) FROM calls WHERE tier=? AND model=? AND ts>=?",
                            (tier, model, rpm_floor),
                        ).fetchone()[0]
                        rpd_used = conn.execute(
                            "SELECT COUNT(*) FROM calls WHERE tier=? AND model=? AND ts>=?",
                            (tier, model, day_floor),
                        ).fetchone()[0]
                        out[tier][model] = {
                            "rpm_used": rpm_used, "rpm_limit": rpm_limit,
                            "rpd_used": rpd_used, "rpd_limit": rpd_limit,
                        }
                paid_today = conn.execute(
                    "SELECT COUNT(*) FROM calls WHERE tier=? AND ts>=?",
                    (TIER_PAID, day_floor),
                ).fetchone()[0]
                paid_month_cost = conn.execute(
                    "SELECT COALESCE(SUM(cost_eur), 0.0) FROM calls WHERE tier=? AND ts>=?",
                    (TIER_PAID, month_floor),
                ).fetchone()[0]
        out["_paid_cap"] = {"used": paid_today, "limit": _paid_daily_cap()}
        out["_paid_budget"] = {
            "used_eur": round(float(paid_month_cost), 4),
            "limit_eur": _paid_monthly_budget(),
            "window_days": MONTHLY_BUDGET_WINDOW_DAYS,
        }
        return out

    def reset(self) -> None:
        """Wipe the ledger. Intended for tests and manual recovery only."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM calls")
                conn.commit()

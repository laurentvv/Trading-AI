"""Local quota tracker for Google Gemini (paid Tier 1 + free Tier).

The Google AI Studio API exposes **no endpoint** to query the remaining
quota in real time. The only reliable server-side signal is the
``ResourceExhausted`` (HTTP 429) error *after* a call is rejected. To avoid
burning calls once we are near a limit, we keep a local SQLite ledger of
successful calls and pre-emptively block new ones when a limit is reached.

Two key tiers are tracked, each with its **own** per-model limit map:

* ``LIMITS_PAID``  — Tier 1 (paid) limits. Generous; includes Gemini 2.5 Pro.
* ``LIMITS_FREE``  — Free Tier limits. Tight; Pro is 0/0/0 (unavailable).

On top of Google's per-model RPD caps, the **paid** tier is also bounded by a
local daily cap (``GEMINI_PAY_DAILY_CAP``, default 200) — a hard stop that
protects billing regardless of what Google allows. The paid ledger therefore
counts *every* paid call (all models combined) against this single cap.

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
from typing import Dict, Tuple

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
# of Google's per-model RPD. Protects billing. Override via the environment.
DEFAULT_PAID_DAILY_CAP = 200

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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS calls (
                        id    INTEGER PRIMARY KEY AUTOINCREMENT,
                        tier  TEXT NOT NULL,
                        model TEXT NOT NULL,
                        ts    INTEGER NOT NULL
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
        * (paid tier only) the global ``GEMINI_PAY_DAILY_CAP``.

        Unknown models are treated as unlimited (``True``) so a new model
        name never hard-blocks the pipeline.
        """
        now = time.time()
        rpm_floor = now - RPM_WINDOW_SECONDS
        day_floor = self._midnight_epoch(now)

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

        # Paid tier: also enforce the global daily cap.
        if tier == TIER_PAID:
            cap = _paid_daily_cap()
            if paid_today >= cap:
                logger.info(
                    "Gemini quota pre-check: PAID daily cap reached (%d/%d). "
                    "Falling back to free tier.", paid_today, cap,
                )
                return False
        return True

    def record(self, model: str, tier: str = TIER_FREE) -> None:
        """Record one successful call to ``model`` on ``tier`` at now."""
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO calls (tier, model, ts) VALUES (?, ?, ?)",
                    (tier, model, now),
                )
                conn.commit()

    def status(self) -> Dict[str, Dict[str, int]]:
        """Snapshot of current usage vs limits, for dashboards/logging.

        Returns ``{tier: {model: {rpm_used, rpm_limit, rpd_used, rpd_limit}}}``
        plus a synthetic ``"_paid_cap"`` entry showing usage vs the paid daily
        cap. Only models that appear in either LIMITS map are reported.
        """
        now = time.time()
        rpm_floor = now - RPM_WINDOW_SECONDS
        day_floor = self._midnight_epoch(now)
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
        out["_paid_cap"] = {"used": paid_today, "limit": _paid_daily_cap()}
        return out

    def reset(self) -> None:
        """Wipe the ledger. Intended for tests and manual recovery only."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM calls")
                conn.commit()

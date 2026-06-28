"""Local quota tracker for the Google Gemini Free Tier.

The Google AI Studio API exposes **no endpoint** to query the remaining
quota in real time. The only reliable signal is the ``ResourceExhausted``
HTTP 429 error *after* a call is rejected. To avoid burning calls once we
are already near the limit, we maintain a local ledger of successful calls
and pre-emptively block new ones when the Free Tier limits are reached.

Two windows are tracked per model family:

* **RPM** (requests per minute) — sliding 60-second window.
* **RPD** (requests per day) — resets at local midnight.

The ledger is a small SQLite database at ``data_cache/gemini_quota.db``
(gitignored like the other caches). Concurrency is safe: every public
method acquires a ``threading.Lock``, because the trading pipeline runs
multiple models in parallel threads (see ``enhanced_trading_example.py``).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Free Tier limits for this account, as of June 2026 (Google AI Studio quota
# dashboard). Format: (requests_per_minute, requests_per_day).
#
# NOTE on Gemini 2.5 Pro: it returns 0/0/0 on this key (unavailable) and is
# therefore intentionally OMITTED. The reasoning tier uses the Flash family.
# Gemma 4 models (26B / 31B) are text-only via this API (not vision-capable),
# so they appear only in the text/reasoning cascade, not the vision one.
#
# Each use-case in gemini_gateway.py defines an ordered CASCADE over these
# models; quota is additive across the cascade (e.g. the reasoning cascade
# can serve ~2560 decisions/day in theory before any fallback to free-llm).
LIMITS: Dict[str, Tuple[int, int]] = {
    # --- Reasoning / decision tier (text, heaviest budget) ---------------
    "gemini-3.5-flash": (5, 20),
    "gemini-3-flash-preview": (5, 20),
    "gemini-3.1-flash-lite": (15, 500),
    "gemini-2.5-flash": (5, 20),
    "gemma-4-31b": (15, 1500),
    "gemma-4-26b": (15, 1500),
    # --- Vision tier (multimodal; Gemma excluded — not image-capable) ----
    "gemini-2.0-flash": (5, 20),
    # --- Web-summary tier (text, light) ---------------------------------
    "gemini-2.5-flash-lite": (10, 20),
    "gemini-2.0-flash-lite": (10, 20),
}

# Default storage location. ``data_cache/`` is already gitignored and used by
# the rest of the pipeline for transient state.
DEFAULT_DB_PATH = Path("data_cache/gemini_quota.db")

# Sliding window for RPM, in seconds.
RPM_WINDOW_SECONDS = 60


class QuotaTracker:
    """Thread-safe local ledger of Gemini API calls.

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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS calls (
                        id    INTEGER PRIMARY KEY AUTOINCREMENT,
                        model TEXT NOT NULL,
                        ts    INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_calls_model_ts ON calls (model, ts)"
                )

    @staticmethod
    def _midnight_epoch(now: float) -> float:
        """Unix timestamp of the most recent local midnight (start of today)."""
        today = datetime.fromtimestamp(now).date()
        midnight = datetime.combine(today, dtime.min)
        return midnight.timestamp()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def check(self, model: str) -> bool:
        """Return ``True`` if a call to ``model`` is within RPM+RPD budget.

        Returns ``False`` (i.e. "do not call") when either the per-minute or
        per-day local count has already reached the Free Tier limit. Unknown
        models are treated as unlimited (``True``) so a new model name never
        hard-blocks the pipeline.
        """
        limits = LIMITS.get(model)
        if limits is None:
            return True
        rpm_limit, rpd_limit = limits
        now = time.time()
        rpm_floor = now - RPM_WINDOW_SECONDS
        day_floor = self._midnight_epoch(now)

        with self._lock:
            with self._connect() as conn:
                rpm_count = conn.execute(
                    "SELECT COUNT(*) FROM calls WHERE model = ? AND ts >= ?",
                    (model, rpm_floor),
                ).fetchone()[0]
                rpd_count = conn.execute(
                    "SELECT COUNT(*) FROM calls WHERE model = ? AND ts >= ?",
                    (model, day_floor),
                ).fetchone()[0]

        if rpm_count >= rpm_limit:
            logger.warning(
                "Gemini quota pre-check: %s RPM budget exhausted (%d/%d in last 60s).",
                model, rpm_count, rpm_limit,
            )
            return False
        if rpd_count >= rpd_limit:
            logger.warning(
                "Gemini quota pre-check: %s RPD budget exhausted (%d/%d today).",
                model, rpd_count, rpd_limit,
            )
            return False
        return True

    def record(self, model: str) -> None:
        """Record one successful call to ``model`` at the current instant."""
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO calls (model, ts) VALUES (?, ?)",
                    (model, now),
                )
                conn.commit()

    def status(self) -> Dict[str, Dict[str, int]]:
        """Snapshot of current usage vs limits, for dashboards/logging.

        Returns ``{model: {"rpm_used": n, "rpm_limit": N, "rpd_used": m,
        "rpd_limit": M}}`` for every model in :data:`LIMITS`.
        """
        now = time.time()
        rpm_floor = now - RPM_WINDOW_SECONDS
        day_floor = self._midnight_epoch(now)
        out: Dict[str, Dict[str, int]] = {}
        with self._lock:
            with self._connect() as conn:
                for model, (rpm_limit, rpd_limit) in LIMITS.items():
                    rpm_used = conn.execute(
                        "SELECT COUNT(*) FROM calls WHERE model = ? AND ts >= ?",
                        (model, rpm_floor),
                    ).fetchone()[0]
                    rpd_used = conn.execute(
                        "SELECT COUNT(*) FROM calls WHERE model = ? AND ts >= ?",
                        (model, day_floor),
                    ).fetchone()[0]
                    out[model] = {
                        "rpm_used": rpm_used,
                        "rpm_limit": rpm_limit,
                        "rpd_used": rpd_used,
                        "rpd_limit": rpd_limit,
                    }
        return out

    def reset(self) -> None:
        """Wipe the ledger. Intended for tests and manual recovery only."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM calls")
                conn.commit()

"""GO-gate 6 tests — scheduler instance lock, loop resilience, brief catch-up.

Audit 2026-08-19 (docs/AUDIT_PROD_INDEPENDANT_2026-08-19.md):
  I1 — no instance lock: concurrent scheduler instances were OBSERVED in
       PROD (3 launches within 61s on 18/08, doubled cycles on 19/08).
  I2 — no crash recovery: any non-KeyboardInterrupt exception killed the
       scheduler silently; no catch-up window for the morning brief (the
       01:00-01:59 window missed 22 consecutive days in the previous run).
"""

import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import schedule


class TestSchedulerLock(unittest.TestCase):
    """GO-gate 6 (I1): inter-process exclusivity + stale-lock breaking."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lock_path = Path(self._tmp.name) / "scheduler.lock"
        self.addCleanup(self._tmp.cleanup)

    def test_first_instance_acquires(self):
        self.assertTrue(schedule.acquire_scheduler_lock(self.lock_path))
        self.assertTrue(self.lock_path.exists())
        self.assertEqual(self.lock_path.read_text(), str(os.getpid()))

    def test_second_instance_refused(self):
        self.assertTrue(schedule.acquire_scheduler_lock(self.lock_path))
        self.assertFalse(schedule.acquire_scheduler_lock(self.lock_path))

    def test_release_allows_reacquisition(self):
        self.assertTrue(schedule.acquire_scheduler_lock(self.lock_path))
        schedule.release_scheduler_lock(self.lock_path)
        self.assertFalse(self.lock_path.exists())
        self.assertTrue(schedule.acquire_scheduler_lock(self.lock_path))

    def test_stale_lock_is_broken(self):
        self.assertTrue(schedule.acquire_scheduler_lock(self.lock_path))
        # Age the lock beyond the stale threshold (simulate a dead instance).
        old = time.time() - (schedule.SCHEDULER_LOCK_STALE_SECONDS + 3600)
        os.utime(self.lock_path, (old, old))
        self.assertTrue(schedule.acquire_scheduler_lock(self.lock_path))

    def test_fresh_lock_is_not_broken(self):
        self.assertTrue(schedule.acquire_scheduler_lock(self.lock_path))
        self.lock_path.touch()  # live keeper refresh
        self.assertFalse(schedule.acquire_scheduler_lock(self.lock_path))


class TestLoopResilience(unittest.TestCase):
    """GO-gate 6 (I2): an unexpected exception must not kill the loop."""

    def test_run_loop_iteration_swallows_exceptions(self):
        state = {"last_run_time": "Aucun", "next_run": datetime.now(),
                 "last_morning_brief_date": None, "last_council_date": None}
        with patch.object(schedule, "scheduler_tick", side_effect=RuntimeError("boom")):
            # Must NOT raise — the loop survives (contract critère 22).
            result = schedule.run_loop_iteration(state)
        self.assertFalse(result)

    def test_run_loop_iteration_passes_on_success(self):
        state = {"last_run_time": "Aucun", "next_run": datetime.now(),
                 "last_morning_brief_date": None, "last_council_date": None}
        with patch.object(schedule, "scheduler_tick") as tick:
            result = schedule.run_loop_iteration(state)
        self.assertTrue(result)
        tick.assert_called_once_with(state)

    def test_keyboard_interrupt_still_propagates(self):
        state = {}
        with patch.object(schedule, "scheduler_tick", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                schedule.run_loop_iteration(state)


class TestMorningBriefCatchUp(unittest.TestCase):
    """GO-gate 6: the brief runs any time after 01:00 when missing (no more
    single 01:00-01:59 window), exactly once per day, disk-guarded."""

    def _state(self):
        return {"last_run_time": "Aucun", "next_run": datetime.now(),
                "last_morning_brief_date": None, "last_council_date": None}

    def test_brief_runs_in_afternoon_when_missing(self):
        # Wednesday 14:23, market closed, no brief produced today.
        with patch.object(schedule, "is_market_open", return_value=(False, "Test")), \
             patch.object(schedule, "_morning_brief_done_today", return_value=False), \
             patch.object(schedule, "run_morning_brief") as brief, \
             patch.object(schedule, "datetime") as mock_dt, \
             patch.object(schedule, "console"):
            mock_dt.now.return_value = datetime(2026, 8, 19, 14, 23)  # Wednesday
            state = self._state()
            schedule.scheduler_tick(state)
            brief.assert_called_once()
            # Second tick the same day: not re-run (in-memory guard).
            schedule.scheduler_tick(state)
            brief.assert_called_once()

    def test_brief_skipped_when_disk_says_done(self):
        with patch.object(schedule, "is_market_open", return_value=(False, "Test")), \
             patch.object(schedule, "_morning_brief_done_today", return_value=True), \
             patch.object(schedule, "run_morning_brief") as brief, \
             patch.object(schedule, "datetime") as mock_dt, \
             patch.object(schedule, "console"):
            mock_dt.now.return_value = datetime(2026, 8, 19, 14, 23)
            schedule.scheduler_tick(self._state())
            brief.assert_not_called()

    def test_brief_not_run_before_one_am(self):
        with patch.object(schedule, "is_market_open", return_value=(False, "Test")), \
             patch.object(schedule, "_morning_brief_done_today", return_value=False), \
             patch.object(schedule, "run_morning_brief") as brief, \
             patch.object(schedule, "datetime") as mock_dt, \
             patch.object(schedule, "console"):
            mock_dt.now.return_value = datetime(2026, 8, 19, 0, 40)  # 00:40, before 01:00
            schedule.scheduler_tick(self._state())
            brief.assert_not_called()

    def test_morning_brief_done_today_file_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            brief_file = Path(tmp) / "morning_market_brief.md"
            brief_file.write_text("# brief", encoding="utf-8")
            self.assertTrue(schedule._morning_brief_done_today(brief_file))
            old = time.time() - 3 * 86400
            os.utime(brief_file, (old, old))
            self.assertFalse(schedule._morning_brief_done_today(brief_file))
            self.assertFalse(schedule._morning_brief_done_today(Path(tmp) / "absent.md"))


if __name__ == "__main__":
    unittest.main()

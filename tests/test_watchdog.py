"""Watchdog d'exploitation (2026-09-29) : scheduler mort, cycle absent, position sans stop, déduplication, relance.

Motif : le scheduler est resté arrêté 6,4 jours (15/09 → 22/09) sans relance ni alerte ; disponibilité 66 %.
"""

import datetime as dt
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import watchdog as wd  # noqa: E402
from src import notifier  # noqa: E402

# Mardi 29/09/2026 11:00 : en séance.
NOW = dt.datetime(2026, 9, 29, 11, 0, 0)
SATURDAY = dt.datetime(2026, 10, 3, 11, 0, 0)
NIGHT = dt.datetime(2026, 9, 29, 3, 0, 0)


def _lock(base: Path, pid: int = 4242, age_sec: float = 10, now: dt.datetime = NOW):
    lock = base / "scheduler.lock"
    lock.write_text(str(pid))
    ts = now.timestamp() - age_sec
    os.utime(lock, (ts, ts))


def _sched_log(base: Path, last_cycle: dt.datetime | None):
    lines = ["2026-09-29 01:00:25,802 - INFO - 🌅 Lancement du Morning Brief"]
    if last_cycle:
        lines.append(f"{last_cycle:%Y-%m-%d %H:%M:%S},123 - INFO - 🚀 Lancement du cycle de trading pour ['SXRV.DE']")
        lines.append(f"{last_cycle:%Y-%m-%d %H:%M:%S},999 - INFO - ✅ Cycle terminé avec succès")
    (base / "scheduler.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _state(base: Path, tickers: dict):
    (base / "t212_portfolio_state.json").write_text(json.dumps({"tickers": tickers}), encoding="utf-8")


ALIVE = lambda pid: True  # noqa: E731
DEAD = lambda pid: False  # noqa: E731


@pytest.fixture
def base(tmp_path):
    return tmp_path


class TestSchedulerAlive:
    def test_healthy_scheduler_has_no_alert(self, base):
        _lock(base)
        assert wd.check_scheduler_alive(base, NOW, ALIVE) is None

    def test_missing_lock(self, base):
        a = wd.check_scheduler_alive(base, NOW, ALIVE)
        assert a.key == "scheduler-dead" and a.level == "CRITICAL"

    def test_dead_pid(self, base):
        _lock(base)
        assert "n'existe plus" in wd.check_scheduler_alive(base, NOW, DEAD).message

    def test_frozen_lock(self, base):
        _lock(base, age_sec=600)
        assert "non rafraîchi" in wd.check_scheduler_alive(base, NOW, ALIVE).message

    def test_unreadable_lock(self, base):
        (base / "scheduler.lock").write_text("pas un pid")
        assert wd.check_scheduler_alive(base, NOW, ALIVE).key == "scheduler-dead"


class TestCycleFreshness:
    def test_recent_cycle_is_fine(self, base):
        _sched_log(base, NOW - dt.timedelta(minutes=25))
        assert wd.check_cycle_freshness(base, NOW) is None

    def test_old_cycle_in_session_alerts(self, base):
        _sched_log(base, NOW - dt.timedelta(minutes=95))
        a = wd.check_cycle_freshness(base, NOW)
        assert a.key == "no-cycle" and "95 min" in a.message

    def test_no_cycle_at_all_alerts(self, base):
        _sched_log(base, None)
        assert wd.check_cycle_freshness(base, NOW).key == "no-cycle"

    def test_outside_the_session_nothing_is_expected(self, base):
        _sched_log(base, NOW - dt.timedelta(hours=20))
        assert wd.check_cycle_freshness(base, SATURDAY) is None
        assert wd.check_cycle_freshness(base, NIGHT) is None

    def test_first_cycle_of_the_day_is_not_judged_before_0900(self, base):
        early = dt.datetime(2026, 9, 29, 8, 45)
        _sched_log(base, early - dt.timedelta(hours=15))
        assert wd.check_cycle_freshness(base, early) is None


class TestPositionsHaveStops:
    def test_open_position_without_stop_is_critical(self, base):
        _state(base, {"SXRVd_EQ": {"active_position": {"quantity": 1.0}}, "OD7Fd_EQ": {"active_position": None}})
        alerts = wd.check_positions_have_stops(base)
        assert [a.key for a in alerts] == ["no-stop:SXRVd_EQ"]
        assert alerts[0].level == "CRITICAL"

    def test_protected_position_is_fine(self, base):
        _state(base, {"SXRVd_EQ": {"active_position": {"quantity": 1.0, "stop_order_id": 55701244265}}})
        assert wd.check_positions_have_stops(base) == []

    def test_missing_or_broken_state(self, base):
        assert wd.check_positions_have_stops(base) == []
        (base / "t212_portfolio_state.json").write_text("{cassé")
        assert wd.check_positions_have_stops(base)[0].key == "state-unreadable"


class TestErrorBurst:
    def _log(self, base, n, when):
        lines = [f"{when:%Y-%m-%d %H:%M:%S},000 - ERROR - boom {i}" for i in range(n)]
        lines.append(f"{when:%Y-%m-%d %H:%M:%S},000 - INFO - ok")
        (base / "trading.log").write_text("\n".join(lines), encoding="utf-8")

    def test_burst_over_threshold(self, base):
        self._log(base, 30, NOW - dt.timedelta(minutes=10))
        assert wd.check_error_burst(base, NOW).key == "error-burst"

    def test_old_errors_do_not_count(self, base):
        self._log(base, 200, NOW - dt.timedelta(hours=5))
        assert wd.check_error_burst(base, NOW) is None

    def test_few_errors_are_fine(self, base):
        self._log(base, 5, NOW - dt.timedelta(minutes=10))
        assert wd.check_error_burst(base, NOW) is None


class TestDeduplication:
    def test_alert_is_sent_once_then_after_cooldown(self):
        a = wd.Alert("scheduler-dead", "CRITICAL", "mort")
        state: dict = {}
        first, _ = wd.select_alerts_to_send([a], state, NOW)
        again, _ = wd.select_alerts_to_send([a], state, NOW + dt.timedelta(minutes=30))
        later, _ = wd.select_alerts_to_send([a], state, NOW + dt.timedelta(minutes=wd.ALERT_COOLDOWN_MIN + 1))
        assert (len(first), len(again), len(later)) == (1, 0, 1)

    def test_resolution_is_reported_once(self):
        a = wd.Alert("no-cycle", "CRITICAL", "x")
        state: dict = {}
        wd.select_alerts_to_send([a], state, NOW)
        _, resolved = wd.select_alerts_to_send([], state, NOW + dt.timedelta(minutes=15))
        _, resolved_again = wd.select_alerts_to_send([], state, NOW + dt.timedelta(minutes=30))
        assert resolved == ["no-cycle"] and resolved_again == []


class TestRunOnce:
    def _dead_setup(self, base):
        _sched_log(base, NOW - dt.timedelta(minutes=120))
        _lock(base, pid=4242)
        return MagicMock()

    def test_dead_scheduler_alerts_and_persists_state(self, base):
        notifier_mock = self._dead_setup(base)
        res = wd.run_once(base, NOW, notifier=notifier_mock, pid_alive=DEAD)
        titles = [c.args[0] for c in notifier_mock.call_args_list]
        assert any("scheduler-dead" in t for t in titles) and any("no-cycle" in t for t in titles)
        assert (base / "watchdog_state.json").exists()
        # deuxième passage 15 min plus tard : déduplication, aucun renvoi
        notifier_mock.reset_mock()
        wd.run_once(base, NOW + dt.timedelta(minutes=15), notifier=notifier_mock, pid_alive=DEAD)
        notifier_mock.assert_not_called()
        assert res["paused"] is False

    def test_dry_run_neither_alerts_nor_writes(self, base):
        notifier_mock = self._dead_setup(base)
        res = wd.run_once(base, NOW, dry_run=True, notifier=notifier_mock, pid_alive=DEAD, restart=True)
        notifier_mock.assert_not_called()
        assert not (base / "watchdog_state.json").exists()
        assert res["alerts"] and res["action"] == ""

    def test_pause_silences_everything(self, base):
        notifier_mock = self._dead_setup(base)
        (base / "watchdog.pause").write_text("maintenance")
        launcher = MagicMock()
        res = wd.run_once(base, NOW, restart=True, notifier=notifier_mock, launcher=launcher, pid_alive=DEAD)
        assert res["paused"] is True
        notifier_mock.assert_not_called()
        launcher.assert_not_called()

    def test_healthy_system_stays_silent(self, base):
        _sched_log(base, NOW - dt.timedelta(minutes=10))
        _lock(base)
        _state(base, {"SXRVd_EQ": {"active_position": {"stop_order_id": 1}}})
        notifier_mock = MagicMock()
        res = wd.run_once(base, NOW, notifier=notifier_mock, pid_alive=ALIVE)
        assert res["alerts"] == []
        notifier_mock.assert_not_called()

    def test_status_flag_is_read_only(self, base, monkeypatch, capsys):
        """--status ne doit ni alerter, ni relancer, ni toucher watchdog_state.json (cooldown de 2 h)."""
        sent, restart = MagicMock(), MagicMock()
        monkeypatch.setattr(notifier, "notify", sent)
        monkeypatch.setattr(wd, "maybe_restart", restart)
        monkeypatch.setattr(wd.dt, "datetime", type("D", (dt.datetime,), {"now": classmethod(lambda cls, tz=None: NOW)}))
        rc = wd.main(["--status", "--restart", "--base", str(base)])  # aucun verrou : scheduler « mort »
        assert rc == 1 and "scheduler-dead" in capsys.readouterr().out
        sent.assert_not_called()
        restart.assert_not_called()
        assert not (base / "watchdog_state.json").exists()


class TestRestart:
    def test_restart_removes_orphan_lock_and_launches(self, base):
        _sched_log(base, NOW - dt.timedelta(minutes=120))
        _lock(base, pid=4242)
        launcher = MagicMock()
        res = wd.run_once(base, NOW, restart=True, notifier=MagicMock(), launcher=launcher, pid_alive=DEAD)
        launcher.assert_called_once_with(base)
        assert not (base / "scheduler.lock").exists()
        assert "relancé" in res["action"]

    def test_never_restarts_over_a_live_scheduler(self, base):
        _lock(base, pid=4242, age_sec=10)
        launcher = MagicMock()
        action = wd.maybe_restart(base, {}, NOW, launcher, ALIVE)
        launcher.assert_not_called()
        assert action == ""
        assert (base / "scheduler.lock").exists()

    def test_restart_rate_limit_stops_crash_loops(self, base):
        launcher = MagicMock()
        state: dict = {}
        for i in range(wd.MAX_RESTARTS_PER_HOUR):
            wd.maybe_restart(base, state, NOW + dt.timedelta(minutes=i), launcher, DEAD)
        action = wd.maybe_restart(base, state, NOW + dt.timedelta(minutes=10), launcher, DEAD)
        assert launcher.call_count == wd.MAX_RESTARTS_PER_HOUR
        assert "ignorée" in action

    def test_restart_window_expires_after_an_hour(self, base):
        launcher = MagicMock()
        state = {"restarts": [(NOW - dt.timedelta(minutes=90)).isoformat()] * 3}
        wd.maybe_restart(base, state, NOW, launcher, DEAD)
        launcher.assert_called_once()

    def test_no_lock_at_all_still_restarts(self, base):
        launcher = MagicMock()
        wd.maybe_restart(base, {}, NOW, launcher, DEAD)
        launcher.assert_called_once_with(base)


class TestNotifier:
    def test_no_channel_never_raises(self, monkeypatch):
        for var in ("NTFY_TOPIC", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            monkeypatch.delenv(var, raising=False)
        assert notifier.configured_channels() == []
        assert notifier.notify("t", "m", "CRITICAL") == []

    def test_ntfy_delivery(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "mon-topic")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        with patch("src.notifier.requests.post", return_value=MagicMock(ok=True)) as post:
            delivered = notifier.notify("Titre é", "corps", "CRITICAL")
        assert delivered == ["ntfy"]
        assert post.call_args.args[0] == "https://ntfy.sh/mon-topic"
        assert post.call_args.kwargs["headers"]["Priority"] == "urgent"

    def test_ntfy_title_is_ascii_header_safe(self, monkeypatch):
        """Titre non ASCII : RFC 2047 (jamais d'octets bruts dans l'en-tête HTTP)."""
        import base64

        monkeypatch.setenv("NTFY_TOPIC", "t")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        with patch("src.notifier.requests.post", return_value=MagicMock(ok=True)) as post:
            notifier.notify("Trading-AI : résolu ✅", "m", "INFO")
        title = post.call_args.kwargs["headers"]["Title"]
        assert isinstance(title, str) and title.isascii()
        assert base64.b64decode(title[len("=?UTF-8?B?"):-len("?=")]).decode("utf-8") == "Trading-AI : résolu ✅"
        with patch("src.notifier.requests.post", return_value=MagicMock(ok=True)) as post:
            notifier.notify("ascii only", "m", "INFO")
        assert post.call_args.kwargs["headers"]["Title"] == "ascii only"

    def test_unexpected_send_error_never_escapes(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "t")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:X")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        with patch("src.notifier.requests.post", side_effect=UnicodeEncodeError("latin-1", "é", 0, 1, "boom")):
            assert notifier.notify("t", "m") == []

    def test_telegram_failure_does_not_leak_the_token(self, monkeypatch, caplog):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET-TOKEN")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        with patch("src.notifier.requests.post", side_effect=requests.ConnectionError("https://api.telegram.org/bot123:SECRET-TOKEN")):
            with caplog.at_level("WARNING"):
                delivered = notifier.notify("t", "m")
        assert delivered == []
        assert "SECRET-TOKEN" not in caplog.text


class TestNextcloudTalk:
    ENV = {
        "NEXTCLOUD_URL": "https://cloud.example.org/",
        "NEXTCLOUD_USER": "bot",
        "NEXTCLOUD_PASSWORD": "S3CRET-PASS",
        "NEXTCLOUD_TALK_TOKEN": "abc123",
    }

    def _setenv(self, monkeypatch, **overrides):
        for var in ("NTFY_TOPIC", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            monkeypatch.delenv(var, raising=False)
        for k, v in {**self.ENV, **overrides}.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)

    def test_channel_needs_all_four_variables(self, monkeypatch):
        self._setenv(monkeypatch)
        assert notifier.configured_channels() == ["nextcloud"]
        self._setenv(monkeypatch, NEXTCLOUD_TALK_TOKEN=None)
        assert notifier.configured_channels() == []

    def test_message_is_posted_to_the_talk_chat_endpoint(self, monkeypatch):
        self._setenv(monkeypatch)
        with patch("src.notifier.requests.post", return_value=MagicMock(ok=True)) as post:
            delivered = notifier.notify("Trading-AI : scheduler-dead", "arrêt", "CRITICAL")
        assert delivered == ["nextcloud"]
        assert post.call_args.args[0] == "https://cloud.example.org/ocs/v2.php/apps/spreed/api/v1/chat/abc123"
        kw = post.call_args.kwargs
        assert kw["auth"] == ("bot", "S3CRET-PASS") and kw["headers"]["OCS-APIRequest"] == "true"
        assert "scheduler-dead" in kw["json"]["message"] and "arrêt" in kw["json"]["message"]

    def test_failure_never_leaks_credentials(self, monkeypatch, caplog):
        self._setenv(monkeypatch)
        boom = requests.ConnectionError("https://bot:S3CRET-PASS@cloud.example.org")
        with patch("src.notifier.requests.post", side_effect=boom), caplog.at_level("WARNING"):
            assert notifier.notify("t", "m") == []
        assert "S3CRET-PASS" not in caplog.text

    def test_http_error_is_not_delivery(self, monkeypatch):
        self._setenv(monkeypatch)
        with patch("src.notifier.requests.post", return_value=MagicMock(ok=False, status_code=401)):
            assert notifier.notify("t", "m") == []


def test_nextcloud_refuses_plain_http(monkeypatch):
    for k, v in TestNextcloudTalk.ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("NEXTCLOUD_URL", "http://cloud.example.org")
    for var in ("NTFY_TOPIC", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(var, raising=False)
    with patch("src.notifier.requests.post") as post:
        assert notifier.notify("t", "m") == []
    post.assert_not_called()

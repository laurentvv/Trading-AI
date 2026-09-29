"""Régression 2026-09-29 : FinAcumen doit tourner après CHAQUE Morning Brief.

Le commit 752ecb8 (timeouts subprocess) avait déplacé le bloc FinAcumen dans le
``except subprocess.TimeoutExpired`` de ``run_morning_brief`` : depuis le 25/09, plus aucune
analyse FinAcumen n'était ajoutée au brief (elle ne tournait qu'en cas de timeout du brief).
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import schedule  # noqa: E402


def _install_fake_run(monkeypatch, brief_behaviour):
    """Remplace subprocess.run : `brief_behaviour` pilote le Morning Brief, FinAcumen réussit."""
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        if "morning_brief/morning_brief.py" in cmd:
            if brief_behaviour == "timeout":
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))
            if brief_behaviour == "crash":
                raise OSError("boom")
            return SimpleNamespace(returncode=0 if brief_behaviour == "ok" else 1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(schedule.subprocess, "run", fake_run)
    return calls


def _finacumen_calls(calls):
    return [c for c in calls if "src/finacumen_main.py" in c]


def test_finacumen_runs_after_successful_brief(monkeypatch):
    calls = _install_fake_run(monkeypatch, "ok")
    schedule.run_morning_brief()
    assert len(_finacumen_calls(calls)) == len(schedule.TICKERS)


def test_finacumen_runs_when_brief_times_out(monkeypatch):
    calls = _install_fake_run(monkeypatch, "timeout")
    schedule.run_morning_brief()
    assert len(_finacumen_calls(calls)) == len(schedule.TICKERS)


def test_finacumen_runs_when_brief_fails_or_crashes(monkeypatch):
    for behaviour in ("fail", "crash"):
        calls = _install_fake_run(monkeypatch, behaviour)
        schedule.run_morning_brief()
        assert len(_finacumen_calls(calls)) == len(schedule.TICKERS), behaviour


def test_finacumen_section_is_appended_to_the_brief(monkeypatch, tmp_path):
    _install_fake_run(monkeypatch, "ok")
    brief = tmp_path / "morning_brief" / "output" / "morning_market_brief.md"
    brief.parent.mkdir(parents=True)
    brief.write_text("# Brief du jour\n", encoding="utf-8")
    state_dir = tmp_path / "data_cache" / "finacumen"
    state_dir.mkdir(parents=True)
    first = schedule.TICKERS[0]
    (state_dir / f"finacumen_{first}.json").write_text(
        json.dumps({"signal": "BUY", "confidence": 0.85, "analysis": "prix réels cités"}), encoding="utf-8"
    )

    schedule.run_morning_brief()

    text = brief.read_text(encoding="utf-8")
    assert text.startswith("# Brief du jour")
    assert "## 5. Analyse Qualitative Profonde (FinAcumen)" in text
    assert f"### {first}" in text and "BUY (Confiance: 0.85)" in text
    # Ticker sans résultat : signalé, pas d'exception.
    assert "Résultat non généré" in text


def test_stub_brief_created_when_brief_never_generated(monkeypatch, tmp_path):
    _install_fake_run(monkeypatch, "timeout")
    schedule.run_morning_brief()
    brief = tmp_path / "morning_brief" / "output" / "morning_market_brief.md"
    assert brief.exists()
    assert "Morning Brief de base non généré" in brief.read_text(encoding="utf-8")

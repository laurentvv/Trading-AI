"""get_morning_brief_context : brief du jour, sinon repli sur finacumen_daily.md, jamais un fichier périmé."""

import os
import time
from pathlib import Path

from src import llm_client


def _write(name: str, text: str, age_hours: float = 0.0) -> Path:
    path = Path("morning_brief/output") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    ts = time.time() - age_hours * 3600
    os.utime(path, (ts, ts))
    return path


def test_no_file_gives_empty_context():
    assert llm_client.get_morning_brief_context() == ""


def test_fresh_brief_is_served():
    _write("morning_market_brief.md", "BRIEF DU JOUR")
    assert "BRIEF DU JOUR" in llm_client.get_morning_brief_context()


def test_finacumen_fallback_when_the_brief_is_missing():
    _write("finacumen_daily.md", "FINACUMEN SEUL")
    assert "FINACUMEN SEUL" in llm_client.get_morning_brief_context()


def test_most_recent_fresh_file_wins():
    _write("morning_market_brief.md", "BRIEF DE LA VEILLE", age_hours=20)
    _write("finacumen_daily.md", "FINACUMEN DU JOUR", age_hours=1)
    ctx = llm_client.get_morning_brief_context()
    assert "FINACUMEN DU JOUR" in ctx and "BRIEF DE LA VEILLE" not in ctx


def test_stale_files_are_never_served():
    _write("morning_market_brief.md", "VIEUX BRIEF", age_hours=30)
    _write("finacumen_daily.md", "VIEUX FINACUMEN", age_hours=48)
    assert llm_client.get_morning_brief_context() == ""

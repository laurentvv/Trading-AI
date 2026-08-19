import pytest
from datetime import datetime, timedelta
from pathlib import Path
from morning_brief.tools.analyze_trading_logs import AnalyzeTradingLogsTool

def test_analyze_trading_logs_ignores_benign_patterns(tmp_path):
    log_file = tmp_path / "test_trading.log"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sample_log = f"""
{now_str},100 - INFO - Début de la recherche Web Macro (timeout global 30s)...
{now_str},200 - INFO - Querying textual LLM for decision (timeout 240s, autres modèles en parallèle)...
{now_str},300 - WARNING - findfont: Failed to find font weight semibold, now using 700.
{now_str},400 - WARNING - Warning: You are sending unauthenticated requests to the HF Hub.
{now_str},500 - WARNING - EIA crude_imports payload refused: 3 rows, latest period 2026-05-01 (110d old, stale content). Not cached; will retry next cycle.
{now_str},600 - WARNING - Réduction forte de grebenkov : win_rate 22.77% → facteur 0.00 (sous le plancher 25%).
{now_str},700 - WARNING - Cache stale: last data date is 2026-08-18 (1.4 days old), refreshing...
{now_str},800 - ERROR - Error during DuckDuckGo sync search: No results found.
{now_str},900 - INFO - Analysis completed successfully.
"""
    log_file.write_text(sample_log.strip(), encoding="utf-8")

    tool = AnalyzeTradingLogsTool()
    result = tool.forward(log_path=str(log_file), hours_back=24)

    assert "Health: 100/100" in result
    assert "Errors: 0" in result
    assert "Warnings: 0" in result
    assert "API disconnects: 0" in result

def test_analyze_trading_logs_catches_real_errors_and_disconnects(tmp_path):
    log_file = tmp_path / "test_trading_errors.log"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sample_log = f"""
{now_str},100 - ERROR - Database disk image is malformed
{now_str},200 - WARNING - FallbackGateway: provider 'gemini_free' failed (Request to provider 'gemini_free' timed out after 60.0s.). Trying next...
{now_str},300 - WARNING - Connection reset by peer from broker API
{now_str},400 - INFO - Trading 212 order executed with slippage: -0.05%
"""
    log_file.write_text(sample_log.strip(), encoding="utf-8")

    tool = AnalyzeTradingLogsTool()
    result = tool.forward(log_path=str(log_file), hours_back=24)

    assert "Errors: 1" in result
    assert "Warnings: 2" in result
    assert "API disconnects: 2" in result
    assert "Slippage: 1" in result
    assert "Health: 71/100" in result

def test_analyze_trading_logs_time_window_filtering(tmp_path):
    log_file = tmp_path / "test_trading_history.log"
    old_time = (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")
    recent_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sample_log = f"""
{old_time},100 - ERROR - Old critical failure that happened 2 days ago
{old_time},200 - WARNING - Old connection refused from 2 days ago
{recent_time},100 - INFO - Clean cycle executed today
"""
    log_file.write_text(sample_log.strip(), encoding="utf-8")

    tool = AnalyzeTradingLogsTool()
    result = tool.forward(log_path=str(log_file), hours_back=24)

    assert "Health: 100/100" in result
    assert "Errors: 0" in result
    assert "Warnings: 0" in result
    assert "API disconnects: 0" in result

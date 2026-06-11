from smolagents import Tool


def _resolve_db_path(db_path: str | None) -> str:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if db_path:
        resolved = Path(db_path).resolve()
        if not resolved.is_relative_to(project_root):
            raise ValueError("db_path must be within the project directory.")
        return str(resolved)
    return str(project_root / "performance_monitor.db")


def _fetch_ticker_metrics(cursor, tables) -> list[dict]:
    if "realtime_metrics" not in tables:
        return []

    cursor.execute(
        "SELECT DISTINCT ticker FROM realtime_metrics WHERE ticker IS NOT NULL"
    )
    tickers = [row[0] for row in cursor.fetchall()]

    results = []
    for ticker in tickers:
        cursor.execute(
            """
            SELECT daily_return, total_trades, max_drawdown,
                   portfolio_value, active_positions
            FROM realtime_metrics
            WHERE ticker = ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (ticker,),
        )
        row = cursor.fetchone()
        if row:
            results.append(
                {
                    "ticker": ticker,
                    "pnl": round(row[0], 4) if row[0] is not None else 0.0,
                    "trades": row[1] or 0,
                    "drawdown": round(row[2], 4) if row[2] is not None else 0.0,
                    "portfolio_value": round(row[3], 2) if row[3] is not None else 0.0,
                    "active_positions": row[4] or 0,
                    "alerts": 0,
                }
            )
    return results


def _apply_alert_counts(cursor, tables, tickers_result: list[dict]):
    from datetime import datetime, timedelta

    if "performance_alerts" not in tables:
        return

    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    cursor.execute(
        """
        SELECT ticker, COUNT(*)
        FROM performance_alerts
        WHERE acknowledged = 0 AND timestamp > ?
        GROUP BY ticker
        """,
        (cutoff,),
    )
    alert_counts = dict(cursor.fetchall())
    for t in tickers_result:
        t["alerts"] = alert_counts.get(t["ticker"], 0)


def _compute_overall_health(tickers_result: list[dict]) -> str:
    if not tickers_result:
        return "HEALTHY"

    max_dd = max(t["drawdown"] for t in tickers_result)
    total_alerts = sum(t["alerts"] for t in tickers_result)

    if max_dd > 0.10 or total_alerts > 10:
        return "CRITICAL"
    if max_dd > 0.05 or total_alerts > 5:
        return "WARNING"
    return "HEALTHY"


def _format_summary(tickers_result: list[dict], overall: str, max_dd: float) -> str:
    parts = [
        f"{t['ticker']}: PnL={t['pnl']:+.2%} DD={t['drawdown']:.2%}"
        for t in tickers_result
    ]
    return f"{overall} | MaxDD={max_dd:.2%} | " + " | ".join(parts)


class AuditPortfolioPerformanceTool(Tool):
    name = "audit_portfolio_performance"
    description = (
        "Queries the Trading-AI performance_monitor.db SQLite database for portfolio "
        "metrics per ticker: PnL, trade count, max drawdown, and unacknowledged alerts. "
        "Returns a compact summary string. Full data saved to output/tools/."
    )
    inputs = {
        "db_path": {
            "type": "string",
            "nullable": True,
            "description": (
                "Path to performance_monitor.db. "
                "Defaults to the project root performance_monitor.db if not specified."
            ),
        }
    }
    output_type = "string"

    def forward(self, db_path: str | None = None) -> str:
        import sqlite3
        from pathlib import Path
        from morning_brief.tools import save_tool_result

        db_file = Path(_resolve_db_path(db_path))
        if not db_file.exists():
            result = {"status": "NO_DATA", "tickers": [], "overall_health": "UNKNOWN"}
            save_tool_result("portfolio", result)
            return "NO_DATA: database not found."

        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=10)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            tickers_result = _fetch_ticker_metrics(cursor, tables)
            _apply_alert_counts(cursor, tables, tickers_result)

            overall = _compute_overall_health(tickers_result)
            max_dd = max((t["drawdown"] for t in tickers_result), default=0.0)

            result = {
                "status": "OK",
                "tickers": tickers_result,
                "overall_health": overall,
                "max_drawdown_global": round(max_dd, 4),
            }
            save_tool_result("portfolio", result)

            return _format_summary(tickers_result, overall, max_dd)

        except Exception as e:
            result = {"status": "ERROR", "tickers": [], "overall_health": "ERROR", "error": str(e)}
            save_tool_result("portfolio", result)
            return f"ERROR: {e}"
        finally:
            if conn is not None:
                conn.close()

from smolagents import Tool


class AnalyzeTradingLogsTool(Tool):
    name = "analyze_trading_logs"
    description = (
        "Analyzes the Trading-AI system log file for errors, warnings, "
        "API disconnects, slippage events, and model errors. "
        "Returns a compact summary string. Full data saved to output/tools/."
    )
    inputs = {
        "log_path": {
            "type": "string",
            "nullable": True,
            "description": (
                "Path to the trading.log file. "
                "Defaults to the project root trading.log if not specified."
            ),
        }
    }
    output_type = "string"

    def forward(self, log_path: str | None = None) -> str:
        import re
        from pathlib import Path
        from morning_brief.tools import save_tool_result

        project_root = Path(__file__).resolve().parents[2]

        if not log_path:
            log_path = str(project_root / "trading.log")

        log_file = Path(log_path).resolve()
        if not log_file.is_relative_to(project_root):
            return "ERROR: log_path must be within the project directory."
        if not log_file.exists() or log_file.stat().st_size == 0:
            result = {
                "status": "NO_DATA",
                "errors": [],
                "warnings": [],
                "api_disconnects": 0,
                "slippage_events": [],
                "health_score": 50,
            }
            save_tool_result("trading_logs", result)
            return "NO_DATA: log file not found or empty."

        content = log_file.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        errors = []
        warnings = []
        api_disconnects = 0
        slippage_events = []

        for line in lines:
            if re.search(r"\bERROR\b|\bCRITICAL\b", line, re.IGNORECASE):
                errors.append(line.strip()[-200:])
            elif re.search(r"\bWARNING\b", line, re.IGNORECASE):
                warnings.append(line.strip()[-200:])

            if re.search(
                r"circuit.?breaker|timeout|connection.*(fail|reset|refused|lost)|FRED.*failed",
                line,
                re.IGNORECASE,
            ):
                api_disconnects += 1

            if re.search(r"slippage", line, re.IGNORECASE):
                slippage_events.append(line.strip()[-200:])

        errors = errors[-20:]
        warnings = warnings[-20:]
        slippage_events = slippage_events[-10:]

        health_score = 100
        health_score -= min(len(errors) * 5, 40)
        health_score -= min(len(warnings) * 2, 20)
        health_score -= min(api_disconnects * 5, 20)
        health_score -= min(len(slippage_events) * 10, 20)
        health_score = max(0, health_score)

        result = {
            "status": "OK",
            "error_count": len(errors),
            "warning_count": len(warnings),
            "api_disconnects": api_disconnects,
            "slippage_count": len(slippage_events),
            "health_score": health_score,
            "errors_sample": errors[:5],
            "warnings_sample": warnings[:5],
            "slippage_sample": slippage_events[:3],
        }
        save_tool_result("trading_logs", result)

        return (
            f"Health: {health_score}/100 | "
            f"Errors: {len(errors)} | Warnings: {len(warnings)} | "
            f"API disconnects: {api_disconnects} | Slippage: {len(slippage_events)}"
        )

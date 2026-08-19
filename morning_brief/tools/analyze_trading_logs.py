from morning_brief.tools.base import BaseTool as Tool


class AnalyzeTradingLogsTool(Tool):
    name = "analyze_trading_logs"
    description = (
        "Analyzes the Trading-AI system log file for errors, warnings, "
        "API disconnects, slippage events, and model errors over a recent time window. "
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
        },
        "hours_back": {
            "type": "integer",
            "nullable": True,
            "description": (
                "Number of hours back to analyze (default: 24). "
                "Set to 0 or null to analyze the entire log file."
            ),
        },
    }
    output_type = "string"

    def forward(self, log_path: str | None = None, hours_back: int | None = 24) -> str:
        import re
        from datetime import datetime, timedelta
        from pathlib import Path
        from morning_brief.tools import save_tool_result

        import tempfile
        project_root = Path(__file__).resolve().parents[2]
        temp_dir = Path(tempfile.gettempdir()).resolve()

        if not log_path:
            log_path = str(project_root / "trading.log")

        log_file = Path(log_path).resolve()
        if not (log_file.is_relative_to(project_root) or log_file.is_relative_to(temp_dir)):
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

        # Time-window filtering
        timestamp_re = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
        if hours_back and hours_back > 0:
            cutoff = datetime.now() - timedelta(hours=hours_back)
            recent_lines = []
            in_window = True if not lines else False
            for line in lines:
                m = timestamp_re.match(line)
                if m:
                    try:
                        line_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                        in_window = (line_time >= cutoff)
                    except ValueError:
                        pass
                if in_window:
                    recent_lines.append(line)
            lines = recent_lines

        benign_warning_patterns = [
            r"findfont:",
            r"Warning: You are sending unauthenticated requests to the HF Hub",
            r"EIA crude_imports payload refused",
            r"R[eé]duction (?:forte )?de \w+ : win_rate",
            r"No results found during web research",
            r"Using DDG snippets only",
            r"MA200 not available for .*, trying MA50 fallback",
            r"Cache stale: last data date is",
            r"Failed to crawl .* Using snippet",
        ]

        benign_error_patterns = [
            r"Error during DuckDuckGo (?:sync )?search: No results found",
            r"No results found during web research",
        ]

        disconnect_patterns = [
            r"circuit.?breaker",
            r"connection.*(?:fail|reset|refused|lost|abort)",
            r"request.*timed?\s*out",
            r"timed?\s*out\s*(?:after|on)",
            r"(?:connect|read|socket)\s*timeout",
            r"FRED.*failed",
        ]

        errors = []
        warnings = []
        api_disconnects = 0
        slippage_events = []

        for line in lines:
            if re.search(r"\bERROR\b|\bCRITICAL\b", line, re.IGNORECASE):
                if not any(re.search(p, line, re.IGNORECASE) for p in benign_error_patterns):
                    errors.append(line.strip()[-200:])
            elif re.search(r"\bWARNING\b", line, re.IGNORECASE):
                if not any(re.search(p, line, re.IGNORECASE) for p in benign_warning_patterns):
                    warnings.append(line.strip()[-200:])

            # Real disconnects / connection timeouts (excluding informational parameters like "(timeout 240s...)")
            if not re.search(r"\(timeout\s+\d+s|timeout\s*=\s*\d+|timeout\s+global\s+\d+s", line, re.IGNORECASE):
                if any(re.search(p, line, re.IGNORECASE) for p in disconnect_patterns):
                    api_disconnects += 1

            if re.search(r"\bslippage\b", line, re.IGNORECASE):
                slippage_events.append(line.strip()[-200:])

        errors_sample = errors[-20:]
        warnings_sample = warnings[-20:]
        slippage_sample = slippage_events[-10:]

        health_score = 100
        health_score -= min(len(errors_sample) * 5, 40)
        health_score -= min(len(warnings_sample) * 2, 20)
        health_score -= min(api_disconnects * 5, 20)
        health_score -= min(len(slippage_sample) * 10, 20)
        health_score = max(0, health_score)

        result = {
            "status": "OK",
            "error_count": len(errors_sample),
            "warning_count": len(warnings_sample),
            "api_disconnects": api_disconnects,
            "slippage_count": len(slippage_sample),
            "health_score": health_score,
            "errors_sample": errors_sample[:5],
            "warnings_sample": warnings_sample[:5],
            "slippage_sample": slippage_sample[:3],
        }
        save_tool_result("trading_logs", result)

        return (
            f"Health: {health_score}/100 | "
            f"Errors: {len(errors_sample)} | Warnings: {len(warnings_sample)} | "
            f"API disconnects: {api_disconnects} | Slippage: {len(slippage_sample)}"
        )


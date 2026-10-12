import re
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Secrets carried in a URL query string. httpx logs every request URL at INFO, and the Gemini
# REST API takes its key as `?key=...`: from August to October 2026, ~1 100 lines of trading.log*
# held the gemini_free key in clear, and an Alpha Vantage `apikey=` leaked through the news stderr.
_QUERY_SECRET_RE = re.compile(r"(?i)([?&](?:key|apikey|api_key|token|access_token)=)[^&\s\"'#]+")
# Secrets quoted in free text ("your API key as XXXX", "token: XXXX").
_TEXT_SECRET_RE = re.compile(
    r"(?i)\b(key|token|apikey|api_key|secret|password)\b(\s+(?:as|is|:|=))?\s+([A-Za-z0-9_\-]{12,})"
)


def redact_secrets(text: str) -> str:
    """Masks API keys and tokens (URL parameters and free text) before they reach a log."""
    if not text:
        return text or ""
    text = _QUERY_SECRET_RE.sub(r"\1***REDACTED***", text)
    return _TEXT_SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2) or ''} ***REDACTED***", text)


class SecretRedactingFilter(logging.Filter):
    """Handler filter: redacts the formatted message and the traceback of every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg, record.args = redacted, None
        if record.exc_info and not record.exc_text:
            record.exc_text = redact_secrets(logging.Formatter().formatException(record.exc_info))
        return True


def setup_environment(log_file="trading.log"):
    """
    Configures the environment for the Trading AI application.
    Forces UTF-8 encoding for stdout on Windows to support emojis and sets up standard logging.
    """
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, Exception):
            pass

    log_path = Path(log_file)
    if log_path.parent != Path("."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    for handler in handlers:
        handler.addFilter(SecretRedactingFilter())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

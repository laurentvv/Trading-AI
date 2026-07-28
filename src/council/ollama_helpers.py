"""Shared helpers for Ollama model introspection.

Both `setup_council_models.py` (PROD bootstrap) and
`src/council/weekend_council.py` (runtime) need to check whether a given model
is installed in the local Ollama daemon. The two copies had drifted only in the
base URL (hardcoded vs. module constant) — this shared helper removes the
duplicate-code finding (python-health-audit 2026-07-28).
"""

from __future__ import annotations

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


def is_model_installed(model: str, base_url: str = OLLAMA_DEFAULT_BASE_URL) -> bool:
    """Return True if ``model`` is present in the local Ollama daemon.

    Resilient by design: any connection error, non-200 response, or malformed
    JSON yields ``False`` (so callers treat the model as unavailable and fall
    back, rather than crashing the whole pipeline on a transient Ollama hiccup).
    """
    try:
        import requests

        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        installed = {m.get("name", "") for m in resp.json().get("models", [])}
        return model in installed
    except Exception:
        return False

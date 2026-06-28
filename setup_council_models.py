#!/usr/bin/env python3
"""
Installs the local Ollama models required by the Weekend Council.

The council's value comes from each persona running on a DISTINCT model
lineage. Some members run on the cloud (Google Gemini, prefixed ``gemini:``)
and need NO local install — only a ``GEMINI_API_KEY`` / ``GEMINI_API_KEY_PAY``
in ``.env`` (see .env.example). This script installs the remaining LOCAL
(Ollama) families (Gemma / GLM / Qwen / Mistral). If a local model is missing,
that member silently falls back to the canonical Gemma default — the council
still produces a report, but with reduced reasoning diversity.

Run after a fresh install or on a new PROD server:
    uv run python setup_council_models.py

Each Ollama model is ~5-10 GB; total download is ~20 GB. Already-installed
models are skipped (idempotent).
"""

import subprocess
import sys

from src.council.council_prompts import JUDGE_MODEL, MEMBER_MODELS

# The canonical default model is also required as the universal fallback.
from src.llm_client import TEXT_LLM_MODEL

# Models routed to the cloud (prefixed "gemini:") are never installed locally —
# they hit the Google Gemini API via GeminiGateway and only need an API key.
CLOUD_PREFIX = "gemini:"


def model_installed(model: str) -> bool:
    """Checks whether a model is already present in the local Ollama."""
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        installed = {m.get("name", "") for m in resp.json().get("models", [])}
        return model in installed
    except Exception:
        return False


def main() -> int:
    # Deduplicate while preserving a stable order (default first, then members,
    # then the judge which usually equals the default).
    all_models: list[str] = []
    for model in [TEXT_LLM_MODEL, *MEMBER_MODELS.values(), JUDGE_MODEL]:
        if model not in all_models:
            all_models.append(model)

    # Cloud models (gemini:*) hit the Gemini API via GeminiGateway — they need
    # an API key, not a local download. Separate them so we only `ollama pull`
    # the local ones.
    cloud_models = [m for m in all_models if m.startswith(CLOUD_PREFIX)]
    required = [m for m in all_models if not m.startswith(CLOUD_PREFIX)]

    print("=" * 70)
    print("Weekend Council — installation des modèles Ollama")
    print("=" * 70)
    if cloud_models:
        print(f"\n☁  Modèles cloud (Gemini API, pas de pull local) :")
        for m in cloud_models:
            tier = "payante" if "pro" in m.lower() else "gratuite"
            print(f"     - {m}  (clé {tier} via .env)")
        print("   Configure GEMINI_API_KEY (membres) et GEMINI_API_KEY_PAY (Juge)")
        print("   dans .env — voir .env.example pour le détail.")
    print(f"\nModèles Ollama requis ({len(required)}, ~20 GB total au pire) :\n")
    for i, model in enumerate(required, 1):
        print(f"  {i}. {model}")
    print()

    missing = [m for m in required if not model_installed(m)]
    if not missing:
        print("✅ Tous les modèles du council sont déjà installés.")
        return 0

    print(f"⬇ {len(missing)} modèle(s) à télécharger : {', '.join(missing)}\n")

    # Check Ollama is running before pulling.
    try:
        import requests

        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("❌ Ollama n'est pas accessible sur localhost:11434.")
        print("   Démarrez-le avec `ollama serve` puis relancez ce script.")
        return 1

    for model in missing:
        print(f"\n⬇ Pulling {model} ...")
        result = subprocess.run(["ollama", "pull", model])
        if result.returncode != 0:
            print(f"❌ Échec du téléchargement de {model} (code {result.returncode})")
            return result.returncode
        print(f"✅ {model} installé.")

    print("\n🎉 Tous les modèles Ollama du council sont prêts.")
    print("   Diversité locale (Gemma / GLM / Qwen / Mistral) + cloud (Gemini) active.")
    if cloud_models:
        print("   Rappel : vérifie aussi les clés Gemini dans .env "
              "(GEMINI_API_KEY, GEMINI_API_KEY_PAY).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

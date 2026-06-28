#!/usr/bin/env python3
"""
Installs the four Ollama model families required by the Weekend Council.

The council's value comes from each persona running on a DISTINCT model
lineage (Gemma / GLM / Qwen / LFM). If a model is missing, that member
silently falls back to the canonical Gemma default — the council still
produces a report, but with reduced reasoning diversity. This script
ensures all four families are present.

Run after a fresh install or on a new PROD server:
    uv run python setup_council_models.py

Each model is ~5-10 GB; total download is ~30 GB. Already-installed models
are skipped (idempotent).
"""

import subprocess
import sys

from src.council.council_prompts import JUDGE_MODEL, MEMBER_MODELS

# The canonical default model is also required as the universal fallback.
from src.llm_client import TEXT_LLM_MODEL


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
    required: list[str] = []
    for model in [TEXT_LLM_MODEL, *MEMBER_MODELS.values(), JUDGE_MODEL]:
        if model not in required:
            required.append(model)

    print("=" * 70)
    print("Weekend Council — installation des modèles Ollama")
    print("=" * 70)
    print(f"Modèles requis ({len(required)}, ~30 GB total au pire) :\n")
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

    print("\n🎉 Tous les modèles du council sont prêts.")
    print("   La diversité de raisonnement (Gemma / GLM / Qwen / LFM) est active.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Smoke test TimesFM 3.0 — téléchargement réel + inférence + timing CPU.

À lancer UNE FOIS PAR MACHINE avant le premier cycle du scheduler :
le 1er appel télécharge ~1,3 Go (config.json + model.safetensors) dans le
cache HF (%USERPROFILE%\\.cache\\huggingface\\hub). La tâche TimesFM d'un cycle
a un timeout de 180 s : un téléchargement à froid PENDANT le cycle donnerait
HOLD 0.0 (d'où ce pré-chauffage).

    uv run python tests/smoke_timesfm3.py

Vérifie :
  1. chargement `TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")`
     (device auto : cuda si dispo, sinon cpu),
  2. shapes API 3.0 : forecast (horizon,), quantiles (horizon, 9),
  3. temps d'inférence à contexte 2048 (TIMESFM_CONTEXT du wrapper) et 1024.

Garde-fou migration : si l'inférence à 2048 dépasse ~60 s, repasser
TIMESFM_CONTEXT à 1024 dans src/timesfm_model.py (timeout cycle : 180 s).

Si le téléchargement renvoie 401/403 : le modèle est peut-être « gated » —
accepter la licence sur https://huggingface.co/google/timesfm-3.0-pytorch
puis définir HF_TOKEN (.env ou variable d'environnement).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
            print("[1/4] HF_TOKEN détecté — authentification HuggingFace effectuée.")
        except Exception as e:  # pragma: no cover
            print(f"[!] Authentification HF échouée (non bloquant) : {e}")
    else:
        print("[1/4] HF_TOKEN non défini — téléchargement sans authentification.")

    print("[2/4] Chargement de TimesFM 3.0 (google/timesfm-3.0-pytorch)...")
    print("      (1er lancement : téléchargement ~1,3 Go — patienter sans interrompre)")
    t0 = time.perf_counter()
    try:
        from timesfm3 import TimesFM3Forecaster

        forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
    except Exception as e:
        print(f"[ÉCHEC] Chargement impossible : {e}")
        print("        401/403 => modèle gated : accepter la licence sur la page HF + HF_TOKEN.")
        return 1
    print(f"      OK en {time.perf_counter() - t0:.1f} s "
          f"(device={getattr(forecaster, 'device', 'auto')}).")

    # Série synthétique : random walk + saisonnalité, comme une daily de prix.
    rng = np.random.default_rng(42)
    n = 2048
    t = np.arange(n)
    series = 100.0 + np.cumsum(rng.normal(0, 1.0, n)) + 3.0 * np.sin(2 * np.pi * t / 21.0)

    print("[3/4] Inférence horizon=5, return_quantiles=True...")
    try:
        out = forecaster.predict(series[-2048:], horizon=5, return_quantiles=True, make_positive=True)
        forecast = np.asarray(out.forecast)
        quantiles = np.asarray(out.quantiles)
    except Exception as e:
        print(f"[ÉCHEC] Inférence : {e}")
        return 1

    assert forecast.shape == (5,), f"forecast.shape={forecast.shape}, attendu (5,)"
    assert quantiles.shape == (5, 9), f"quantiles.shape={quantiles.shape}, attendu (5, 9)"
    print(f"      forecast (médiane)={np.round(forecast, 2).tolist()}")
    print(f"      P10 (dernier pas)={quantiles[-1, 0]:.2f}  P50={quantiles[-1, 4]:.2f}  "
          f"P90={quantiles[-1, 8]:.2f}")

    print("[4/4] Timing CPU (2 passes par contexte, hors chargement)...")
    ok = True
    for ctx in (2048, 1024):
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            forecaster.predict(series[-ctx:], horizon=5)
            times.append(time.perf_counter() - t0)
        best = min(times)
        print(f"      contexte {ctx:4d} : {best:6.2f} s")
        if ctx == 2048 and best > 60.0:
            ok = False
    if not ok:
        print("[ATTENTION] Contexte 2048 > 60 s : repasser TIMESFM_CONTEXT à 1024")
        print("            dans src/timesfm_model.py (timeout tâche cycle : 180 s).")

    print("[PASS] TimesFM 3.0 opérationnel — le cache HF est prêt pour le scheduler."
          if ok else "[PASS avec réserve] Voir l'attention ci-dessus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

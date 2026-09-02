# Plan de migration PROD — TimesFM 3.0 + reset des données (run 2 démo)

**Date :** 2026-09-02 — **Statut : prêt à exécuter** (après validation DEV et push sur `main`)
**Contexte :** migration TimesFM 2.5 → 3.0 (`google/timesfm-3.0-pytorch`, package PyPI
`timesfm>=3.0.1`). Le run démo 1 (démarré 2026-08-19) est compromises : 3 ordres en
13 jours (critère ≥ 20 round-trips inatteignable), incidents corrigés en cours de
route, scheduler arrêté depuis le 01/09 14:42 pour l'audit J+13. Décision : reset
complet et **run 2 de 30 jours avec TimesFM 3.0**, toujours en démo.

> ⚠️ **Licence** : les poids TimesFM 3.0 sont sous `timesfm-non-commercial-license-v1.0`
> (usage non-commercial ; le README Google précise « non-production »). Les versions
> ≤ 2.5 sont Apache-2.0. Usage personnel/démo accepté par le propriétaire — à
> reconsidérer avant toute exploitation commerciale.

> ⚠️ **Ce plan s'exécute sur la machine PROD** (pas le poste DEV). PROD reçoit le
> code par `git pull` sur `main` ; le scheduler y est lancé **depuis `logs_prod/`**.

---

## 0. Pré-requis côté DEV (déjà faits au 2026-09-02)

- [x] Code migré sur la branche `TimesFM` : wrapper `src/timesfm_model.py` (API
      `timesfm3.TimesFM3Forecaster`, contexte 2048, médiane, retry-init), fin du
      vendoring (`setup_timesfm.py`/`.gitmodules`/`[tool.uv.sources]` supprimés),
      `pyproject.toml` → `timesfm>=3.0.1,<4`, `check_setup()` → `find_spec("timesfm3")`.
- [x] `reset_for_fresh_test.py` étendu : détection des artefacts runtime dans
      `logs_prod/` + `--include-logs-prod` / `--keep-logs-prod` (trou de l'incident
      2026-08-24 : `logs_prod/model_performance.db` pré-reset avait survécu au reset
      du 19/08 et pollué les poids adaptatifs pendant 5 jours).
- [x] Suite mockée : 282 passed / 3 skipped.
- [ ] Smoke DEV : `uv run python tests/smoke_timesfm3.py` (download ~1,3 Go + timing).
- [ ] Merge `TimesFM` → `main` **et push** (accord explicite requis — règle AGENTS.md §5).

## 1. Arrêter proprement le scheduler (PROD)

Le scheduler est arrêté depuis le 01/09 14:42 (audit J+13). Si une instance tournait :

1. Ctrl+C dans la fenêtre `start_scheduler.bat` (exit 0 → le superviseur s'arrête).
2. Vérifier qu'aucun `python.exe` résiduel ne tourne (Task Manager).
3. `logs_prod\scheduler.lock` : périmé au bout de 15 min (auto-récupération) ; le
   supprimer manuellement si un redémarrage immédiat est nécessaire.

## 2. Reset du compte démo Trading 212 (app mobile/web)

**Étape clé, à faire AVANT le reset local.** Le reset du compte démo dans l'app T212 :

- annule la position SXRV ouverte (achat 28/08 @1458.78) et son stop GTC standing
  (vérifié le 19/08 : « ancienne position annulée par le reset compte »),
- remet le cash virtuel à zéro,
- **vide l'historique d'ordres** → sans ça, le realized P&L FIFO de l'ancienne run
  (recalculé depuis l'historique broker) serait hérité dans l'equity du run 2.

## 3. Déployer le code (PROD)

```powershell
# depuis la racine du repo sur PROD
git pull

# le vendoring TimesFM 2.5 n'est plus utilisé — le supprimer du disque
Remove-Item -Recurse -Force vendor        # si présent

# installer timesfm 3.0.1 depuis PyPI (+ retirer l'editable vendor)
uv sync
```

Vérification : `uv run python -c "from timesfm3 import TimesFM3Forecaster; print('OK')"`.

## 4. HF_TOKEN sur PROD (TODO ouvert depuis mai 2026)

1. Créer un token « Read » sur https://huggingface.co/settings/tokens.
2. Variable utilisateur Windows (persistent) :
   ```powershell
   [System.Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_xxx", "User")
   ```
   (ou ajouter `HF_TOKEN=hf_xxx` dans `.env`).
3. Si le modèle est « gated » (401/403 au download) : accepter la licence sur
   https://huggingface.co/google/timesfm-3.0-pytorch avec le compte lié au token.

## 5. Pré-télécharger le modèle AVANT le premier cycle (critique)

La tâche TimesFM d'un cycle a un **timeout de 180 s** : un téléchargement à froid
(~1,3 Go) pendant le cycle 1 donnerait HOLD 0.0. Pré-chauffer le cache HF :

```powershell
uv run python tests/smoke_timesfm3.py
```

Le script télécharge le modèle, vérifie les shapes `(horizon,)`/`(horizon, 9)` et
mesure le temps d'inférence CPU à contexte 2048 et 1024.

**Garde-fou** : si l'inférence à 2048 dépasse ~60 s sur le CPU PROD, repasser
`TIMESFM_CONTEXT = 1024` dans `src/timesfm_model.py` (commit + re-déploiement).
Le cache HF (`%USERPROFILE%\.cache\huggingface\hub`) n'est **pas** touché par le
reset : le téléchargement ne se fait qu'une fois.

## 6. Tests sur PROD

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q --basetemp=data_cache/test_tmp
```

Attendu : 282 passed / 3 skipped (harness live ignorés).

## 7. Reset des données (PROD)

```powershell
# 1. Aperçu (liste tout, y compris les artefacts logs_prod/ détectés)
uv run python reset_for_fresh_test.py --dry-run

# 2. Reset complet — INCLURE logs_prod/ (CWD du scheduler : DBs, state T212,
#    journal, data_cache/, lock y vivent réellement sur PROD)
uv run python reset_for_fresh_test.py --yes --include-logs-prod
```

Sans `--include-logs-prod` ni `--keep-logs-prod`, `--yes` **refuse** de s'exécuter
(exit 2) : plus de reset silencieux incomplet (cause racine de l'incident 24/08).
Tout est déplacé vers `reset_backup/<timestamp>/` (réversible). Les `.md`
(rapports d'audit) de `logs_prod/` sont préservés, ainsi que `.env*`, `.venv`,
`.git`, `memory-bank/` et le code source.

Effet : state T212, `trading_history.db`, `model_performance.db`,
`performance_monitor.db`, journal CSV, caches (prix, EIA, classic, PPO,
finacumen), briefs et rapports council → vierges. Les poids adaptatifs
repartent des `DEFAULT_BASE_WEIGHTS` (garde min 10 observations / 20 avant
pénalités win-rate). PPO et classic se réentraînent au 1er cycle.

## 8. Lancer le run 2

```powershell
# depuis logs_prod/ (CWD du scheduler — les artefacts runtime y atterrissent)
cd logs_prod
..\start_scheduler.bat
```

- Une seule instance (le verrou refuse la 2ᵉ).
- **1er cycle long** : re-téléchargement ~5 ans de data, retrain classic
  (calibration isotonic) + PPO depuis zéro (2000 steps), re-fetch EIA.
- Vérifier dans les fenêtres de log : `API TimesFM 3.0 (Torch) chargée avec
  succès`, `Initialisation de TimesFM 3.0 (google/timesfm-3.0-pytorch)...`,
  et au 1er cycle `TimesFM 3.0 prediction: <signal>`.

## 9. Checklist post-lancement

- [ ] `logs_prod\trading_journal.csv` : nouvelles lignes, colonne `T212_Equity`
      cohérente (1000 €/ticker au départ, pas d'héritage FIFO de la run 1).
- [ ] `logs_prod\trading.log` : pas de `CRITICAL` inexpliqué ; TimesFM 3.0 présent.
- [ ] Si achat : position visible dans l'app T212 avec stop GTC dédié dessous.
- [ ] `scheduler.log` : cycles 08:30→18:00 présents (fenêtre lun-ven).

## 10. Clôture administrative (memory-bank)

- [ ] `log.md` : entrée de clôture du run 1 (3 ordres, incidents corrigés
      en cours de route, critères ≥20 round-trips / dispo non atteignables).
- [ ] `progress.md` / `activeContext.md` : run 2 démarré <date>, GO/NO-GO à
      <date + 30 j>, critères inchangés (docs/PLAN_RUN_DEMO_30J.md §5).
- [ ] `changelog.md` : entrée produit (TimesFM 3.0, fin vendoring, reset étendu).

---

## Rappel kill-switch (inchangé)

1. Ctrl+C sur le scheduler (ou `Stop-Process -Name python -Force`).
2. Vendre manuellement les positions dans l'app T212 (les stops broker restent
   actifs en attendant — c'est le but).
3. Révoquer la clé API dans les paramètres T212 si nécessaire.

## Annexe — différences 2.5 → 3.0 (pour mémoire)

| | 2.5 (avant) | 3.0 (après) |
|---|---|---|
| Install | clone GitHub `vendor/timesfm` + patch `__init__.py` + editable | PyPI `timesfm>=3.0.1` (package `timesfm3`) |
| Checkpoint | `google/timesfm-2.5-200m-pytorch` (200M) | `google/timesfm-3.0-pytorch` (0.3B, ~1,3 Go) |
| Chargement | `TimesFM_2p5_200M_torch(...)` + `.compile(ForecastConfig)` | `TimesFM3Forecaster.from_pretrained(...)` (pas de compile) |
| Inférence | `forecast(horizon, inputs=[...])` → tuple | `predict(prices, horizon, return_quantiles=True)` → `.forecast`/`.quantiles` |
| Contexte | 1024 | **2048** (max 15360, padding auto par patch de 32) |
| Quantiles | tête optionnelle, ignorés par le wrapper | natifs (9, P10–P90) — médiane = signal, quantiles en métadonnées |
| Échec init | HOLD 0.0 pour toujours (singleton) | retry d'init à chaque `predict()` |
| Licence poids | Apache-2.0 | `timesfm-non-commercial-license-v1.0` |

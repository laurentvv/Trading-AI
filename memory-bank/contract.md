# Contrat de Validation

> **Rôle** — Contrat de validation technique négocié entre la planification et l'évaluation.
> Liste d'assertions strictes et testables. **Figé juste avant l'écriture de la première ligne de code** :
> une fois le code démarré, ce contrat ne doit plus être modifié par le générateur.
> Voir `AGENTS.md §1` et `§2` (invariants de sécurité) pour le contexte.

## Critères d'Acceptation Automatisés

### Défense JSON bi-couche (AGENTS.md §2.1) — critique
- [ ] Critère 1 : Les 4 sites d'appel LLM production portent le préfixe `"<|think|> "` dans leur system prompt (`src/llm_client.py:188`, `src/llm_client.py:236`, `src/oil_bench_model.py:158`, `src/web_researcher.py:205`).
- [ ] Critère 2 : Les 4 sites utilisent un paramètre `format` strict (pas le loose `format:json`) avec `additionalProperties: false` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`, `SCHEMA_FINACUMEN_SOLVER`).
- [ ] Critère 3 : Les 4 system prompts se terminent par le suffixe défensif `"...never add a 'thought' key."`.
- [ ] Critère 4 : `tests/check_llm_json.py` → tous les cas `*_schema*` et `oil_*` retournent OK (JSON propre même avec `<|think|>` actif).

### Invariants runtime (AGENTS.md §2.2)
- [ ] Critère 5 : Le budget par ticker est `INITIAL_BUDGETS` (1000€) et non un fallback codé en dur 5000€.
- [ ] Critère 6 : Le seuil de péremption du cache Parquet est de 1 jour (les fichiers plus anciens sont auto-rafraîchis).
- [ ] Critère 7 : `CYCLE_TIMEOUT_SECONDS` vaut 40 min dans `main.py:38`.
- [ ] Critère 8 : Sur timeout de cycle, `cancel_event` est positionné pour empêcher un thread orphelin de placer un ordre T212.
- [ ] Critère 9 : Le placement d'ordre T212 est sérialisé par un `threading.Lock` par ticker.

### Contrat de données (ModelResult)
- [ ] Critère 10 : Tous les modèles de l'ensemble retournent un objet `ModelResult` typé (`signal`, `confidence`, `reasoning`).
- [ ] Critère 11 : Les poids de base sont normalisés (somme → 1.0) au point d'usage dans `enhanced_decision_engine.py` / `adaptive_weight_manager.py`.

### FinAcumen (AGENTS.md §6, système réparé)
- [ ] Critère 12 : `lookup_ohlc` accepte `indicator` de type `str` (→ float) ou `list[str]` (→ dict).
- [ ] Critère 13 : Le sandbox `NumericalReasoningEngine` n'expose pas `__import__` ET pré-injecte `pd`/`np`.
- [ ] Critère 14 : `src/agents/solver.py` distingue l'exécution (`python_code` non vide) de la réponse finale (`action in BUY|SELL|HOLD`) sur le *contenu*, pas sur la présence de clé.

### Weekend Council (AGENTS.md §4, 11ème voix)
- [ ] Critère 15 : `get_council_ticker_stance` reçoit le **ticker de trading** (`self.ticker`, ex. `SXRV.DE`), pas le ticker d'analyse.
- [ ] Critère 16 : Le council est dans `fixed_weight_models` (l'AdaptiveWeightManager ne rescale pas son poids).
- [ ] Critère 17 : La confiance du vote council décroît linéairement (pleine à J+0 → 0 à J+7).

### Gemini Gateway (AGENTS.md §8)
- [ ] Critère 18 : En cas d'épuisement de quota / 429 / 503, la gateway retourne `None` (ne lève jamais d'exception) → l'appelant retombe sur Ollama local.
- [ ] Critère 19 : Le cap de facturation local `GEMINI_PAY_DAILY_CAP` est pré-vérifié par le `QuotaTracker` avant chaque appel payant.

## Protocole d'Évaluation

* **Commande de tests unitaires mockés (déterministes, sans Ollama)** :
  ```
  .venv\Scripts\python.exe -m pytest tests/test_llm_client.py tests/test_llm_prompts.py tests/test_oil_bench_model.py tests/test_weekend_council.py -v
  ```
* **Commande harness JSON live (nécessite `ollama serve`)** :
  ```
  .venv\Scripts\python.exe tests/check_llm_json.py
  ```
  *(Acceptance : tous les cas `*_schema*` et `oil_*` OK ; les cas `format:json` loose sont documentés comme défaillants et hors production.)*
* **Commande pipeline complet T212 démo** :
  ```
  uv run main.py --t212
  ```
* **Comportement attendu** :
  * Tests unitaires mockés : zéro avertissement critique, zéro échec.
  * Harness JSON : zéro échec sur les cas schema-strict.
  * Pipeline : exit 0, zéro erreur `"Could not find valid JSON"` dans `trading.log`.
* **Note Windows (CRITICAL)** : `uv run pytest ...` peut échouer avec "Failed to canonicalize script path" — préférer la forme directe `.venv\Scripts\python.exe -m pytest`. Ne jamais utiliser de commandes Linux-only (`rm -rf`, `ls`, `cat`).

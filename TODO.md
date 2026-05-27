# Ajout recherche WEB
~~Utilisation de **DuckDuckGo Search** via la librairie Python `duckduckgo-search` : src\web_researcher.py~~ [FAIT]

~~Query LLM pour web search~~ [FAIT]

~~On ajoutera les infos de la recherche au prompt llm texte~~ [FAIT]

# Résilience réseau et prix temps réel (2026-04-15)
~~Circuit breaker yfinance (info vs download séparés)~~ [FAIT]
~~Timeout 10s sur tous les appels yfinance~~ [FAIT]
~~Prix live T212 via `get_t212_price()`~~ [FAIT]
~~Skip `_yf_ticker_info()` quand cache parquet existe~~ [FAIT]
~~Séparer circuit breakers info/download pour ne pas bloquer Vincent Ganne~~ [FAIT]

# Backtesting Production (2026-05-05)
~~QuantConnect Lean (Docker) — remplacé par backtest_prod.py autonome~~ [FAIT]
~~`backtest_prod.py` — Moteur de backtest standalone avec signaux reels + frais T212~~ [FAIT]
- [ ] Optimisation des poids modeles via backtest_prod.py grid search
- [ ] Ajouter stop-loss / take-profit dans le backtest
- [ ] Generer un rapport de performance complet avec graphiques

# Problèmes restants
- [ ] Sentiment 0 headlines en PROD (Alpha Vantage quota ou config)
- [ ] LLM inference lent (~60s/call) - bottleneck principal du cycle
- [ ] Web recherche retourne parfois des articles obsolètes (ex: CNBC 2015)
- [x] Urea (UME=F) MA200 = nan (résolu par MA50 fallback dans `src/data.py`)
- [ ] HF_TOKEN non défini sur PROD
- [x] update_prediction_outcome() jamais appelé — résolu par feedback loop dans t212_executor.py (2026-05-12)

# TensorTrade et Cache (2026-04-28/29)
~~Intégrer TensorTrade / PPO comme 9ème signal~~ [FAIT]
~~Cache auto-invalidation si > 2 jours~~ [FAIT]
~~Fallback MA50 quand MA200 insuffisant~~ [FAIT]
~~Script `refresh_cache.py` pour forcer le rafraîchissement~~ [FAIT]
~~Supprimer DB files du tracking git~~ [FAIT]
- [ ] Optimiser le temps d'entraînement PPO (actuel ~500 timesteps)
- [ ] Evaluer la corrélation TensorTrade vs Classic sur backtest long
- [ ] Synchroniser les 9 traductions i18n avec les mises à jour README (TensorTrade, cache)

# Robustesse Prod (2026-05-12)
~~Feedback loop adaptatif (update_outcomes_for_date)~~ [FAIT]
~~Poids progressifs pour modèles en test (0.05)~~ [FAIT]
~~Normalisation des poids à la volée (somme → 1.0)~~ [FAIT]
~~Seuil cache 2j → 1j~~ [FAIT]
~~Budgets T212 par ticker (INITIAL_BUDGETS)~~ [FAIT]
~~Filtre grokipedia dans les logs~~ [FAIT]
- [ ] Configurer HF_TOKEN sur le serveur PROD (variable d'environnement Windows)
- [ ] Extraire les poids de base dans un fichier de config partagé (éliminer la duplication)

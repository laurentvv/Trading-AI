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

# Backtesting Institutionnel Lean (2026-05-04)
~~Intégration QuantConnect Lean comme couche de backtesting additive~~ [FAIT]
~~`src/lean_bridge.py` — Conversion trading_journal.csv → signaux Lean~~ [FAIT]
~~`src/lean_validator.py` — Validation automatisée via backtest Lean~~ [FAIT]
~~`TradingAI-Lean/` — Projet Lean avec baseline + framework + Alpha Models~~ [FAIT]
~~T212 fee model (0.1%) + slippage volume-share dans les algos Lean~~ [FAIT]
- [ ] Injecter les signaux réels du journal dans les backtests Lean (via insights.json)
- [ ] Porter les 9 modèles IA comme vrais Alpha Models Lean (actuellement approximations momentum)
- [ ] Optimisation des poids via Lean Optimizer (grid search)
- [ ] Générer un premier rapport institutionnel complet

# Problèmes restants
- [ ] Sentiment 0 headlines en PROD (Alpha Vantage quota ou config)
- [ ] LLM inference lent (~60s/call) - bottleneck principal du cycle
- [ ] Web recherche retourne parfois des articles obsolètes (ex: CNBC 2015)
- [x] Urea (UME=F) MA200 = nan (résolu par MA50 fallback dans `src/data.py`)
- [ ] HF_TOKEN non défini sur PROD

# TensorTrade et Cache (2026-04-28/29)
~~Intégrer TensorTrade / PPO comme 9ème signal~~ [FAIT]
~~Cache auto-invalidation si > 2 jours~~ [FAIT]
~~Fallback MA50 quand MA200 insuffisant~~ [FAIT]
~~Script `refresh_cache.py` pour forcer le rafraîchissement~~ [FAIT]
~~Supprimer DB files du tracking git~~ [FAIT]
- [ ] Optimiser le temps d'entraînement PPO (actuel ~500 timesteps)
- [ ] Evaluer la corrélation TensorTrade vs Classic sur backtest long
- [ ] Synchroniser les 9 traductions i18n avec les mises à jour README (TensorTrade, cache)

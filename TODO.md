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

# Problèmes restants
- [ ] Sentiment 0 headlines en PROD (Alpha Vantage quota ou config)
- [ ] LLM inference lent (~60s/call) - bottleneck principal du cycle
- [ ] Web recherche retourne parfois des articles obsolètes (ex: CNBC 2015)
- [ ] Urea (UME=F) MA200 = nan (historique insuffisant)
- [ ] HF_TOKEN non défini sur PROD

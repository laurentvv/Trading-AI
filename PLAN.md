# PLAN.md — Audit technique de Trading-AI

> **Date** : 2026-09-25 · **Commit audité** : `d5c3854` · **Nature** : analyse uniquement, aucun fichier de code modifié.
> **Méthode** : lecture ciblée des points d'entrée (`main.py`, `schedule.py`), des modules critiques (exécution T212, risque, données, LLM, dashboard), puis mesures outillées sur une copie isolée du dépôt (Python 3.12, `uv sync --frozen`) : `pytest --cov`, `ruff` (règles par défaut + `S,B,BLE,TRY,PL,UP,SIM`), `vulture`, `pip-audit` sur `uv.lock`.
>
> | Mesure | Résultat |
> |---|---|
> | Suite de tests (environnement vierge, hors `test_crawl4ai`/`test_full_cycle`) | 294 réussis · **2 en échec** · 3 ignorés |
> | Couverture globale | **46 %** — `enhanced_trading_example.py` 12 %, `performance_monitor.py` 16 %, `data.py` 25 %, `main.py` 27 %, `web_dashboard/app.py` 0 %, `morning_brief.py` 0 % |
> | Ruff (règles par défaut) | 37 findings (tous dans `tests/`) · règles étendues : 878 dont 152 `except Exception` (BLE001), 101 `logger.error` au lieu de `logger.exception` (TRY400) |
> | Dépendances | 8 dépendances runtime jamais importées · 1 vulnérabilité connue (`nltk 3.10.3`, transitive via `crawl4ai`) |
> | CI | Aucune (`.github/` absent) ; `pre-commit` limité à 4 hooks génériques |

---

## 1. Diagnostic

**Bloquant**
1. **Les sorties de position ne s'exécutent pas sur un cycle HOLD** : `main.py::_execute_t212_orders` (l. 141) ne transmet que les BUY/SELL, et `src/t212_executor.py::execute_t212_trade` (l. 1691) s'arrête aussi *avant* le bloc de sortie. Take-profit, trailing stop, time-stop, hard-stop côté exécuteur et remontée du stop ne tournent donc jamais pendant un cycle HOLD, soit la majorité des cycles. Seul le stop GTC placé chez le courtier protège la position.
2. **La couche de risque amont ne fonctionne pas** : la clé `market_data["price_series"]` lue par `main.py` n'est jamais remplie par `src/enhanced_trading_example.py`. Le hard-stop et l'inertie de sortie de `src/advanced_risk_manager.py` reçoivent donc `price_data=None`. De plus, `entry_price_index` est remplacé par le prix de l'ETF à chaque synchronisation T212. Enfin, le seuil de confiance du moteur (`_apply_risk_management`) est ignoré, parce que `main.py` utilise `final_signal`.
3. **`sync_state_from_t212` reconstruit l'état complet à chaque lecture** (`src/t212_executor.py`) : `highest_value` est perdu, si bien que le trailing stop logiciel ne peut jamais se déclencher. Même risque pour `entry_time` si le courtier ne renvoie pas `createdAt`.
4. **Le scheduler peut rester bloqué indéfiniment** : `schedule.py::run_trading_cycle` appelle `subprocess.run` sans timeout, et les threads orphelins de `main.py` ne sont pas des threads démon, donc ils sont attendus à la sortie de l'interpréteur. Un seul appel réseau figé (par exemple `requests.get` sans timeout dans `src/data.py:484`) bloque tout, et le lock-keeper continue de rafraîchir le verrou, ce qui masque le blocage.
5. **Sécurité** : `src/core/tools.py` exécute du code produit par le LLM avec `exec()`, et l'isolation se contourne (accès à `os` via `pd.io.common.os`, vérifié) sur la machine qui détient les clés T212. Par ailleurs, le dashboard (`web_dashboard/app.py`) utilise `admin/admin` par défaut, et sa documentation préconise `--host 0.0.0.0`.

**Important**
6. **Garde de fraîcheur en jours calendaires** (`src/data.py::get_etf_data`, 3 j) : la barre du vendredi est refusée le lundi avant l'ouverture américaine (^NDX) et avant 09:00 pour les ETF. Les cycles sont alors abandonnés, et `schedule.py` n'a ni calendrier des séances ni jours fériés.
7. **Comptabilité et état courtier** : le calcul FIFO repose sur `get_t212_order_history(limit=50)` sans pagination (l. 436). Par ailleurs, `_get_active_stop_order` renvoie la même valeur en cas d'échec et en l'absence de stop, ce qui contredit l'invariant « Failed fetch ≠ empty ».
8. **Tests** : ils écrivent dans le `trading_history.db` du répertoire courant (une transaction BUY fantôme via `tests/test_prod_fixes_2026_08_24.py`). La couverture est de 46 %, aucun test ne couvre le chemin `main → executor`, les tests réseau sont mélangés aux tests unitaires et il n'y a pas de CI.
9. **Architecture** : deux modules concentrent trop de responsabilités (`t212_executor.py`, 1 807 lignes ; `enhanced_trading_example.py`, 1 207 lignes). Les modules sont importés sous deux chemins (`src.X` et `X`), l'import a des effets de bord (`sys.exit`, `mkdir`, configuration du logging), les chemins dépendent du répertoire de lancement, et la configuration est dispersée (`scheduler_config.json` est chargé alors qu'il n'existe pas).

**Mineur**
10. Dépendances inutiles (`tensortrade`, `jax`, `shap`, etc.), code mort, alertes jamais transmises à l'opérateur (`email_config` n'est jamais fourni), bugs d'affichage du dashboard et écarts entre la documentation et le code (`README.md`, `AGENTS.md`).

---

## 2. Checklist de Refactoring

> Priorités : `[P0]` risque ou bug · `[P1]` dette technique · `[P2]` confort.
> Chaque correction doit préserver les invariants de `AGENTS.md` §2 : jamais de retry aveugle d'un ordre market, `write_db=not is_t212`, confirmation du fill avant toute écriture, stop GTC uniquement remonté.

### 2.1 Chemin d'exécution et gestion du risque (argent réel)

- [ ] [P0] `main.py` (`_execute_t212_orders`), `src/t212_executor.py` (`execute_t212_trade`) : quand une position est ouverte, évaluer les sorties à **chaque** cycle, y compris HOLD. Extraire `manage_open_position(state, current_pos)` (hard-stop → take-profit → trailing → time-stop, puis remontée du stop) et l'appeler avant le filtre `signal not in BUY/SELL`. Test attendu : cycle HOLD avec une position à +9 % ⇒ vente take-profit.
- [ ] [P0] `src/enhanced_trading_example.py` (`perform_enhanced_analysis`), `main.py` : exposer la série de prix attendue par `get_risk_adjusted_signal` (`market_data["price_series"]`). Soit la calculer sur l'**ETF** (même unité que `averagePricePaid`), soit conserver un vrai `entry_price_index` qui ne soit pas écrasé par la synchronisation. Ajouter un test qui échoue si `price_data is None` alors que `is_holding=True`.
- [ ] [P0] `main.py` : partir de `decision.risk_adjusted_signal` (seuil de confiance 0.20 et garde de volatilité extrême du moteur) au lieu de `decision.final_signal`. Supprimer la double évaluation de `get_risk_adjusted_signal` (une fois dans `perform_enhanced_analysis`, une fois dans `main.py`) au profit d'un pipeline de risque unique et documenté.
- [ ] [P0] `src/t212_executor.py` (`sync_state_from_t212`, `load_portfolio_state`) : fusionner l'état local au lieu de le reconstruire. `highest_value = max(local, courant)`, et conserver `entry_price_index` et `entry_time` d'origine. Vérifier la présence de `createdAt` dans `/equity/positions` : en son absence, le repli sur `now()` fait bloquer toutes les ventes par l'anti-churn et le time-stop n'est jamais atteint.
- [ ] [P0] `src/t212_executor.py` (`_evaluate_time_stop`) : appliquer `TIME_STOP_SOFT_LOSS` (déclarée mais jamais utilisée) ou corriger la docstring. Aujourd'hui, le time-stop force la vente quelle que soit la perte, jusqu'à -10 %.
- [ ] [P0] `src/t212_executor.py` (`_get_active_stop_order`) : distinguer trois cas (stop présent / aucun stop / état inconnu). Ne jamais placer de stop de « self-heal » ni annuler ou vendre sur un état inconnu, conformément à l'invariant « Failed fetch ≠ empty ».
- [ ] [P0] `src/t212_executor.py` (`get_t212_order_history`, `sync_state_from_t212`) : parcourir toutes les pages via `nextPagePath` (50 éléments maximum par page) avant le calcul FIFO. Au-delà de 50 fills par instrument, le P&L réalisé et l'equity sont faux. Ajouter un test avec plus de 120 fills.
- [ ] [P1] `src/t212_executor.py` : découper en modules sans changer de comportement.
  - `broker/t212_client.py` : HTTP, `safe_request`, gestion du 429 partout (`get_t212_price`, `get_t212_account_summary`, `_position_exists`, `_confirm_fill` et `_cancel_order` contournent aujourd'hui `safe_request`).
  - `portfolio/state_store.py` : écriture JSON atomique.
  - `portfolio/accounting.py` : FIFO et equity.
  - `execution/exit_rules.py` : règles de sortie.
  - `execution/orders.py` : passage d'ordres.
- [ ] [P1] `src/t212_executor.py` (`_execute_buy_order`) : supprimer l'envoi d'un `takeProfit` attaché. Le rejet par l'API est confirmé (commit `deda82c`), donc chaque achat commence par un POST refusé en 400.
- [ ] [P2] `src/t212_executor.py` (`_evaluate_take_profit`, `_evaluate_hard_stop`, `_execute_sell_order`, `_execute_buy_order`) : remplacer les accès directs `current_pos["walletImpact"]["currentValue"]` par un accesseur défensif commun, et utiliser `walletImpact.totalCost` comme coût de référence.
- [ ] [P2] `src/t212_executor.py` (`_place_stop_order`) : un POST de stop est rejoué sur erreur réseau via `safe_request`. Appliquer la même réconciliation que pour les ordres market (relire `/equity/orders` avant de rejouer).

### 2.2 Robustesse du scheduler et des cycles

- [ ] [P0] `schedule.py` (`run_trading_cycle`, `run_morning_brief`) : ajouter un `timeout=` à `subprocess.run` (par exemple 2 × `CYCLE_TIMEOUT_SECONDS` + marge). À expiration, tuer l'arbre de processus (`taskkill /T /F` sous Windows) et loguer en CRITICAL.
- [ ] [P0] `main.py` (bloc `__main__`), `src/enhanced_trading_example.py` (`get_model_predictions`) : les workers `ThreadPoolExecutor` ne sont pas des threads démon et sont attendus à la sortie de l'interpréteur, donc `shutdown(wait=False)` ne libère pas le processus. Après un timeout, terminer le processus explicitement (vidage des logs puis `os._exit`), ou lancer chaque ticker dans un sous-processus qu'on peut tuer.
- [ ] [P0] `src/data.py:484` (`get_alpha_vantage_data`) : ajouter un `timeout`. C'est le seul `requests.get` sans timeout du chemin critique (Ruff S113). Activer la règle S113 en CI.
- [ ] [P0] `src/data.py` (`get_etf_data`, `_price_cache_is_fresh`) : mesurer la fraîcheur en **séances** (calendriers XETR/XPAR/NYSE, par exemple `exchange_calendars`) plutôt qu'en jours calendaires. Ajouter les tests « lundi 10:00 » et « lendemain de férié ».
- [ ] [P1] `schedule.py` (`is_market_open`) : aligner la fenêtre sur les séances réelles Xetra/Euronext (09:00–17:30) et sur les jours fériés. Aujourd'hui, des ordres market peuvent partir à 08:30 (avant l'ouverture) ou à 18:00 (après la clôture).
- [ ] [P1] `schedule.py` : renommer le fichier (par exemple `scheduler.py`). Il masque le paquet PyPI `schedule`, déclaré dans `pyproject.toml` mais jamais utilisé. Mettre à jour `start_scheduler.bat` et `tests/test_scheduler_lock.py`.
- [ ] [P1] `schedule.py` (`run_weekend_council`) : le Council bloque la boucle principale jusqu'à 48 h (`COUNCIL_TIMEOUT`). Le lancer en processus détaché ou le borner. Aligner la docstring (« samedi & dimanche ») sur le code (samedi 01:00 uniquement).
- [ ] [P2] `main.py` (`_get_cancel_event`) : protéger l'accès au dictionnaire avec `_TICKER_LOCKS_GUARD`, comme dans `_get_ticker_lock`.

### 2.3 Sécurité

- [ ] [P0] `src/core/tools.py` (`NumericalReasoningEngine.execute`), `src/agents/solver.py` : sortir l'`exec()` du processus principal.
  - Option 1 : sous-processus isolé, **sans variables d'environnement** (ni clés T212 ni clés LLM), avec limite de temps et de mémoire et un répertoire temporaire.
  - Option 2 : remplacer par un DSL de calcul restreint (`lookup_ohlc` et arithmétique).
  - Aujourd'hui, `pd`/`np` injectés donnent accès à `os`. `sys.stdout` est aussi redirigé globalement, ce qui n'est pas sûr entre threads.
- [ ] [P0] `web_dashboard/app.py`, `.env.example`, `web_dashboard/README.md` : refuser le démarrage si `DASHBOARD_USER`/`DASHBOARD_PASS` sont absents ou valent `admin`. Documenter `--host 127.0.0.1` ou un reverse-proxy TLS plutôt que `0.0.0.0`, et limiter le débit des tentatives d'authentification HTTP Basic.
- [ ] [P1] `web_dashboard/templates/base.html`, `web_dashboard/templates/reports.html` : épingler les scripts CDN (`lucide@latest`, `alpinejs@3.x.x`, `marked`, `chart.js` ne sont pas versionnés) avec un attribut `integrity=` (SRI), ou les servir localement. Remplacer le CDN Tailwind « play », qui n'est pas prévu pour la production.
- [ ] [P1] `src/enhanced_trading_example.py` (`_fetch_news_task`), `src/news_fetcher.py` : ne plus passer `ALPHA_VANTAGE_API_KEY` en argument de ligne de commande, où elle est visible dans la liste des processus. La transmettre par l'environnement du sous-processus.
- [ ] [P1] `src/llm_client.py` (`construct_llm_prompt`) : traiter les titres de news, le contexte web (crawl4ai), le Morning Brief et le verdict du Council comme des données non fiables : délimiteurs explicites, troncature, rappel dans le prompt système que ces blocs ne sont pas des instructions. Ces contenus orientent un signal d'achat réel (risque d'injection de prompt).
- [ ] [P1] `src/news_fetcher.py` (`ALPHA_EAR_PATH`) : le code de production importe des scripts depuis `.agents/skills/alphaear-news/scripts`, un outillage d'agent IA dupliqué dans `.kilocode/` et `.qwen/`. Intégrer ce client dans `src/` et figer sa version.
- [ ] [P2] `src/bootstrap.py` : transformer `_redact_secrets` (`src/enhanced_trading_example.py`) en `logging.Filter` global (masquer `apikey=`, `Authorization`, etc. dans toutes les URL et traces loguées).

### 2.4 Qualité du code et architecture

- [ ] [P1] `src/enhanced_trading_example.py` : sortir l'orchestrateur de production de ce fichier nommé « example » (par exemple `src/pipeline/trading_system.py`) et le découper : préparation des données, exécution parallèle des modèles (`get_model_predictions`, environ 300 lignes), fusion et risque, reporting.
- [ ] [P1] `src/*.py`, `main.py`, `web_dashboard/app.py`, `src/finacumen_main.py` : importer partout via le paquet `src.`. Aujourd'hui `enhanced_decision_engine`, `t212_executor` et `llm_client` sont chargés deux fois (`X` et `src.X`), d'où les `try/except ImportError` de `update_performance_monitoring`. Supprimer les `sys.path.append/insert` et rendre le projet installable (`[build-system]` dans `pyproject.toml`).
- [ ] [P1] `src/enhanced_trading_example.py:103` (`sys.exit(1)` si la clé Alpha Vantage manque), `src/data.py:22`, `src/classic_model.py:19` (`mkdir`), `main.py` et `schedule.py` (`setup_environment` au niveau module) : déplacer ces effets de bord dans les fonctions `main()`. Aujourd'hui, un simple `import` peut terminer le processus ou créer des fichiers.
- [ ] [P1] Chemins relatifs au répertoire courant (`STATE_FILE` dans `src/t212_executor.py`, `DB_PATH` dans `src/database.py`, `model_performance.db` dans `src/adaptive_weight_manager.py`, `performance_monitor.db`, `CACHE_DIR` dans `src/data.py`, `trading_journal.csv` dans `main.py`) : les résoudre depuis une racine unique (`TRADING_AI_HOME`). En production, le programme est lancé depuis `logs_prod/`, et le résultat dépend du répertoire de lancement.
- [ ] [P1] Configuration dispersée : la centraliser dans un module typé (`src/settings.py`, pydantic-settings ou dataclass figée).
  - Correspondance des tickers : `TICKER_MAPPING_T212`, `ANALYSIS_MAPPING`, `_T212_MAPPED` de `src/data.py`, `TOPICS_MAP`, `TICKERS` de `schedule.py`, valeurs par défaut `argparse` de `main.py`, tests `is_oil` par sous-chaîne.
  - Budgets, seuils de sortie (dupliqués entre `src/t212_executor.py` et `src/advanced_risk_manager.py`), horaires.
  - Supprimer la lecture silencieuse de `scheduler_config.json`, absent du dépôt et ignoré par `.gitignore` via `*.json`.
- [ ] [P1] `src/llm_client.py` (`_async_query_nexus`, `get_llm_decision`, `get_visual_llm_decision`) : valider la sortie LLM avec un schéma pydantic : `signal` ∈ {BUY, SELL, HOLD}, `confidence` réel dans [0, 1] (convertir `"0.8"` ou `85`). Aujourd'hui, une confiance non numérique fait échouer `_calculate_weighted_score`, donc tout le cycle.
- [ ] [P1] Gestion des erreurs : réduire les 152 `except Exception` (BLE001) aux frontières qui le justifient. Remplacer les `logger.error(f"...{e}")` dans les `except` par `logger.exception` (101 cas TRY400). Priorité : `src/t212_executor.py`, `src/data.py`, `src/enhanced_trading_example.py`.
- [ ] [P1] `src/t212_executor.py` (`_update_feedback_loop`), `src/adaptive_weight_manager.py` (`update_outcomes_for_date`) : supprimer ce second circuit d'écriture des résultats. Il marque toutes les prédictions du jour d'entrée avec le P&L du trade (0/1 binaire, date d'horloge), alors que le resolver utilise des rendements de marché (-1/0/1) datés par la dernière barre. Les étiquettes deviennent incohérentes et faussent les pénalités de win-rate.
- [ ] [P1] `src/classic_model.py` (`train_ensemble_model`, `get_classic_prediction`) :
  - réentraîner le modèle déployé sur **100 %** de l'historique étiqueté après la sélection (aujourd'hui, les 20 % les plus récents, environ un an, sont exclus) ;
  - remplacer `ffill().bfill()` par `ffill()` seul, pour éviter une fuite d'information future ;
  - purger `data_cache/models/*.pkl` (un pickle par jour de données, jamais supprimé).
- [ ] [P1] `src/features.py` (`_align_macro_data`) : les variables macro ne sont renseignées que sur la dernière ligne. Elles valent 0 à l'entraînement et leur vraie valeur en inférence (écart entraînement/inférence). Les aligner en séries temporelles (`merge_asof` sur les dates de publication) ou les retirer de `select_features`.
- [ ] [P1] `src/enhanced_decision_engine.py` (`_detect_market_regime`) : `market_data["adx"]` n'est jamais fourni, donc le régime est toujours `unknown` et `regime_adjustments` ne sert à rien. Fournir l'ADX ou supprimer la branche.
- [ ] [P2] Code mort signalé par vulture, à supprimer ou justifier :
  - `check_ollama_health`, `_query_ollama` (`src/llm_client.py`) ;
  - `retrain_if_stale` (`src/classic_model.py`) ;
  - `update_adaptive_thresholds`, `get_model_weights_recommendation`, constantes `*_BONUS*` (`src/enhanced_decision_engine.py`) ;
  - `_yf_ticker_info` (`src/data.py`), `src/read_simul.py` ;
  - `calculate_position_sizing` : affiché mais ignoré, puisque la taille est fixée à 100 % dans le code ;
  - `insert_model_signal` : la table `model_signals` n'est jamais alimentée alors que `src/council/weekend_council.py::fetch_recent_model_signals` la lit.
- [ ] [P2] `src/database.py` : utiliser `contextlib.closing` partout (`get_latest_portfolio_state` et `get_latest_transaction` ne ferment pas la connexion en cas d'exception). Activer `PRAGMA journal_mode=WAL`, car le scheduler, le dashboard et le Council accèdent à la base en parallèle.
- [ ] [P2] Dates et heures : passer à des datetimes UTC avec fuseau (`datetime.now(UTC)`). Le mélange heure locale naïve / `createdAt` UTC du courtier est aujourd'hui masqué par des `except TypeError`.

### 2.5 Tests

- [ ] [P0] `tests/` : rendre la suite hermétique avec une fixture `autouse` qui redirige `DB_PATH`, `STATE_FILE`, `model_performance.db`, `data_cache/` et les logs vers `tmp_path`. `tests/test_prod_fixes_2026_08_24.py::TestMaxAvailableSizing::test_buy_allocates_100_percent_of_available_budget` appelle le vrai `insert_transaction` et insère un BUY fantôme dans le `trading_history.db` du répertoire courant, ce qui viole l'invariant d'isolation de la base. En environnement vierge, il échoue avec `no such table: transactions`. Importer `main`/`schedule` crée aussi `trading.log` et `scheduler.log`.
- [ ] [P1] `pyproject.toml`, `tests/` : séparer les harnais « live » avec des marqueurs `live`/`network`, exclus par défaut (`addopts = "-m 'not live'"`). Déplacer vers `tests/live/` :
  - `test_hyperliquid.py` (appel réseau réel, en échec hors ligne) ;
  - `test_crawl4ai.py` ;
  - `test_full_cycle.py` (script destructif qui supprime les bases et passe des ordres démo) ;
  - `check_*`, `bench_*`, `backtest_llm_quality.py`, `run_short_backtest.py`.
- [ ] [P1] `tests/` : ajouter des tests d'intégration du chemin d'exécution réel, avec modèles et courtier simulés : `run_trading_analysis(..., is_t212=True)` pour les cas HOLD avec position ouverte (sorties), BUY sans position, SELL bloqué (anti-churn, garde anti-perte) et timeout de cycle (`cancel_event`).
- [ ] [P1] Couverture : viser au moins 70 % sur `src/t212_executor.py` (67 %), `main.py` (27 %), `src/enhanced_trading_example.py` (12 %), `src/data.py` (25 %), `src/performance_monitor.py` (16 %) et `web_dashboard/app.py` (0 %, via `fastapi.testclient`). Publier le rapport `pytest-cov`.
- [ ] [P1] CI : créer `.github/workflows/ci.yml` sur `windows-latest`, comme le DEV et la PROD : `uv sync --frozen`, `ruff check`, `pytest -m "not live"`, `pip-audit`. Ajouter `ruff` et les tests rapides à `.pre-commit-config.yaml`.
- [ ] [P2] `tests/` : corriger les 37 findings Ruff (imports et variables inutilisés) et les coroutines jamais attendues (`tests/test_llm_client.py`, `tests/test_weekend_council.py`).

### 2.6 Dépendances

- [ ] [P1] `pyproject.toml` : retirer les dépendances runtime jamais importées :
  - `tensortrade`, qui installe `ipython`, `plotly` et `pytest` en production (le modèle « TensorTrade » est une implémentation maison Gymnasium + Stable-Baselines3) ;
  - `jax`, `jaxlib`, `einops`, `shap`, `seaborn`, `beautifulsoup4`, `setuptools`, `schedule`.
  Déplacer `pre-commit` (avec `pytest`, `pytest-cov`, `ruff`) dans `[dependency-groups] dev`, et déclarer explicitement `gymnasium`, importé directement par `src/tensortrade_model.py`.
- [ ] [P1] `uv.lock` : `nltk 3.10.3`, transitive via `crawl4ai`, est signalée par `pip-audit` (PYSEC-2026-3740). Suivre le correctif amont et contraindre la version dès qu'il est publié.
- [ ] [P1] `morning_brief/requirements.txt` : supprimer ce fichier, dupliqué et divergent de `pyproject.toml` (`beautifulsoup4` n'y est pas utilisé).
- [ ] [P2] `pyproject.toml` : réévaluer les épinglages `numpy<2.0` et `pandas==2.2.3`, qui bloquent les mises à jour. Renseigner `description` (actuellement « Add your description here ») et la licence (MIT annoncée, mais aucun fichier `LICENSE`).
- [ ] [P2] `pyproject.toml` : documenter l'index PyTorch CPU (`[tool.uv.sources]`). La résolution sous Linux télécharge environ 3 Go de wheels CUDA inutiles pour une inférence sur CPU.

### 2.7 Performance

- [ ] [P1] `main.py`, `src/t212_executor.py` : un cycle BUY/SELL fait environ 10 appels T212 (synchronisation complète dans `_execute_t212_orders`, nouvelle synchronisation dans `execute_t212_trade`, puis `_get_portfolio_info`). Ne synchroniser qu'une fois par cycle et transmettre l'état en paramètre, ce qui limite aussi les 429.
- [ ] [P1] `src/data.py` (`get_etf_data`) :
  - mémoriser les résultats pendant le cycle (clé `(ticker, date)`) : jusqu'à 5 appels par ticker et par cycle (ETF dans `__init__`, indice, ETF de secours, `CRUDP.PA` + `SXRV.DE` pour Grebenkov) ;
  - ne plus retélécharger `^VIX` en `period="max"` à chaque rafraîchissement ;
  - calculer la durée de validité du cache à partir du mtime du fichier et du calendrier des séances. Aujourd'hui, tout est retéléchargé à chaque appel le week-end.
- [ ] [P2] `src/llm_client.py` (`_run_sync`) : réutiliser une seule boucle asyncio et un `AIGateway` par processus, au lieu d'un `asyncio.run` et d'un nouveau contexte à chaque appel.
- [ ] [P2] `src/tensortrade_model.py` : sortir le fine-tuning PPO (500 timesteps par cycle) du chemin critique (tâche nocturne) et ne faire que l'inférence en séance.
- [ ] [P2] `src/web_researcher.py` : réutiliser une instance Chromium (crawl4ai) ou mettre en cache le contexte web par (requête, jour). Un navigateur est lancé à chaque cycle, et le timeout externe de 30 s laisse des threads, donc des navigateurs, orphelins.

### 2.8 Observabilité

- [ ] [P1] `src/performance_monitor.py`, `src/enhanced_trading_example.py` : `PerformanceMonitor` est créé sans `email_config`, si bien qu'aucun événement critique (position sans stop, synchronisation impossible, timeout de cycle) n'atteint l'opérateur. Brancher un canal de notification (voir Fonctionnalité 1).
- [ ] [P1] `src/bootstrap.py` : ajouter `%(name)s` et un identifiant de cycle (`cycle_id`, ticker) au format des logs. Configurer le logging une seule fois (`morning_brief/morning_brief.py` et `src/finacumen_main.py` ont leur propre `basicConfig`). Éviter que plusieurs processus fassent tourner le même fichier en même temps sous Windows (`QueueHandler`, ou un fichier par processus).
- [ ] [P1] `main.py` (`_write_trading_journal`) : journaliser **tous** les modèles votants (manquent `oil_bench`, `grebenkov`, `hmm_model` et `council`) ainsi que le motif des corrections de risque. Gérer les changements d'en-tête : un CSV existant garde son ancien en-tête, ce que `backtest_prod.py` contourne avec des colonnes `extra_i`.
- [ ] [P2] `schedule.py` : écrire un fichier heartbeat (`last_cycle_ok`, durée, code retour) exploitable par le dashboard et par une supervision externe.

### 2.9 Dashboard web

- [ ] [P0] `web_dashboard/app.py:96` (`read_performance`) : la requête trie `portfolio_history` par `timestamp`, une colonne qui n'existe pas (c'est `date`). L'erreur est avalée par un `print` et la page reste toujours vide.
- [ ] [P0] `web_dashboard/app.py:155` (`read_positions`) : l'état local est lu pour `"QQQ"` et `"CL=F"` au lieu des clés réelles `SXRVd_EQ` et `OD7Fd_EQ`, donc la réconciliation affichée est vide. Itérer sur `TICKER_MAPPING_T212`.
- [ ] [P0] `web_dashboard/app.py:51` (`get_recent_logs`) : lit `main.log`, qui n'existe pas. Le pipeline écrit dans `trading.log` (`main.py` → `setup_environment("trading.log")`).
- [ ] [P2] `web_dashboard/app.py` : remplacer les `print` par du logging, remonter les imports en tête de fichier (`# noqa: E402`), et charger les rapports Markdown à la demande (aujourd'hui, tout l'historique est lu à chaque requête).

### 2.10 Documentation

- [ ] [P1] `README.md` : corriger les écarts avec le code :
  - `src/vincent_ganne_model.py` n'existe pas (la classe est dans `src/enhanced_decision_engine.py`) ;
  - `scheduler_config.json` est absent ;
  - le Council tourne le samedi à 01:00 uniquement, pas « samedi et dimanche à 09:00 » ;
  - un cycle prend 100 à 119 s selon les mesures de `memory-bank/log.md`, pas « 30-45 s » ;
  - le filtre de confiance est à 0.20 dans le code, pas à 40 % ;
  - le fichier `LICENSE` est absent ;
  - « TensorTrade » est une implémentation maison, pas le paquet.
- [ ] [P1] `AGENTS.md` §2.2 et `memory-bank/contract.md` (critère 21) : le verrou est documenté comme périmé après 2 h, mais `SCHEDULER_LOCK_STALE_SECONDS` vaut 15 min dans `schedule.py`.
- [ ] [P1] `src/enhanced_trading_example.py` (commentaires des phases A et C), `src/llm_client.py` : supprimer les références à Ollama. Les budgets de timeout sont encore justifiés par les LLM locaux supprimés.
- [ ] [P2] `src/t212_executor.py` et `src/enhanced_trading_example.py` : corriger les commentaires erronés (`SXRVd_EQ` décrit comme « iShares S&P 500 » alors que c'est le Nasdaq 100 ; `CRUDP.PA` décrit comme « Lyxor » dans `ANALYSIS_MAPPING` alors que c'est WisdomTree).
- [ ] [P2] Racine du dépôt :
  - ranger les rapports dans `docs/` (`RAPPORT_AUDIT_GLOBAL.md`, `audit_passage_prod_juin2026.md`, `fingpt_implementation_plan.md`, `COUNCIL_TEST_PROD.md`, `SYSTEM_SUMMARY.md`, `python_health_report.md`) ;
  - ranger les scripts d'exploitation dans `scripts/` (`reset_for_fresh_test.py`, `clean_phantom_trades.py`, `audit_prod_logs.py`, `backtest_prod.py`, `refresh_cache.py`) ;
  - générer `GEMINI.md` et `QWEN.md` à partir de `AGENTS.md` pour éviter les divergences ;
  - marquer les 9 README `i18n/` comme non maintenus, ou les générer.
- [ ] [P2] `web_dashboard/README.md` : remplacer `uv pip install …` (qui inclut `sqlalchemy`, inutilisé) par `uv sync` et une commande de lancement sécurisée.

---

## 3. Checklist de Nouvelles Fonctionnalités

### Fonctionnalité 1 — Supervision et alertes en temps réel (heartbeat + dead-man switch)

Le système tourne 30 jours sans surveillance humaine avec de l'argent (démo, puis réel), mais aucun événement critique ne sort des fichiers de log.

- [ ] Créer `src/notifications.py` : un `NotificationHandler(logging.Handler)` déclenché à partir du niveau CRITICAL, et une API `notify(event, severity, payload)`. Transports ntfy.sh et Telegram Bot API via `requests` (timeout de 5 s), avec envoi asynchrone par file (`queue.Queue`) et thread démon.
- [ ] Brancher les événements métier : position sans stop courtier (`_ratchet_stop_order`, `_handle_failed_sell`), N synchronisations T212 échouées d'affilée, données périmées à la source (`get_etf_data`), timeout de cycle (`main.py`), fill non confirmé, ordre exécuté (côté, quantité, prix réel).
- [ ] Heartbeat : `schedule.py` écrit `heartbeat.json` à chaque tick. Un dead-man switch **externe** (ping healthchecks.io ou équivalent) alerte si aucun cycle n'a réussi depuis 2 × `INTERVAL_MINUTES` pendant une séance.
- [ ] Ajouter `GET /health` dans `web_dashboard/app.py`, sans authentification et sans donnée sensible : âge du heartbeat, résultat du dernier cycle, âge du verrou, positions protégées ou non.
- [ ] Envoyer un résumé quotidien à 17:45 : equity par ticker, P&L réalisé et latent, nombre de cycles, erreurs et avertissements.
- [ ] Anti-spam : dédupliquer par clé d'événement sur une fenêtre glissante (30 min). Tests unitaires avec un transport simulé.
- [ ] Ajouter les variables dans `.env.example` (`NOTIFY_BACKEND`, `NTFY_TOPIC`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) et documenter dans `AGENTS.md` §2.

**Approche technique** — Le handler est installé une seule fois dans `src/bootstrap.setup_environment`, ce qui capte immédiatement tous les `logger.critical` déjà présents aux points GO-gate sans toucher à chaque appel. Seuls les événements positifs (ordre exécuté, résumé) passent par `notify()`. L'envoi HTTP réutilise `requests`, déjà en dépendance, dans un thread démon alimenté par une file : une panne de notification ne doit jamais retarder un cycle ni le passage d'un ordre. ntfy.sh est le transport par défaut (push mobile, auto-hébergeable, sans compte), Telegram est optionnel. Écartés : SMTP (code présent dans `performance_monitor.py` mais jamais configuré, fragile avec la 2FA) et Prometheus/Grafana (disproportionné pour une seule machine Windows). Le dead-man switch doit être externe, car un processus bloqué (cas du `subprocess.run` sans timeout) ne peut pas signaler lui-même son blocage.

### Fonctionnalité 2 — Backtest fidèle à la production avec courtier simulé

Aujourd'hui, `backtest_prod.py` rejoue uniquement les signaux du journal, et `scripts/backtest_ensemble_10y.py` ignore la logique de sortie. Aucun des deux ne valide la chaîne réelle signal → risque → sorties → ordres avant un run démo de 30 jours.

- [ ] Extraire de `src/t212_executor.py` un port `BrokerPort` (Protocol : `positions()`, `cash()`, `place_market()`, `place_stop()`, `cancel()`, `order_history()`) et un adaptateur `T212Broker`, sans changement de comportement (tests GO-gate au vert).
- [ ] Extraire les règles de sortie en fonctions pures dans `src/execution/exit_rules.py` (entrée : position, prix, horloge ; sortie : décision et motif), utilisées à la fois en live et en backtest.
- [ ] Implémenter `SimBroker` : fills au prix d'ouverture de la séance suivante, frais de 0,1 %, précision `TICKER_QUANTITY_PRECISION`, stop GTC déclenché sur le `Low` de la barre, aucun fill hors séance.
- [ ] Créer `scripts/backtest_pipeline.py` : horloge simulée, walk-forward strict (entraînement du modèle classique, du HMM et de Grebenkov uniquement sur des données ≤ t), réponses LLM rejouées depuis le registre de décisions (Fonctionnalité 3) ou remplacées par HOLD.
- [ ] Produire un rapport : courbe d'equity, drawdown maximal, Sharpe, nombre d'allers-retours, taux de réussite, comparaison buy & hold. Export CSV/PNG dans `backtest_results/` (déjà dans `.gitignore`).
- [ ] Recherche par grille des seuils (`adaptive_thresholds`, TP/SL, `MIN_HOLDING_HOURS`) avec validation hors échantillon. Cela couvre les entrées de `TODO.md` « grid search » et « stop-loss / take-profit dans le backtest ».
- [ ] Test de parité : une même série de prix doit produire la même séquence d'ordres en live simulé et en backtest.

**Approche technique** — La clé est le port `BrokerPort`. Le code d'exécution actuel mêle HTTP, état et décisions. Une fois séparé, le vrai `EnhancedDecisionEngine`, l'`AdvancedRiskManager` et les règles de sortie tournent à l'identique contre `SimBroker`, ce qui détecte des régressions comme « sorties ignorées sur HOLD ». Les données viennent des parquets de `data_cache/` déjà produits par `src/data.py`. Les modèles CPU sont réentraînés en walk-forward (déjà possible via `train_ensemble_model(walk_forward=True)`). Les votes LLM, coûteux et non déterministes, sont rejoués depuis les réponses enregistrées pour rester fidèles et reproductibles. Écartés : vectorbt et backtrader (leur modèle d'exécution ne reproduit pas la logique maison : garde anti-perte, anti-churn, stop GTC qui ne fait que monter), et QuantConnect Lean (déjà rejeté pour sa dépendance à Docker).

### Fonctionnalité 3 — Registre des décisions et explicabilité dans le dashboard

Chaque audit en production reconstruit aujourd'hui les décisions en analysant des logs texte (`audit_prod_logs.py`, 705 lignes) et un CSV à colonnes fixes qui omet quatre modèles.

- [ ] Ajouter les tables `decision_log` (une ligne par cycle et par ticker) et `model_votes` (une ligne par modèle) dans `src/database.py`, avec migration idempotente et index `(ticker, ts)`.
- [ ] Générer un `cycle_id` (UUID) dans `main.py` et le propager dans les logs (voir 2.8), `HybridDecision` et les appels à `execute_t212_trade`.
- [ ] Enregistrer chaque étape : votes bruts, poids adaptatifs, score pondéré, seuils, `_apply_risk_management`, `get_risk_adjusted_signal`, règles de sortie évaluées, anti-churn, garde anti-perte, ordre (id, statut) et fill (prix réel).
- [ ] Conserver les réponses LLM (texte tronqué, fournisseur, modèle, latence, motif d'échec) avec une rétention configurable (par défaut 90 jours) : elles servent au rejeu de la Fonctionnalité 2.
- [ ] Ajouter au dashboard les pages `/decisions` (liste filtrable) et `/decisions/{cycle_id}` (enchaînement score → corrections → ordre), avec l'attribution par modèle dans Chart.js (déjà chargé).
- [ ] Faire lire cette table par `src/council/weekend_council.py`, qui lit aujourd'hui une table `model_signals` vide, et par `morning_brief/tools/analyze_trading_logs.py`.
- [ ] Tests : l'écriture a lieu même en cas d'exception ou de timeout de cycle, et aucune transaction n'est écrite (invariant `write_db`).

**Approche technique** — On reste sur SQLite et `src/database.py`, sans nouvelle dépendance. Une migration idempotente s'appuie sur le modèle existant (`_migrate_model_signals_table`), en mode WAL (voir 2.4). L'écriture se fait une fois en fin de cycle depuis `main.py`, dans un `finally`, y compris pour HOLD, timeout et erreur. Le registre trace des décisions, pas des trades : il reste donc compatible avec l'invariant `write_db=not is_t212`, qui interdit seulement les transactions simulées. Les pages s'ajoutent au FastAPI et aux templates Jinja existants, derrière l'authentification (voir 2.3). À terme, le registre remplace `trading_journal.csv` et l'essentiel de `audit_prod_logs.py`, et alimente les entrées de la Fonctionnalité 2. Écartés : MLflow et Weights & Biases (orientés suivi d'entraînement, trop lourds pour une machine Windows unique) ; un journal JSONL seul (difficile à requêter dans le dashboard et le Council).

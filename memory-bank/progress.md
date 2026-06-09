## 1. Statut Actuel
- **Progression Globale**: Phase 3 : Optimisation Architecturale & Extensibilité.
- **Dernière Étape Complétée**: Refactoring structurel complet (BaseModel) et centralisation de la configuration.
- **Étape Actuelle**: Validation de la stabilité en production avec la nouvelle architecture modulaire.

## 2. Ce Qui Fonctionne
- **Architecture Modulaire (BaseModel)**: Interface standardisée pour tous les modèles IA, permettant un ajout facile de nouveaux signaux sans modifier le moteur de décision.
- **Configuration Centralisée (`scheduler_config.json`)**: Tous les seuils techniques, de risque et les poids sont désormais pilotables via JSON, sans toucher au code.
- **Logging Standardisé**: Remplacement des `print()` par `logging` dans tout le pipeline d'exécution Trading 212.
- **Moteur Hybride Gemma 4**: Utilisation de `hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K` pour une analyse tri-modale (texte, vision, news) plus fine.
- **Skill AlphaEar News**: Récupération des tendances financières "hot" (Weibo, WallstreetCN) intégrée au flux décisionnel.
- **Nouveau Scheduler Autonome**: Script `schedule.py` gérant les horaires de marché (8h30-18h00) et l'intervalle de 30 minutes avec dashboard live.

...

## 5. Corrections Récentes
- **2026-06-06**: Standardisation `ModelResult`, Centralisation du Risque et Code Health Grade B
  * **Contexte** : Hétérogénéité des retours des modèles (certains renvoyaient des dicts, d'autres des tuples) et logique de gestion des pertes (anti-loss) éparpillée dans plusieurs modèles individuels.
  * **Changement** : Transition totale vers la dataclass `ModelResult` pour les 11 modèles de l'ensemble (incluant Grebenkov, TensorTrade, TimesFM, Ollama). Retrait de la gestion de l'anti-loss/trailing-stop des modèles individuels pour la confier exclusivement à `AdvancedRiskManager`.
  * **Qualité** : Audit du projet via le skill `python-health-audit`. Nettoyage de tous les codes morts/imports inutilisés trouvés par Ruff et Vulture (Grade B validé). Exclusion de `vendor/` pour le suivi qualité.
  * **Tests** : Suite pytest à 100% de succès après adaptation des tests à l'usage d'attributs objets.
- **2026-06-06**: Ré-activation du mode pensée Gemma (`<|think|>`)
  * **Contexte** : Le token `<|think|>` avait été retiré des prompts système en mai 2026 pour neutraliser un bug d'extraction JSON (débris `<|channel>thought` corrompant les réponses). L'analyse rétrospective a montré que la **vraie** défense est le schéma JSON strict (`additionalProperties: false`) passé via le paramètre `format` d'Ollama — pas l'absence du token.
  * **Changement** : Ré-introduction de `"<|think|> "` en préfixe du system prompt aux 4 sites d'appel production : `src/llm_client.py:188` (texte), `:236` (visuel), `src/oil_bench_model.py:158` (oil bench), `src/web_researcher.py:205` (recherche web). Suffixe défensif `"...never add a 'thought' key."` **conservé** comme seconde couche.
  * **Validation (branche `think-mode`)** :
    - 12/12 tests mockés pytest OK (`test_llm_client`, `test_llm_prompts`, `test_oil_bench_model`).
    - `tests/check_llm_json.py` : 6/10 cas OK — **tous** les cas schema-strict passent avec `<|think|>` actif (`v3_schema`, `v6_schema`, `v7_schema_strict`, `oil_v1_buggy`, `oil_v2_fixed`, `oil_v3_schema`). Les 4 échecs sont sur des variantes `format:json` (loose) qui **ne sont pas** utilisées en production.
    - `uv run main.py --t212` : exit 0, 4.66 min total, 2 nouvelles lignes dans `trading_journal.csv` (CRUDP.PA HOLD 19.16%, SXRV.DE BUY 35.18%), zéro erreur `"Could not find valid JSON"` dans `trading.log`.
  * **Décision** : Documentée dans `docs/ADR-001-think-mode-dual-layer-defence.md` (défense bi-couche, preuves du harness, procédure de reversal).
  * **Branche** : `think-mode` (créée pour la validation, fusionnée sur `main`).
- **2026-05-15**: Intégration Données T212 Hybrides (Live Price + Portfolio Sync)
  * **`_inject_t212_live_price()`** dans `src/data.py` : Après le chargement des données Yahoo (OHLCV historique), la dernière barre Close/High/Low des ETFs T212 est patchée avec le prix live Trading 212 (via `/equity/positions`). Ne s'applique qu'aux tickers mappés (`SXRV.DE`, `SXRV.FRK`, `CRUDP.PA`). L'indice (^NDX, ^VIX, CL=F) n'est pas affecté.
  * **`sync_state_from_t212()`** dans `src/t212_executor.py` : Nouvelle fonction qui reconstruit l'état du portefeuille directement depuis les données T212 réelles (`/equity/positions` pour les positions ouvertes, `/equity/history/orders` pour le P&L réalisé). Utilise un matching FIFO pour les lots d'achat.
  * **`load_portfolio_state()` réécrit** : T212 est désormais la source de vérité primaire. Le fichier JSON local (`t212_portfolio_state.json`) sert uniquement de fallback offline. Après chaque sync T212, l'état est sauvegardé localement pour les consultations hors-ligne.
  * **Nouvelles fonctions utilitaires** : `get_t212_positions()`, `get_t212_account_summary()`, `get_t212_order_history()` — wrappers propres autour de l'API T212 v0.
  * **Validation** : Cycle complet testé sur SXRV.DE (HOLD, confiance 31.23%, risque VERY_HIGH). T212 sync fonctionne correctement (`SXRVd_EQ | no position | realized P&L=+0.00 EUR`).
  * **Limitation connue** : Le prix live T212 n'est disponible que pour les instruments avec position ouverte (via `/equity/positions`). SXRV.DE sans position → pas de prix live T212, Yahoo reste la source.
- **2026-05-13**: Bug Fix Budget T212
  * **`load_portfolio_state()` early return** : Quand `t212_portfolio_state.json` n'existait pas, la fonction retournait `{"tickers": {}}` sans initialiser le ticker — le bloc `if ticker:` (qui set le budget 1000€) était sauté. Corrigé : le `else` englobe maintenant `_read_with_retry` + migration.
  * **Fallback achat `5000.0` → `DEFAULT_INITIAL_BUDGET`** : Dans `execute_t212_trade()`, `state.get("current_capital", 5000.0)` utilisait 5000€ comme fallback au lieu de `DEFAULT_INITIAL_BUDGET` (1000€). Conséquence PROD : 4 ordres rejetés "Insufficient funds" + 1 achat à 4750€ au lieu de 950€.
- **2026-05-12**: Améliorations Prod & Robustesse
  * **Feedback Loop Adaptatif** : Après chaque vente confirmée sur T212, `update_outcomes_for_date()` enregistre les résultats réels (return_1d, win/loss) dans `model_performance.db` via une connexion SQLite unique. L'AdaptiveWeightManager peut désormais ajuster les poids dynamiquement.
  * **Poids Progressifs** : Tous les modèles expérimentaux (vincent_ganne, oil_bench, tensortrade) passent de poids 0.0 à 0.05 (phase de test active). Poids normalisés à la volée (somme→1.0) dans le moteur de décision.
  * **Seuil Cache 2→1 jour** : Le cache Parquet s'invalide désormais après 1 jour au lieu de 2. Âge du cache journalisé en jours fractionnels.
  * **Budgets T212 par Ticker** : Remplacement du budget fixe 5000€ par `INITIAL_BUDGETS` configurable (1000€ par ticker).
  * **Filtre Grokipedia** : Suppression des warnings non-bloquants crawl4ai/grokipedia dans les logs.
  * **Test `test_weight_alignment.py`** : Corrigé le chemin d'import et mis à jour les poids attendus (1.05 normalisé à l'usage).
- **2026-05-06**: Refactoring Architectural Majeur
  * **Découplage des Modèles** : Introduction de la classe abstraite `BaseModel` et standardisation des prédictions via `ModelResult`. `VincentGanneModel` est le premier migré nativement.
  * **Moteur de Décision Générique** : `EnhancedDecisionEngine` supporte désormais une liste dynamique de modèles, simplifiant l'ajout de futurs signaux IA.
  * **Source de Vérité Unique** : Création de `scheduler_config.json` centralisant tous les "Magic Numbers" (seuils WTI/Brent/DXY, limites de drawdown, seuils de volatilité).
  * **Injection de Dépendance** : L'orchestrateur injecte la configuration dans tous les composants subordonnés.
  * **Logging de Production** : Standardisation complète des logs dans `t212_executor.py` pour un meilleur suivi via le scheduler.
  * **Robustesse JSON** : Correction de la gestion des fichiers d'état corrompus dans l'exécuteur T212.
  * **Migration PROD** : Mise à jour réussie de l'état du portefeuille en production (`logs_prod/`) vers le nouveau format multi-ticker.
- **2026-05-05**: Remplacement Lean par Backtest Prod Autonome


## 4. Problèmes Connus
- **Résolu**: Le planificateur précédent était non fonctionnel et provoquait l'échec de l'analyse quotidienne. Ceci a été résolu avec le nouveau `src/intelligent_scheduler.py`.
- **Résolu**: Problèmes de transition de phase et de persistance de la date de démarrage du projet (corrigés le 12 septembre 2025).
- **Résolu (2025-09-12)**: Corrigé un bug de persistance où la transition de phase n'était pas sauvegardée immédiatement, causant une réinitialisation de la phase au redémarrage du planificateur.

## 5. Corrections Récentes
- **2026-05-05**: Remplacement Lean par Backtest Prod Autonome
  * Suppression de **`TradingAI-Lean/`**, **`src/lean_bridge.py`**, **`src/lean_validator.py`**, **`run_lean_backtest.py`** — la dependance Docker/Lean etait trop lourde et les signaux n'etaient pas injectes.
  * Creation de **`backtest_prod.py`** : moteur de backtest autonome qui replaye les signaux reels de `logs_prod/trading_journal.csv` contre les prix parquet de `data_cache/` avec frais T212 (0.1%).
  * Compare strategie signaux vs buy-and-hold : Sharpe, MaxDD, Win Rate, Alpha par ticker.
  * Genere `logs_prod/backtest_report.json` et courbes d'equity en CSV.
  * **Zero dependance externe** : pas de Docker, pas de Lean CLI, pas de QuantConnect.
- **2026-05-04**: Intégration QuantConnect Lean (Backtesting Institutionnel) — *(supprimé le 2026-05-05, remplacé par backtest_prod.py)*
  * Création de **`src/lean_bridge.py`** : convertit `trading_journal.csv` en signaux Lean (CSV + JSON). Gère les 2 formats historiques (7 et 13 colonnes). Mappe les tickers EUR→US (SXRV.DE→QQQ, CRUDP.PA→USO).
  * Création de **`src/lean_validator.py`** : validation automatisée des changements via backtest Lean avec seuils configurables (Sharpe > 0.5, MaxDD < 25%, Return > -10%).
  * Création de **`run_lean_backtest.py`** : CLI unifié pour exporter les signaux (`--export-signals`), valider (`--validate`), et comparer les algorithmes (`--compare`).
  * Création du projet **`TradingAI-Lean/`** avec :
    - `main.py` : baseline buy-and-hold avec frais T212 (0,1%) et slippage volume-share.
    - `TradingAIFrameworkAlgorithm.py` : framework Alpha→Portfolio→Risk→Execution complet.
    - `AlphaModels/TradingAICompositeAlpha.py` : 5 Alpha Models (Classic, TimesFM, Sentiment, RiskMomentum, VincentGanne) + composite avec les mêmes poids que l'`EnhancedDecisionEngine`.
    - `CustomData/EIAMacroData.py` : data feed personnalisé pour l'API EIA.
  * **Zéro impact sur la production** : le bridge lit uniquement le journal, les fichiers sont additifs.
  * Documentation mise à jour : `README.md`, `SYSTEM_SUMMARY.md`, `TODO.md`, `memory-bank/`.
- **2026-05-04**: Bugfix T212 API et Outils de Diagnostic
  * Correction d'un **`KeyError: 'averagePrice'`** dans `t212_executor.py:327` : l'API T212 peut omettre le champ `averagePrice` dans les réponses positions. Le code utilise désormais un fallback défensif (`currentValue / quantity`).
  * Déplacement des scripts de diagnostic de `logs_prod/` vers `tests/` avec chemins relatifs (`Path(__file__).parent.parent`) :
    - `tests/check_cache.py` : inspection des fichiers Parquet (dates, tailles).
    - `tests/check_db.py` : inspection des bases SQLite (tables, colonnes, dernières lignes).
    - `tests/check_live.py` : prix live via yfinance pour les tickers suivis.
  * Validation d'un cycle complet de trading T212 : CRUDP.PA (HOLD) et SXRV.DE (BUY, position existante synchronisée).
- **2026-04-29**: Cache Auto-Invalidation et Fix Données Périmées
  * Implémentation du **cache stale detection** dans `src/data.py` (lignes 148-154) : si `last_date` du cache Parquet est > 2 jours dans le passé, un `force_refresh` est déclenché automatiquement.
  * Création de **`refresh_cache.py`** : utilitaire CLI pour forcer le rafraîchissement de tous les tickers (`^NDX`, `CL=F`, `SXRV.DE`, `CRUDP.PA`).
  * Scripts d'intégration **`update_decision_engine.py`** et **`update_enhanced_trading.py`** pour injecter TensorTrade dans le pipeline existant.
  * **Suppression des fichiers DB** du tracking git (`performance_monitor.db`, `trading_history.db`) — générés localement uniquement.
- **2026-04-28**: Modèle TensorTrade et Tests
  * Création de **`src/tensortrade_model.py`** : agent RL (PPO via stable-baselines3) dans un environnement Gymnasium custom (SimpleTradingEnv).
  * Ajout de **3 fichiers de tests** : `tests/test_tensortrade_model.py` (158 lignes), `tests/test_tensortrade_integration.py` (139 lignes), `tests/bench_tensortrade.py` (98 lignes).
  * Intégration au moteur de décision avec poids **10%** (réduit llm_text 20%→15%, timesfm 25%→20%).
  * Ajout de `tensortrade>=1.0.4` et `stable-baselines3` dans `pyproject.toml`.
- **2026-04-27**: Fallback MA50 et Robustesse des Données
  * Implémentation du **MA50 fallback** dans `src/data.py` : quand `MA200` est `nan` (historique insuffisant, ex: Urea/UME=F), le système utilise `MA50` comme référence mobile.
  * Résout le problème `Urea (UME=F) MA200 = nan` mentionné dans `TODO.md`.
- **2026-04-16**: Intégration EIA et Modèle OilBench
  * Développement du **`EIAClient`** : analyse automatisée des données de l'Energy Information Administration (v2 API).
  * Métriques fondamentales exploitées : Stocks hebdomadaires de brut (WSTK), Importations mensuelles US, et **Taux d'utilisation des raffineries** (moyenne nationale agrégée par PADD).
  * Intégration des prévisions **STEO** (Short-Term Energy Outlook) pour le WTI, le Brent et la demande mondiale.
  * Création du **`OilBenchModel`** : modèle cognitif spécialisé (Gemma 4) pour le trading du WTI, fusionnant indicateurs techniques, fondamentaux EIA et sentiment.
  * Alignement des signaux : introduction de `STRONG_BUY` et `STRONG_SELL` pour une meilleure granularité décisionnelle.
  * Validation réussie via simulation sur le ticker `CRUDP.PA`.
- **2026-04-15**: Résilience Réseau et Prix Temps Réel
  * Implémentation d'un **circuit breaker yfinance** avec deux trackers séparés (`_yf_info_tracker` pour metadata, `_yf_download_tracker` pour données). Après 3 échecs consécutifs, les appels sont bloqués pendant 120s.
  * Ajout d'un **timeout 10s** sur tous les appels yfinance (avant : 30s+ sans limite). `_yf_ticker_info()` utilise un `ThreadPoolExecutor` avec timeout.
  * **Skip `_yf_ticker_info()`** quand les données viennent du cache parquet — le `info` Yahoo (PEG ratio, etc.) n'est pas utilisé dans le pipeline d'analyse. Gain : ~30-50s par cycle.
  * Nouvelle fonction **`get_t212_price()`** dans `t212_executor.py` : récupère le prix live EUR des ETFs via l'API Trading 212 positions (0.2s au lieu de 10s+ timeout yfinance).
  * **Hiérarchie de prix** : T212 live → MarketDataManager (yfinance) → yfinance history → cache parquet.
  * **`prepare_data_and_features()`** modifié pour utiliser le prix T212 pour les ETF, et le cache pour les indices (`^NDX`, `CL=F`).
  * Séparation des **circuit breakers info vs download** : les timeouts de metadata ne bloquent plus les téléchargements Vincent Ganne.
  * **Résultat** : cycle complet passé de ~7min à ~2min (dev) / ~5min (PROD). Plus aucun timeout yfinance bloquant.
- **2026-04-13**: Modèle Vincent Ganne et Stratégies Avancées
  * Implémentation du **VincentGanneModel** : analyse géopolitique (WTI, Brent, Gaz, Urée, DXY) pour valider les points bas.
  * Mise en place d'un **Verrou Géopolitique** : blocage du BUY si WTI > 94$ (Instabilité énergétique).
  * Introduction du **Risk Management Trend-Aware** : adaptation dynamique des seuils de confiance selon la MM50.
  * Implémentation du **Sizing Progressif** : exposition variable (75%-100%) basée sur le consensus AI.
  * Migration vers **Gemma 3:4b** pour optimiser la consommation RAM tout en maintenant la précision.
  * Automatisation de la désactivation du modèle Vincent Ganne pour le ticker Pétrole (évite l'auto-référence).
- **2026-04-10**: Fiabilisation T212 et Risques
  * Correction du mapping des tickers Trading 212 : `SXRVd_EQ` et `CRUDl_EQ`.
  * Forçage de l'utilisation du signal filtré par le risk manager pour l'exécution réelle.
  * Correction de bugs mineurs (`MarketDataManager` tuple, encodage Windows).
- **2026-04-10**: Migration Cognitive et News
  * Passage de Gemma 3 à **Gemma 4 12B (Unsloth)** pour une meilleure synthèse.
  * Intégration du skill **AlphaEar** via `src/news_fetcher.py`.
  * Création de `schedule.py` pour remplacer l'ancien scheduler complexe.
  * Optimisation des seuils de décision dans `EnhancedDecisionEngine` et `AdvancedRiskManager`.
  * Amélioration du logging du backtester pour une transparence quotidienne.
- **2026-04-13**: Optimisation et Robustesse du Système.
  * Migration vers **Gemma 4 12B (Unsloth)** pour l'analyse textuelle et visuelle.
  * Intégration d'**Hyperliquid** pour le sentiment décentralisé sur le pétrole (Funding Rate, Open Interest).
  * Correction du cycle d'installation (TimesFM doit être patché avant `uv sync`).
  * Support de l'encodage **UTF-8** pour les logs Windows (emojis).
  * Correction de la robustesse des données (FutureWarnings Pandas, KeyErrors T212).
- **2026-04-10**: Intégration de **TimesFM 2.5** (Google Research).
  * Automatisation de l'installation via `setup_timesfm.py` (à exécuter avant `uv sync`).
  * Patch de l'API 2.5 pour permettre l'importation directe de `TimesFM_2p5_200M_torch`.
  * Utilisation des fichiers `safetensors` pour le chargement du modèle.
  * Mise à jour du moteur de décision pour intégrer les signaux de TimesFM avec un poids de 15%.
- **2025-09-15**: Implémentation du portefeuille hypothétique et fiabilisation du suivi des performances.
  * Clarification de l'objectif du projet : système d'aide à la décision, pas un robot de trading autonome.
  * Implémentation d'un portefeuille hypothétique pour suivre la performance des décisions de l'IA.
  * Correction de `performance_monitor.py` pour calculer le `win_rate` sur la base de l'historique des transactions du portefeuille hypothétique.
  * Correction de divers bugs liés à l'intégration des nouvelles fonctionnalités.
  * Dépréciation de `main.py` au profit de `run_now.py` pour l'analyse manuelle.
- **2025-09-12**: Corrections majeures du planificateur intelligent :
  * Réparation de la logique de transition de phase incomplète
  * Correction de la persistance de la date de démarrage du projet
  * Amélioration du chargement de la phase courante depuis la base de données
  * Mise à jour des calculs de progression de phase pour toutes les phases (1-4)
- **2025-09-06**: Remplacement du planificateur défectueux et manquant par un nouveau planificateur robuste (`src/intelligent_scheduler.py`). Correction des erreurs critiques d'exécution (`AttributeError: '_check_performance_alerts'` et `TypeError: JSON serializable`) qui empêchaient l'achèvement des tâches quotidiennes et hebdomadaires.
- **2025-08-20**: Mise à jour du système pour récupérer l'historique complet des données disponibles (période "max") pour QQQ via `yfinance`, augmentant considérablement la quantité de données pour l'entraînement et le backtesting.
- **2025-08-19**: Correction d'un bug empêchant le modèle classique final de s'entraîner correctement à cause de valeurs `NaN` introduites par les nouvelles caractéristiques macroéconomiques. Nettoyage des données implémenté pour assurer la stabilité.
- **2025-08-18**: Correction d'un bug critique qui faisait planter l'application si la variable d'environnement `ALPHA_VANTAGE_API_KEY` n'était pas définie. Le code a été mis à jour pour charger la clé depuis un fichier `.env` et une vérification au démarrage a été ajoutée.


## 1. Statut Actuel
- **Progression Globale**: Phase de Validation en mode "Justesse" (Accuracy First).
- **Dernière Étape Complétée**: Migration vers Gemma 4 et intégration du skill AlphaEar pour le sentiment temps réel.
- **Étape Actuelle**: Exécution autonome via `schedule.py` sur compte démo T212.

## 2. Ce Qui Fonctionne
- **Moteur Hybride Gemma 4**: Utilisation de `gemma4:e4b` pour une analyse tri-modale (texte, vision, news) plus fine.
- **Skill AlphaEar News**: Récupération des tendances financières "hot" (Weibo, WallstreetCN) intégrée au flux décisionnel.
- **Nouveau Scheduler Autonome**: Script `schedule.py` gérant les horaires de marché (8h30-18h00) et l'intervalle de 30 minutes avec dashboard live.
- **Logique de Justesse**: Moteur de décision et gestionnaire de risques calibrés pour prioriser la conservation du capital.
- **TimesFM 2.5**: Prédictions probabilistes de pointe intégrées et stables.
- **TensorTrade / PPO**: Agent RL (stable-baselines3) intégré comme 9ème signal avec poids 10% dans le moteur de décision.
- **Trading 212**: Exécution complète (achat/vente) testée et validée en mode démo.
- **Modèle Quantitatif Classique**: Sélection automatique du meilleur modèle (LR, RF, GB) avec intégration macroéconomique.
- **Cache Auto-Invalidation**: Les données Parquet sont automatiquement rafraîchies si le dernier datapoint date de > 2 jours.
- **MA50 Fallback**: Quand MA200 n'est pas disponible (données insuffisantes, ex: Urea/UME=F), le système utilise MA50 comme référence.

## 3. Ce Qui Reste à Construire
- **Phase 3 : Optimisation** - Optimisation continue des poids modèles et des seuils de confiance.
- **Phase 4 : Maturité** - Déploiement en conditions réelles avec monitoring long terme.

## 4. Problèmes Connus
- **Résolu**: Le planificateur précédent était non fonctionnel et provoquait l'échec de l'analyse quotidienne. Ceci a été résolu avec le nouveau `src/intelligent_scheduler.py`.
- **Résolu**: Problèmes de transition de phase et de persistance de la date de démarrage du projet (corrigés le 12 septembre 2025).
- **Résolu (2025-09-12)**: Corrigé un bug de persistance où la transition de phase n'était pas sauvegardée immédiatement, causant une réinitialisation de la phase au redémarrage du planificateur.

## 5. Corrections Récentes
- **2026-05-04**: Intégration QuantConnect Lean (Backtesting Institutionnel)
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
  * Passage de Gemma 3 à **Gemma 4:e4b** pour une meilleure synthèse.
  * Intégration du skill **AlphaEar** via `src/news_fetcher.py`.
  * Création de `schedule.py` pour remplacer l'ancien scheduler complexe.
  * Optimisation des seuils de décision dans `EnhancedDecisionEngine` et `AdvancedRiskManager`.
  * Amélioration du logging du backtester pour une transparence quotidienne.
- **2026-04-13**: Optimisation et Robustesse du Système.
  * Migration vers **Gemma 4 (e4b)** pour l'analyse textuelle et visuelle.
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

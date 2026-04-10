
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
- **Trading 212**: Exécution complète (achat/vente) testée et validée en mode démo.
- **Modèle Quantitatif Classique**: Sélection automatique du meilleur modèle (LR, RF, GB) avec intégration macroéconomique.

## 3. Ce Qui Reste à Construire
- **Phase 3 : Optimisation** - Le système transitionnera automatiquement vers cette phase le 22 septembre 2025.
- **Phase 4 : Maturité** - Transition prévue pour le 22 octobre 2025.

## 4. Problèmes Connus
- **Résolu**: Le planificateur précédent était non fonctionnel et provoquait l'échec de l'analyse quotidienne. Ceci a été résolu avec le nouveau `src/intelligent_scheduler.py`.
- **Résolu**: Problèmes de transition de phase et de persistance de la date de démarrage du projet (corrigés le 12 septembre 2025).
- **Résolu (2025-09-12)**: Corrigé un bug de persistance où la transition de phase n'était pas sauvegardée immédiatement, causant une réinitialisation de la phase au redémarrage du planificateur.

## 5. Corrections Récentes
- **2026-04-10**: Migration Cognitive et News
  * Passage de Gemma 3 à **Gemma 4:e4b** pour une meilleure synthèse.
  * Intégration du skill **AlphaEar** via `src/news_fetcher.py`.
  * Création de `schedule.py` pour remplacer l'ancien scheduler complexe.
  * Optimisation des seuils de décision dans `EnhancedDecisionEngine` et `AdvancedRiskManager`.
  * Amélioration du logging du backtester pour une transparence quotidienne.
- **2026-04-10**: Intégration de **TimesFM 2.5** (Google Research).
  * Automatisation de l'installation via `setup_timesfm.py` et `uv run setup`.
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

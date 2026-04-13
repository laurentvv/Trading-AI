# Code Review - Trading AI Project

Ce document présente une revue approfondie de la base de code du projet "Trading-AI". Il remplace l'ancienne version et se concentre particulièrement sur la qualité des modèles de décision (Risk Management, Decision Engine), la performance et l'architecture générale. Les problèmes identifiés sont classés par ordre de criticité.

## 🔴 Critique (P0) - À corriger immédiatement pour éviter des pertes financières

### Task 1: Biais haussier (Bullish Bias) hardcodé
- **Fichier:** `src/enhanced_decision_engine.py` (lignes 190 et logique de scoring)
- **Description:** Bien qu'il y ait une constante `BULLISH_BIAS = 0.0` récemment introduite, l'ancienne logique incrémente toujours le score de manière déséquilibrée (`score += 10`, `max_score += 10`) en faveur de l'achat sans justification symétrique pour la vente. Par exemple, le modèle de Vincent Ganne n'ajoute des points que pour évaluer les points bas (bottoms) via le WTI/Brent et les indices au-dessus de la MA200, créant un biais structurel en faveur du BUY.
- **Action:** Refactoriser le système de points (scoring) de Vincent Ganne (et du système en général) pour qu'il puisse générer des scores négatifs symétriques pour les configurations baissières (ex: WTI au-dessus des max, MA200 cassée à la baisse).

### Task 2: Problème d'architecture N+1 Query dans la base de données de performance
- **Fichier:** `src/adaptive_weight_manager.py`
- **Description:** La méthode `calculate_all_models_performance` fait une requête SQLite potentiellement lourde en boucle. De plus, les ouvertures et fermetures de connexion (`sqlite3.connect`) sont multiples et non optimisées (ex: dans les insertions ou mises à jour). Cela provoquera de forts ralentissements avec l'accumulation de données.
- **Action:** Utiliser `cursor.executemany` avec une seule connexion persistante ou un pool de connexions (ex: via SQLAlchemy ou du context management natif). Et regrouper les lectures pour l'ensemble des modèles dans une seule requête SQL si possible.

## 🟠 Haut (P1) - Défauts architecturaux ou problèmes de performance majeurs

### Task 3: Risque de suppression non sécurisée de la base de données (Tests manuels)
- **Fichier:** `test_full_cycle.py`
- **Description:** Le script contient `os.remove(db)` (lignes 44, 48) sans aucune confirmation explicite ni validation de l'environnement (ex: vérifier qu'on est en mode `DEV` ou `TEST`). S'il est exécuté par erreur en production, l'historique complet des trades et de la performance des modèles est détruit irrémédiablement.
- **Action:** Ajouter une confirmation stricte (input utilisateur ou flag `--force`) et s'assurer que les bases de production sont protégées ou dans des dossiers distincts.

### Task 4: Sys.path.append abusifs pour les imports
- **Fichiers:** `main.py`, `backtest_engine.py`, `run_short_backtest.py`, `src/news_fetcher.py`, `src/t212_executor.py`, `src/read_simul.py`
- **Description:** Presque tous les points d'entrée utilisent `sys.path.append(str(Path(__file__).parent / 'src'))` pour résoudre les imports. C'est un anti-pattern Python fragile.
- **Action:** Retirer tous les `sys.path.append` et configurer le projet comme un package installable (`pyproject.toml` avec mode éditable `uv pip install -e .` ou utilisation stricte de `PYTHONPATH` au niveau du conteneur/environnement virtuel).

### Task 5: Traitement Séquentiel des LLMs et APIs
- **Fichier:** L'ensemble du processus de décision (`enhanced_decision_engine.py` / `main.py`)
- **Description:** Les appels au LLM textuel, LLM visuel, Web Search et API externes sont effectués de manière séquentielle, ce qui peut prendre beaucoup de temps (plusieurs dizaines de secondes par cycle).
- **Action:** Implémenter `asyncio` et `aiohttp` (ou des ThreadPools) pour paralléliser la récupération de l'actualité, l'analyse de sentiment et les inférences LLM.

### Task 6: Utilisation inefficace de Pandas (iterrows)
- **Fichier:** `src/adaptive_weight_manager.py` (méthode `calculate_model_performance`)
- **Description:** Utilisation de méthodes itératives peu performantes pour l'analyse des signaux historiques, ce qui ralentit considérablement la mise à jour des poids adaptatifs sur un gros historique.
- **Action:** Remplacer les itérations par des opérations vectorisées Pandas ou utiliser `itertuples(index=False)` lorsque l'itération est absolument nécessaire.

## 🟡 Moyen (P2) - Qualité du code, tests manquants et modèles perfectibles

### Task 7: Risk Manager et Kelly Criterion Incomplets
- **Fichier:** `src/advanced_risk_manager.py`
- **Description:** Le calcul du critère de Kelly repose sur des moyennes de gains/pertes (`avg_win`, `avg_loss`) passés en paramètres (`historical_performance`), mais ces données ne sont pas toujours robustes ou disponibles dans le flux standard, et le Risk Manager applique un Kelly Fractionnaire hardcodé (`max(0, min(0.25, kelly_fraction * 0.25))`). Ce hardcoding du fractionnement limite l'adaptabilité du système.
- **Action:** Rendre le fractionnement de Kelly configurable via une variable d'environnement ou selon le "Risk Level" dynamique. Ajouter des tests unitaires pour valider les mathématiques du `advanced_risk_manager.py`.

### Task 8: Incohérence des "Base Weights" (Poids par défaut)
- **Fichiers:** `src/adaptive_weight_manager.py` et `src/enhanced_decision_engine.py`
- **Description:** Les poids de base des modèles (`base_weights`) sont définis deux fois avec des valeurs légèrement différentes. `enhanced_decision_engine.py` utilise {classic: 0.15, text: 0.20, visual: 0.15, sentiment: 0.10, timesfm: 0.20} tandis que `adaptive_weight_manager.py` utilise {classic: 0.15, text: 0.25, visual: 0.20, sentiment: 0.15, timesfm: 0.25}.
- **Action:** Définir les poids par défaut dans une source unique de vérité (ex: un fichier de configuration, une constante partagée, ou une dataclass globale).

### Task 9: Retraitement constant des données dans `classic_model.py`
- **Fichier:** `src/classic_model.py`
- **Description:** Lors du `train_ensemble_model`, le modèle est ré-entraîné fréquemment, ce qui implique de refaire l'imputation des NaNs et le scaling (`StandardScaler`) à chaque appel. Sur des backtests longs, cela consomme énormément de CPU.
- **Action:** Mettre en cache (ex: via `joblib`) les modèles classiques entraînés s'ils ont été mis à jour il y a moins de 24h/1h, plutôt que de tout refaire systématiquement.

### Task 10: Tests Insuffisants pour l'Exécution (T212) et le Risque
- **Fichiers:** `tests/`
- **Description:** Bien qu'il y ait eu un correctif sur les syntaxes via `ruff`, la couverture de test sur les vrais composants critiques (Risk Management mathématique, Exécuteur Trading 212) est minime ou inexistante.
- **Action:** Implémenter des tests unitaires pour `advanced_risk_manager.py` (vérifier les limites du RiskLevel) et `t212_executor.py` avec des mocks robustes.

## 🟢 Bas (P3) - Nettoyage général et dette technique mineure

### Task 11: Langue Mixte (Français/Anglais)
- **Fichier:** Codebase complète
- **Description:** Le code mélange des docstrings et des logs en français et en anglais, rendant le projet moins professionnel s'il doit être open-source ou collaboratif.
- **Action:** Standardiser les logs, docstrings et variables sur une seule langue (l'anglais est recommandé).

### Task 12: Chemins locaux "En dur"
- **Fichier:** `src/news_fetcher.py` (et potentiellement d'autres)
- **Description:** Utilisation de chemins statiques du type `ALPHA_EAR_PATH = Path("D:/GIT/fork/Trading-AI/.agents/.../scripts")`.
- **Action:** Utiliser des variables d'environnement ou rendre les chemins relatifs à la racine du projet `Path(__file__).parent.parent`.

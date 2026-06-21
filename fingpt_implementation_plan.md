# Plan d'Implémentation de FinGPT pour Agent IA

Ce document détaille les étapes à suivre pour intégrer les capacités de **FinGPT** (analyse de sentiment avancée et prédiction/forecasting) au sein de notre architecture de trading hybride.

L'objectif est d'ajouter FinGPT comme un nouveau modèle de décision indépendant qui participera au consensus via `EnhancedDecisionEngine`, en utilisant Ollama pour l'inférence locale via des modèles au format GGUF.

## Objectifs
1. **Analyse de sentiment et prédiction** : Tirer parti des modèles affinés pour la finance (FinGPT) pour générer des signaux de trading plus précis.
2. **Inférence locale (Ollama)** : Utiliser des modèles FinGPT au format `GGUF` pour être compatibles avec notre infrastructure Ollama locale.
3. **Module Indépendant** : Créer un module `src/fingpt_model.py` s'intégrant proprement dans l'architecture existante.
4. **Intégration au Consensus** : Ajouter ce modèle à `EnhancedDecisionEngine` et gérer ses poids via `config_weights.py`.

---

## Étapes d'Implémentation Détaillées

### Étape 1 : Recherche et Configuration du Modèle FinGPT (GGUF)
* **Action requise :** Le système Ollama nécessite des modèles au format `.gguf`. FinGPT étant principalement distribué sous forme de poids Llama/ChatGLM (LoRA) sur HuggingFace, il faut :
  - Identifier un modèle FinGPT converti en GGUF (par ex. sur HuggingFace Hub, chercher `FinGPT GGUF` ou `FinGPT-Forecaster GGUF`).
  - *Instruction pour l'agent :* Ajouter un script (ex: dans `scripts/setup_fingpt.sh`) expliquant comment télécharger ce modèle GGUF et créer un `Modelfile` pour Ollama (ex: `FROM ./fingpt.gguf`), puis exécuter `ollama create fingpt -f Modelfile`.
  - Assigner le nom du modèle Ollama (ex: `fingpt:latest`) dans la configuration globale ou locale du modèle.

### Étape 2 : Création du module `src/fingpt_model.py`
* **Action requise :** Créer un nouveau fichier `src/fingpt_model.py`.
* **Spécifications du module :**
  - Importer les utilitaires de `src/llm_client.py` pour communiquer avec l'API locale d'Ollama (`http://localhost:11434/api/generate`) ou effectuer des requêtes HTTP directes.
  - Définir une fonction principale (ex: `get_fingpt_decision(ticker: str, recent_news: str, market_data: dict) -> dict`).
  - **Prompt Engineering :** Formater le prompt en respectant les templates d'instruction spécifiques de FinGPT (qui sont souvent basés sur Llama 2 ou 3) pour l'analyse de sentiment et la prédiction. Fournir les actualités (via `news_fetcher.py`) et les données de marché de base en contexte.
  - **Parsing de la réponse :** Analyser la sortie du modèle pour extraire :
    - `signal` (BUY, SELL, HOLD)
    - `confidence` (float entre 0.0 et 1.0)
    - `reason` (explication du raisonnement)
  - **Robustesse :** Gérer les timeouts, les exceptions réseau (`requests.RequestException`), et retourner une décision neutre par défaut (`HOLD`, `confidence: 0.0`) en cas d'échec.

### Étape 3 : Mise à jour de la configuration des poids (`src/config_weights.py`)
* **Action requise :** Modifier `src/config_weights.py` pour inclure le nouveau modèle.
* **Détails :**
  - Ajouter la clé `"fingpt"` au dictionnaire `DEFAULT_BASE_WEIGHTS`.
  - Assigner un poids initial pertinent (par exemple `0.10` ou `0.15`).
  - *Note :* `AdaptiveWeightManager` se chargera de la normalisation globale des poids par la suite.

### Étape 4 : Intégration dans le Moteur de Décision (`src/enhanced_decision_engine.py`)
* **Action requise :** Intégrer les prédictions de FinGPT au calcul du consensus global.
* **Détails :**
  - Dans l'orchestrateur (ex: `main.py` ou le script d'exécution principal), appeler `get_fingpt_decision` en amont pour générer la prédiction.
  - Passer cette prédiction dans le dictionnaire `model_predictions` fourni à `EnhancedDecisionEngine.make_enhanced_decision`.
  - La clé utilisée dans `model_predictions` **doit être strictement** `"fingpt"`, afin de correspondre à la clé définie dans `DEFAULT_BASE_WEIGHTS`. Ceci est crucial pour éviter la désynchronisation signalée dans la mémoire du projet.

### Étape 5 : Tests Unitaires et Validation
* **Action requise :** Créer un fichier de tests `tests/test_fingpt_model.py`.
* **Détails :**
  - Utiliser `sys.path.insert(0, str(Path(__file__).parent.parent))` pour les imports.
  - Mocker l'API Ollama (avec `unittest.mock.patch`) pour simuler les réponses du LLM et éviter les appels réseau réels pendant l'intégration continue.
  - Tester les scénarios de réussite (parsing d'un BUY, d'un SELL).
  - Tester les blocs `except` spécifiques (ex: `ValueError, requests.RequestException`) pour vérifier que le système fallback bien sur un `HOLD` sans planter.

## Règles Strictes pour l'Agent de Codage
- **Environnement Hors-Ligne :** Ne pas supprimer le `.venv` ni forcer des `uv sync` massifs qui échoueraient.
- **Sécurité :** Ne pas utiliser de formatage f-string pour les schémas SQLite s'il y en a. Aucune exécution arbitraire de code généré par l'IA.
- **Tests (Mantra) :** "Non jamais sans tester". Assurez-vous que les tests unitaires passent (`uv run python -m unittest`).
- **Règles Métier :** Le modèle ne doit *jamais* recommander de vendre à perte (SELL en dessous du coût moyen). Vérifier que la logique le prend en compte ou que le module principal l'intercepte.

# QWEN.md - Système de Trading IA pour les ETF du NASDAQ

## Aperçu du Projet

Ce projet est un système sophistiqué de support à la décision de trading pour les ETF du NASDAQ (par exemple, QQQ). Il utilise une approche **IA hybride tri-modale** pour générer des signaux de trading robustes en combinant les idées de trois modèles distincts :

1.  **Modèle Quantitatif Classique :** Un classifieur d'ensemble `scikit-learn` entraîné sur des indicateurs techniques et des **données macroéconomiques** (par exemple, taux d'intérêt, inflation, chômage).
2.  **LLM Basé sur le Texte :** Un LLM (accessible via Ollama, par exemple, Gemma 3) qui analyse les données de marché numériques et fournit une décision et une analyse textuelles.
3.  **LLM Visuel :** Un LLM qui effectue une analyse technique en interprétant une image de graphique financier générée.

Le système intègre ces modèles, exécute un backtest robuste par validation walk-forward pour évaluer les performances historiques, et fournit une décision hybride finale pour le point de données le plus récent. Il est conçu pour le support à la décision, et non pour le trading automatisé.

## Technologies Clés

*   **Langage :** Python 3.10+
*   **Données & Calculs Numériques :** `pandas`, `numpy`, `yfinance` (pour la récupération des données), `pyarrow` (pour la mise en cache Parquet)
*   **Framework ML :** `scikit-learn`
*   **Visualisation :** `matplotlib`, `seaborn`, `mplfinance` (pour la génération de graphiques)
*   **Interface IA/LLM :** `requests` (pour interagir avec une instance locale Ollama)
*   **Utilitaires :** `python-dotenv` (pour les variables d'environnement), `tqdm` (barres de progression), `rich` (sortie console formatée)

## Structure du Projet

```
.
├── memory-bank/             # Documentation complète du projet (contexte, progression, décisions)
├── src/                     # Code source
│   ├── main.py              # Script orchestrateur principal
│   ├── data.py              # Logique de récupération et de mise en cache des données
│   ├── features.py          # Ingénierie des caractéristiques (indicateurs techniques)
│   ├── classic_model.py     # Entraînement et prédiction du modèle Scikit-learn
│   ├── llm_client.py        # Client pour interagir avec les LLM textuels et visuels via Ollama
│   ├── chart_generator.py   # Génère des images de graphiques financiers pour l'analyse IA visuelle
│   ├── backtest.py          # Moteur de backtesting par validation walk-forward avec coûts de transaction
│   ├── sentiment_analysis.py # Analyse le sentiment à partir des titres d'actualités
│   └── news_fetcher.py      # Récupère les titres d'actualités récents pour l'analyse de sentiment
├── data_cache/              # Répertoire pour les données de marché mises en cache (fichiers Parquet)
├── requirements.txt         # Dépendances Python
├── .env                     # (Créé par l'utilisateur) Fichier pour stocker les clés API sensibles (par exemple, ALPHA_VANTAGE_API_KEY)
├── README.md                # Aperçu du projet et instructions d'utilisation
└── QWEN.md                  # Ce fichier (Contexte instructif pour l'IA)
```

## Construction et Exécution

### Prérequis

1.  **Python :** Assurez-vous que Python 3.10 ou une version supérieure est installé.
2.  **Ollama :** Installez et exécutez [Ollama](https://ollama.com/) localement. Téléchargez un LLM approprié (par exemple, `gemma3:27b`) :
    ```bash
    ollama pull gemma3:27b
    ```
3.  **Clé API :** Obtenez une clé API auprès d'Alpha Vantage pour l'analyse de sentiment.

### Configuration

1.  **Cloner le dépôt :**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Créer un Environnement Virtuel (Recommandé) :**
    ```bash
    python -m venv venv
    # Sous Windows
    venv\Scripts\activate
    # Sous macOS/Linux
    # source venv/bin/activate
    ```
3.  **Installer les Dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configurer la Clé API :**
    Créez un fichier nommé `.env` à la racine du projet et ajoutez-y votre clé API Alpha Vantage :
    ```env
    ALPHA_VANTAGE_API_KEY="VOTRE_CLE_API_REELLE_ICI"
    ```

### Utilisation

Pour exécuter le pipeline complet du système de trading, lancez le script principal depuis le répertoire racine du projet :

```bash
python src/main.py
```

**Ce que fait `src/main.py` :**

1.  **Récupère les Données :** Charge ou télécharge les données historiques de marché pour QQQ (et VIX) en utilisant `yfinance`, les mettant en cache localement dans `data_cache/`. Utilise désormais l'historique complet disponible ("max").
2.  **Ingénierie des Caractéristiques :** Calcule les indicateurs techniques (RSI, MACD, Bandes de Bollinger, Moyennes Mobiles, etc.).
3.  **Backtesting :** Exécute un backtest par validation walk-forward en utilisant les signaux du modèle classique, simulant le comportement du LLM pour l'évaluation des performances. Affiche des métriques telles que le Ratio de Sharpe et le Drawdown.
4.  **Génère le Graphique :** Crée une image `trading_chart.png` des 6 derniers mois de données (chandeliers, MM, RSI, MACD) pour le LLM visuel.
5.  **Prend la Décision Finale :**
    *   Entraîne le modèle classique final sur toutes les données disponibles.
    *   Interroge le **LLM textuel** avec les derniers indicateurs numériques.
    *   Interroge le **LLM visuel** avec `trading_chart.png`.
    *   Récupère le sentiment des actualités via `news_fetcher.py` et l'analyse.
    *   Combine les sorties des quatre modèles en une décision hybride finale en utilisant une pondération.
6.  **Affiche les Résultats :** Montre une sortie formatée dans la console détaillant la décision de chaque modèle et le signal hybride final ("ACHAT FORT", "ACHAT", "NEUTRE", "VENTE", "VENTE FORTE") avec un score de fiabilité.
7.  **Enregistre l'Analyse :** Génère un graphique `backtest_analysis.png` montrant la performance du backtest par rapport à un achat et maintien.

## Conventions de Développement

*   **Modularité :** Le code est organisé en modules distincts (`src/`) pour les données, les caractéristiques, les modèles, l'interaction LLM, les graphiques et le backtesting, favorisant la maintenabilité et la clarté.
*   **Documentation :** Le projet utilise un système de "Banque de Mémoire" (`memory-bank/`) pour stocker le contexte évolutif, les décisions d'architecture et la progression. C'est la source de vérité principale pour comprendre la conception du projet.
*   **Configuration :** Les clés API et autres secrets sont gérés via un fichier `.env`, et non codés en dur.
*   **Mise en Cache des Données :** Les données de marché sont mises en cache sous forme de fichiers Parquet pour améliorer les performances et réduire les appels API redondants.
*   **Backtesting Robuste :** Une approche de validation walk-forward est utilisée pour simuler des conditions de trading réalistes et éviter le biais de prédiction.
*   **Journalisation :** Utilise le module `logging` de Python pour une sortie console informative.
*   **Sortie Formatée :** Utilise `rich` pour fournir une sortie claire, structurée et colorisée pour la décision finale.
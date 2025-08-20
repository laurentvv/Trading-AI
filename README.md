# Système de Trading IA Hybride Tri-Modal pour les ETF du NASDAQ

Ce projet est un système sophistiqué de support à la décision de trading qui utilise une approche IA hybride tri-modale pour générer des signaux de trading pour les ETF du NASDAQ. Il combine un modèle quantitatif traditionnel, un Grand Modèle de Langage (LLM) basé sur du texte et un LLM visuel (multi-modal) pour une analyse robuste et nuancée.

## Fonctionnalités Clés

- **Moteur IA Hybride Tri-Modal** : Combine trois modèles d'IA différents pour une décision basée sur un consensus :
    1.  Un classifieur `scikit-learn` entraîné sur des indicateurs techniques quantitatifs et des **données macroéconomiques** (par exemple, taux d'intérêt, inflation).
    2.  Un LLM qui effectue une analyse sur les données numériques brutes.
    3.  Un LLM multi-modal qui effectue une analyse visuelle sur une image de graphique générée.
- **Backtesting Robuste** : Utilise une méthodologie de **validation par walk-forward** pour éviter le biais de prédiction et fournir une évaluation réaliste des performances historiques de la stratégie.
- **Simulation des Coûts de Transaction** : Le backtester tient compte des coûts de transaction pour des calculs de rendement plus réalistes.
- **Mise en Cache Locale des Données** : Les données de marché récupérées sont mises en cache localement dans des fichiers Parquet pour accélérer les exécutions suivantes. Les données macroéconomiques sont également mises en cache.
- **Base de Code Modulaire** : Le code est organisé dans une structure propre et modulaire pour faciliter la maintenance et l'extension.
- **Documentation Complète** : L'évolution, l'architecture et le contexte du projet sont documentés de manière méticuleuse dans le répertoire `memory-bank/`, suivant un processus de développement axé sur la documentation.

## Stack Technologique

- **Python 3.10+**
- **Données & Calculs Numériques :** `pandas`, `numpy`, `yfinance` (récupération des données), `pyarrow` (mise en cache Parquet)
- **Framework ML :** `scikit-learn`
- **Interface IA/LLM :** `requests` (interaction avec Ollama), `ollama` (serveur LLM local, testé avec `gemma3:27b`)
- **Visualisation :** `matplotlib`, `seaborn`, `mplfinance` (graphiques financiers)
- **Utilitaires :** `tqdm` (barres de progression), `rich` (sortie console formatée), `python-dotenv` (variables d'environnement)

## Prérequis

Avant de commencer, assurez-vous d'avoir installé :
- Python 3.10 ou supérieur.
- [Ollama](https://ollama.com/) en cours d'exécution localement.
- Un LLM téléchargé (par exemple, Gemma 3) : `ollama pull gemma3:27b`

## Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Créer et activer un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sous Windows, utilisez `venv\Scripts\activate`
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurer votre Clé API :**
    Créez un fichier nommé `.env` à la racine du projet et ajoutez-y votre clé API Alpha Vantage comme suit :
    ```
    ALPHA_VANTAGE_API_KEY="VOTRE_CLE_API_ICI"
    ```

## Utilisation

Pour exécuter le pipeline complet du système de trading, lancez le script principal depuis le répertoire racine :

```bash
python src/main.py
```

**Ce que fait `src/main.py` :**

1.  **Récupère les Données :** Charge ou télécharge les données historiques de marché pour QQQ (et VIX) en utilisant `yfinance`, les mettant en cache localement dans `data_cache/`. Utilise l'historique complet disponible ("max").
2.  **Ingénierie des Caractéristiques :** Calcule les indicateurs techniques (RSI, MACD, Bandes de Bollinger, Moyennes Mobiles, etc.).
3.  **Backtesting :** Exécute un backtest par validation walk-forward en utilisant les signaux du modèle classique, simulant le comportement du LLM pour l'évaluation des performances. Affiche des métriques telles que le Ratio de Sharpe et le Drawdown.
4.  **Génère le Graphique :** Crée une image `trading_chart.png` des 6 derniers mois de données (chandeliers, MM, RSI, MACD) pour l'analyse du LLM visuel.
5.  **Prend la Décision Finale :**
    *   Entraîne le modèle classique final sur toutes les données disponibles.
    *   Interroge le **LLM textuel** avec les derniers indicateurs numériques.
    *   Interroge le **LLM visuel** avec `trading_chart.png`.
    *   Récupère le sentiment des actualités via `news_fetcher.py` et l'analyse.
    *   Combine les sorties des quatre modèles en une décision hybride finale en utilisant une pondération.
6.  **Affiche les Résultats :** Montre une sortie formatée dans la console détaillant la décision de chaque modèle et le signal hybride final ("ACHAT FORT", "ACHAT", "NEUTRE", "VENTE", "VENTE FORTE") avec un score de fiabilité.
7.  **Enregistre l'Analyse :** Génère un graphique `backtest_analysis.png` montrant la performance du backtest par rapport à un achat et maintien.

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
├── .env                     # (Créé par l'utilisateur) Fichier pour stocker les clés API sensibles (ALPHA_VANTAGE_API_KEY)
└── README.md                # Ce fichier
```

## Conventions de Développement

*   **Modularité :** Le code est organisé en modules distincts (`src/`) pour les données, les caractéristiques, les modèles, l'interaction LLM, les graphiques et le backtesting, favorisant la maintenabilité et la clarté.
*   **Documentation :** Le projet utilise un système de "Banque de Mémoire" (`memory-bank/`) pour stocker le contexte évolutif, les décisions d'architecture et la progression. C'est la source de vérité principale pour comprendre la conception du projet.
*   **Configuration :** Les clés API et autres secrets sont gérés via un fichier `.env`, et non codés en dur.
*   **Mise en Cache des Données :** Les données de marché sont mises en cache sous forme de fichiers Parquet pour améliorer les performances et réduire les appels API redondants.
*   **Backtesting Robuste :** Une approche de validation walk-forward est utilisée pour simuler des conditions de trading réalistes et éviter le biais de prédiction.
*   **Journalisation :** Utilise le module `logging` de Python pour une sortie console informative.
*   **Sortie Formatée :** Utilise `rich` pour fournir une sortie claire, structurée et colorisée pour la décision finale.

## La Banque de Mémoire (Memory Bank)

Ce projet suit une philosophie de "Banque de Mémoire". Le répertoire `memory-bank/` est la source de vérité unique pour le contexte, l'architecture et la progression du projet. Il est conçu pour être une documentation vivante permettant à tout développeur (ou assistant IA) de rapidement comprendre l'état du projet.

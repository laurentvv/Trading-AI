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
- **Pandas & NumPy** : Pour la manipulation des données.
- **Scikit-learn** : Pour le modèle ML classique.
- **Ollama** : Pour servir le LLM local (testé avec `gemma3:27b`).
- **yfinance** : Pour récupérer les données de marché.
- **Matplotlib & Seaborn** : Pour les graphiques.
- **PyArrow** : Pour la gestion des fichiers Parquet.
- **Tqdm** : Pour les barres de progression.
- **mplfinance** : Pour générer les graphiques financiers.

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

Pour exécuter le système de trading, lancez le script principal depuis le répertoire racine :

```bash
python src/main.py
```

Le script effectuera les actions suivantes :
1.  Récupérer ou charger les données de marché depuis le cache.
2.  Exécuter le backtest par walk-forward et afficher le résumé des performances dans la console.
3.  Générer un graphique des données de marché récentes.
4.  Générer une décision de trading finale pour le point de données le plus récent en combinant les sorties du modèle classique, d'un LLM basé sur le texte et de l'analyse visuelle d'un LLM sur le graphique.
5.  Sauvegarder un graphique de l'analyse du backtest sous le nom `backtest_analysis.png`.

## Structure du Projet

```
.
├── memory-bank/        # Documentation du projet (contexte, progression, décisions)
├── src/                # Code source
│   ├── main.py         # Script orchestrateur principal
│   ├── data.py         # Récupération et mise en cache des données
│   ├── features.py     # Ingénierie des caractéristiques
│   ├── classic_model.py # Entraînement du modèle scikit-learn
│   ├── llm_client.py   # Client pour les LLM textuels et visuels
│   ├── chart_generator.py # Crée des images de graphiques pour l'IA visuelle
│   └── backtest.py     # Logique de backtesting
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier
```

## La Banque de Mémoire (Memory Bank)

Ce projet suit une philosophie de "Banque de Mémoire". Le répertoire `memory-bank/` est la source de vérité unique pour le contexte, l'architecture et la progression du projet. Il est conçu pour être une documentation vivante permettant à tout développeur (ou assistant IA) de rapidement comprendre l'état du projet.

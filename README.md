
<div align="center">
  <br />
  <h1>📈 Système de Trading IA Hybride 📈</h1>
  <p>
    Un système expert d'aide à la décision pour le trading d'ETFs sur le NASDAQ, exploitant une intelligence artificielle hybride tri-modale pour des signaux de trading robustes et nuancés.
  </p>
</div>

<div align="center">

[![Statut du Projet](https://img.shields.io/badge/status-en--développement-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Licence](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="enhanced_performance_dashboard.png" alt="Dashboard de Performance" width="800"/>
</p>

---

## 📚 Table des Matières

- [🌟 À Propos du Projet](#-à-propos-du-projet)
  - [✨ Fonctionnalités Clés](#-fonctionnalités-clés)
  - [💻 Stack Technologique](#-stack-technologique)
- [📂 Structure du Projet](#-structure-du-projet)
- [🚀 Démarrage Rapide](#-démarrage-rapide)
  - [✅ Prérequis](#-prérequis)
  - [⚙️ Installation](#️-installation)
- [🛠️ Utilisation](#️-utilisation)
  - [Analyse Manuelle](#-analyse-manuelle)
  - [Analyse Automatisée avec le Planificateur Intelligent](#-analyse-automatisée-avec-le-planificateur-intelligent)
- [🤝 Contribuer](#-contribuer)
- [📜 Licence](#-licence)
- [📧 Contact](#-contact)

---

## 🌟 À Propos du Projet

Ce projet est un système expert d'aide à la décision pour le trading d'ETFs, utilisant une approche d'IA hybride tri-modale. Il est conçu pour fournir une analyse complète et robuste en combinant plusieurs perspectives d'IA.

### 🚀 Stratégie Dual-Ticker (Analyse vs Trading)
Le système utilise une approche innovante pour maximiser la précision des modèles :
- **Analyse Haute Fidélité** : Les modèles IA analysent les **indices de référence mondiaux** (`^NDX` pour le Nasdaq, `CL=F` pour le pétrole WTI). Ces indices offrent un historique plus long et des tendances plus "pures", sans le bruit lié aux horaires de cotation ou aux frais des ETFs.
- **Exécution sur ETF** : Les ordres réels sont passés sur les tickers correspondants sur **Trading 212** (`SXRV.DE`, `CRUDP.PA`), en utilisant les prix réels du marché pour le dimensionnement des positions.

### 🧠 Moteur IA Hybride & Stratégie "Aggressive Growth"
Le système fusionne cinq signaux distincts avec un biais optimisé pour la performance :
1.  **Modèle Quantitatif Classique** : Ensemble RandomForest/GradientBoosting/LogisticRegression entraîné sur indicateurs techniques et macroéconomiques.
2.  **TimesFM 2.5 (Google Research)** : Modèle de fondation de pointe pour la prévision de séries temporelles.
3.  **LLM Textuel (Gemma 4:e4b)** : Analyse contextuelle des données brutes et des actualités en temps réel via le skill **AlphaEar**.
4.  **LLM Visuel (Gemma 4:e4b)** : Analyse directe des graphiques techniques (`enhanced_trading_chart.png`).
5.  **Sentiment Analysis & Deep Web** : Intégration de **DuckDuckGo Search** et **Hyperliquid** (DEX) pour capturer les flux 24/7 et le sentiment des traders crypto comme indicateur avancé.

### 🚀 Stratégie "Beat the Market"
Pour surpasser le **Buy & Hold** du Nasdaq, le système a été calibré pour être plus agressif en tendance haussière :
- **Biais Bullish** : Le moteur de décision injecte un bonus de score pour les indices afin de favoriser la capture de tendance.
- **Gestion des Risques Dynamique** : Les seuils de confiance pour l'achat sont abaissés en période de tendance, et la taille de position maximale a été augmentée à 5%.
- **Veto Assoupli** : Le filtre de volatilité est plus tolérant pour éviter de rater les reprises rapides de marché.

L'objectif est de produire une décision finale (`ACHAT`, `VENTE`, `HOLD`) avec une priorité absolue sur la **justesse** (Accuracy First).

### 🧘 Philosophie de Décision : "La Prudence Cognitive"
Contrairement aux algorithmes de trading classiques qui paniquent dès que la volatilité explose, ce système applique une approche d'investisseur averti :
- **Consensus Fort Requis** : Un modèle quantitatif (Classic) peut crier au loup (`SELL`), mais si les modèles cognitifs (LLM Texte, Vision, TimesFM) restent neutres, le système privilégiera le `HOLD`.
- **Filtre de Confiance** : Une décision de mouvement (Achat ou Vente) n'est validée que si la confiance globale dépasse un seuil de sécurité (généralement 40%). En dessous, le système considère le signal comme du "bruit" et reste en attente.
- **Protection du Capital** : En mode `VERY_HIGH` risque, le `HOLD` sert de bouclier. Il empêche d'entrer sur un marché instable et évite de sortir prématurément sur une simple correction technique si les fondamentaux (News/Vision) ne confirment pas un crash imminent.

### ✨ Fonctionnalités Clés

- **Approche Dual-Ticker** : Analyse l'indice (^NDX, CL=F), trade l'ETF (SXRV.DE, CRUDP.PA).
- **Stratégie "Beat the Market"** : Gestion des risques assouplie et biais haussier renforcé pour surpasser le **Buy & Hold** du Nasdaq.
- **Leading Indicators (Hyperliquid)** : Surveillance en temps réel des actifs `flx:OIL` et `NDX` sur Hyperliquid pour anticiper les ouvertures de marché.
- **Deep Web Research** : Recherche active via DuckDuckGo et extraction de contenu propre via **Crawl4AI**.
- **Backtest Haute Performance** : Moteur optimisé (vectorisation Pandas) capable de simuler plusieurs années en quelques minutes.
- **Scheduler Automatisé** : Nouveau script `schedule.py` pour une exécution continue (8h30-18h00) sur serveur.

### 💻 Stack Technologique

- **Langage** : `Python 3.12+` (gestion via `uv`)
- **Calculs & Données** : `pandas`, `numpy`, `yfinance`, `pyarrow`
- **Machine Learning & Time Series** : `scikit-learn`, `TimesFM 2.5`
- **IA & LLM** : `ollama` (Gemma 4:e4b), `requests`
- **Recherche & Crawl** : `ddgs` (DuckDuckGo), `crawl4ai` (Playwright)
- **Crypto / Leading Indicators** : `hyperliquid-python-sdk`
- **Visualisation** : `matplotlib`, `seaborn`, `rich`

---

## 📂 Structure du Projet

Le projet est organisé de manière modulaire pour une meilleure maintenabilité.

```
Trading-AI/
├── src/                     # Code source principal
│   ├── intelligent_scheduler.py # Planificateur intelligent pour l'exécution automatique
│   ├── enhanced_trading_example.py # Logique principale de l'analyse de trading
│   ├── run_now.py           # Point d'entrée pour l'analyse manuelle
│   ├── data.py              # Gestion des données (API, cache)
│   ├── features.py          # Création des indicateurs techniques
│   ├── classic_model.py     # Modèle quantitatif Scikit-learn
│   ├── llm_client.py        # Client pour les modèles de langage (Ollama)
│   ├── backtest.py          # Moteur de backtesting
│   └── ...                  # Autres modules (graphiques, XAI, etc.)
├── data_cache/              # Données de marché mises en cache
├── memory-bank/             # Documentation et contexte pour l'agent IA
├── .env                     # Fichier pour les clés d'API (à créer)
├── requirements.txt         # Dépendances Python
├── start_scheduler.bat      # Script de démarrage pour le planificateur (Windows)
└── README.md                # Cette documentation
```

---

## 🚀 Démarrage Rapide

Suivez ces étapes pour mettre en place votre environnement de développement local.

### ✅ Prérequis

- Python 3.10 ou supérieur
- [Ollama](https://ollama.com/) installé et en cours d'exécution localement.
- Un modèle LLM téléchargé (ex: `ollama pull gemma3:4b`)

### ⚙️ Installation

1.  **Clonez le dépôt :**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```

2.  **Installez `uv` (si ce n'est pas déjà fait) :**
    Consultez [astral.sh/uv](https://astral.sh/uv) pour les instructions d'installation.

3.  **Initialisez TimesFM 2.5 (Étape CRUCIALE) :**
    Le projet dépend d'une version spécifique de TimesFM située dans `vendor/timesfm`. Exécutez ce script **AVANT** toute autre commande pour cloner et patcher le modèle :
    ```bash
    python setup_timesfm.py
    ```

4.  **Synchronisez l'environnement et installez Playwright :**
    Une fois que les sources de TimesFM sont présentes, vous pouvez synchroniser les dépendances et installer les navigateurs requis pour la recherche web :
    ```bash
    uv sync
    uv run playwright install chromium
    ```

5.  **Configurez votre clé API :**
    Créez un fichier `.env` à la racine du projet et ajoutez votre clé API Alpha Vantage :
    ```
    ALPHA_VANTAGE_API_KEY="VOTRE_CLE_API_ICI"
    ```

---

## 🛠️ Utilisation

Le système est conçu pour être simple et puissant. Il entraîne ses modèles sur les données les plus récentes à chaque exécution avant de donner une décision.

### Mode Simulation (Paper Trading)

Pour tester le système sans risque avec un capital fictif de 1000 €, utilisez le flag `--simul`. Le système gérera un historique strict d'achats et de ventes.

```sh
# Lancer une analyse simulée (Défaut: SXRV.FRK - Nasdaq 100 EUR)
uv run main.py --simul

# Lancer une exécution réelle sur Trading 212
uv run main.py --t212
```

### Exécution Réelle (Trading 212)

Le système est désormais **pleinement intégré** avec Trading 212 :
- **Vérification du Portefeuille** : Avant toute action, le robot consulte votre cash réel et vos positions.
- **Calcul Précis des Fractions** : Calcule le nombre d'actions exact pour atteindre votre budget cible.
- **Tickers Certifiés** : Utilisation des identifiants exacts (`SXRVd_EQ` pour le Nasdaq EUR, `CRUDl_EQ` pour le Pétrole WTI).
- **Sécurité de Risque** : Le signal final est systématiquement filtré par le gestionnaire de risques avant l'envoi de l'ordre (Accuracy First).
- **Gestion des API** : Inclut des mécanismes de retry automatique contre les limites de requêtes (Rate Limiting).

Le script va :
1.  **Récupérer les données** de marché en temps réel.
2.  **Entraîner les modèles** d'IA (Ensemble RandomForest, GradientBoosting, etc.) sur l'historique complet.
3.  **Générer une décision hybride** combinant :
    - Modèle quantitatif classique.
    - Analyse de texte via **gemma3:4b** (Ollama).
    - Analyse visuelle des graphiques techniques.
    - Analyse de sentiment des actualités.
    - Prédiction de série temporelle via **TimesFM**.
4.  **Afficher un signal clair** avec le niveau de confiance et la taille de position recommandée.

### 📝 Journal de Bord (Mode Test)
Pendant la phase de test, le système génère un fichier **`trading_journal.csv`** à chaque exécution. 
- **Utilité** : Permet de relire après coup la justification technique de Gemma 4 pour chaque décision (même les `HOLD`).
- **Suppression** : Ce fichier peut être supprimé à tout moment sans risque. Pour désactiver la fonction, il suffit de retirer le bloc "AJOUT : Journalisation CSV" dans `main.py`.

### Sortie attendue

L'analyse produit un tableau récapitulatif directement dans votre terminal :
- **FINAL DECISION**: BUY / SELL / HOLD
- **CONFIDENCE**: Pourcentage de certitude
- **RISK LEVEL**: Évaluation du risque actuel du marché
- **REC. POSITION**: Montant suggéré pour l'investissement

Les graphiques d'analyse sont sauvegardés sous `enhanced_trading_chart.png`.


---

## 🤝 Contribuer

Les contributions sont ce qui fait de la communauté open source un endroit extraordinaire pour apprendre, inspirer et créer. Toutes les contributions que vous faites sont **grandement appréciées**.

Si vous avez une suggestion pour améliorer ce projet, veuillez forker le dépôt et créer une pull request. Vous pouvez aussi simplement ouvrir une issue avec le tag "enhancement".

1.  Forkez le Projet
2.  Créez votre branche de fonctionnalité (`git checkout -b feature/AmazingFeature`)
3.  Commitez vos changements (`git commit -m 'Add some AmazingFeature'`)
4.  Poussez vers la branche (`git push origin feature/AmazingFeature`)
5.  Ouvrez une Pull Request

---

## 📜 Licence

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations (fichier à ajouter si non présent).

---

## 📧 Contact

Lien du projet : [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

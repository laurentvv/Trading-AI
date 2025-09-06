
<div align="center">
  <br />
  <h1>📈 Tri-Modal Hybrid AI Trading System 📈</h1>
  <p>
    Un système de trading IA sophistiqué pour les ETF du NASDAQ, combinant analyse quantitative, textuelle (LLM) et visuelle (V-LLM) pour des décisions de trading robustes.
  </p>
</div>

<div align="center">

[![Statut du Projet](https://img.shields.io/badge/status-en--d%C3%A9veloppement-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Licence](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Table des Matières

- [À Propos du Projet](#-à-propos-du-projet)
  - [Fonctionnalités Clés](#-fonctionnalités-clés)
  - [Stack Technologique](#-stack-technologique)
- [🚀 Getting Started](#-getting-started)
  - [Prérequis](#-prérequis)
  - [Installation](#-installation)
- [🛠️ Utilisation](#️-utilisation)
  - [Analyse Manuelle](#-analyse-manuelle)
  - [Analyse Automatisée](#-analyse-automatisée)
- [🤝 Contribuer](#-contribuer)
- [📜 Licence](#-licence)
- [📧 Contact](#-contact)

---

## 🌟 À Propos du Projet

Ce projet est un système de support à la décision de trading qui utilise une approche IA hybride tri-modale pour générer des signaux de trading pour les ETF du NASDAQ. Il est conçu pour fournir une analyse complète et nuancée en combinant plusieurs perspectives d'IA.

### ✨ Fonctionnalités Clés

- **Moteur IA Hybride Tri-Modal** : Combine trois modèles d'IA pour une décision par consensus :
  1. Un classifieur `scikit-learn` entraîné sur des indicateurs techniques et des données macroéconomiques.
  2. Un LLM pour l'analyse de données numériques brutes.
  3. Un LLM multi-modal pour l'analyse visuelle de graphiques financiers.
- **Backtesting Robuste** : Utilise une validation *walk-forward* pour une évaluation réaliste des performances historiques.
- **Simulation des Coûts de Transaction** : Intègre les coûts de transaction pour des calculs de rendement plus précis.
- **Mise en Cache des Données** : Met en cache les données de marché et macroéconomiques pour accélérer les exécutions.
- **Planificateur Automatisé** : Exécute les analyses quotidiennes et génère des rapports de manière autonome.

### 💻 Stack Technologique

- **Python 3.10+**
- **Calculs & Données** : `pandas`, `numpy`, `yfinance`, `pyarrow`
- **Machine Learning** : `scikit-learn`
- **IA & LLM** : `requests`, `ollama`
- **Visualisation** : `matplotlib`, `seaborn`, `mplfinance`
- **Utilitaires** : `tqdm`, `rich`, `python-dotenv`, `schedule`

---

## 🚀 Getting Started

Suivez ces étapes pour mettre en place votre environnement de développement local.

### ✅ Prérequis

- Python 3.10 ou supérieur
- [Ollama](https://ollama.com/) installé et en cours d'exécution
- Un modèle LLM téléchargé (ex: `ollama pull gemma3:27b`)

### ⚙️ Installation

1.  **Clonez le dépôt :**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Créez un environnement virtuel :**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
    ```
3.  **Installez les dépendances :**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Configurez votre clé API :**
    Créez un fichier `.env` à la racine et ajoutez votre clé API Alpha Vantage :
    ```
    ALPHA_VANTAGE_API_KEY="VOTRE_CLE_API_ICI"
    ```

---

## 🛠️ Utilisation

### Analyse Manuelle

Pour lancer une analyse unique et obtenir une décision immédiate :

```sh
python src/main.py
```

Le script affichera une analyse détaillée dans la console, y compris la décision finale.

### Analyse Automatisée

Pour que le système fonctionne en continu et effectue des analyses quotidiennes :

```sh
python src/scheduler.py
```

Le planificateur s'exécutera en arrière-plan, effectuera les analyses à l'heure configurée et enregistrera tout dans `scheduler.log`.

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

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.

---

## 📧 Contact

Lien du projet : [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)


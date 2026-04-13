# Fiche Projet : Système de Trading IA Hybride

## 1. Objectif Principal
L'objectif principal de ce projet est de développer un système de support à la décision de trading ultra-performant, combinant des modèles quantitatifs classiques, des modèles de fondation spécialisés (`TimesFM`) et des Large Language Models (**Gemma 4**). Il cible les actifs majeurs comme le NASDAQ (QQQ).

## 2. Exigences Clés
Le système remplit les fonctions clés suivantes :
- **Architecture Tri-Modale** : Fusion de signaux provenant de :
    1. Un ensemble de modèles **Scikit-learn** (RandomForest, GradientBoosting, LogisticRegression) validés par `TimeSeriesSplit`.
    2. Une analyse contextuelle et visuelle via **Gemma 4 (e4b)**.
    3. Une prédiction temporelle via **TimesFM (Google Research)**.
- **Unified Interface** : Un point d'entrée unique (`main.py`) pour l'entraînement et l'analyse.
- **Mode Simulation (Paper Trading)** : Un mode de test réaliste avec capital fictif de 1000 € et historique persistant dans une base de données SQLite.
- **Gestion de la Performance** : Monitoring en temps réel avec génération automatique de tableaux de bord visuels.

## 3. Périmètre
- **Dans le Périmètre** :
    - Entraînement automatique à chaque lancement.
    - Analyse technique, fondamentale (macro) et sentiment (news).
    - Mode simulation strict (alternance Achat/Vente).
    - Documentation via la Memory Bank.
- **Hors Périmètre** :
    - Connexion directe aux APIs de courtiers pour l'exécution réelle (envisagé pour Trading 212 plus tard).
    - Interface graphique lourde (focus CLI et rapports images).

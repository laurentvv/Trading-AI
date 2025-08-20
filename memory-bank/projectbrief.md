# Fiche Projet : Système de Trading IA pour les ETF du NASDAQ

## 1. Objectif Principal
L'objectif principal de ce projet est de développer un système sophistiqué de prise de décision de trading piloté par l'IA, axé sur les Fonds Négociés en Bourse (ETF) cotés au NASDAQ, en ciblant spécifiquement les tickers disponibles sur Euronext Paris (par exemple, `FR0011871110.PA`).

## 2. Exigences Clés
Le système doit remplir les fonctions clés suivantes :
- **Ingestion et Mise en Cache des Données** : Récupérer les données historiques de marché (klines) et les mettre en cache localement pour éviter les téléchargements redondants.
- **Moteur de Décision IA Hybride** : Générer des signaux de trading (`ACHAT`/`VENTE`/`NEUTRE`) en utilisant une approche hybride qui combine :
    1. Un modèle quantitatif traditionnel (par exemple, classifieur `scikit-learn`) entraîné sur des indicateurs techniques.
    2. Un Grand Modèle de Langage (LLM) comme Gemma 3 (via Ollama) qui fournit à la fois un signal direct et une analyse qualitative du marché.
- **Backtesting** : Fournir un cadre de backtesting robuste pour évaluer la performance de la stratégie de trading, y compris des métriques telles que le ratio de Sharpe, le drawdown maximal et le taux de gain, tout en tenant compte des coûts de transaction.
- **Modularité et Maintenabilité** : La base de code doit être bien structurée, modulaire et facile à étendre.

## 3. Périmètre
- **Dans le Périmètre** :
    - Développement de la logique de trading et de l'intégration de l'IA.
    - Création d'un cache de données local ("banque de mémoire kline").
    - Implémentation d'un moteur de backtesting.
    - Documentation complète du projet via le système "Banque de Mémoire".
    - Le système est un outil de support à la décision, et non d'exécution automatisée.
- **Hors Périmètre** :
    - Exécution de trading en direct et intégration de courtiers.
    - Interface utilisateur (le système s'exécute en tant que script).
    - Hébergement et déploiement du modèle Ollama (il est supposé qu'il s'exécute localement).

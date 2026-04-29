# Contexte Produit

## 1. Énoncé du Problème
Les traders individuels et semi-professionnels s'appuient souvent sur un mélange d'analyse technique et de jugement qualitatif pour prendre des décisions de trading. Ce processus peut être chronophage, sujet aux biais émotionnels et difficile à tester de manière systématique. Les outils existants peuvent être soit trop simplistes, incapables de capturer la complexité du marché, soit trop complexes, nécessitant une courbe d'apprentissage abrupte.

## 2. Vision
Ce projet vise à créer un système de support à la décision de trading puissant mais accessible. Il comble le fossé entre l'analyse quantitative et le raisonnement qualitatif, semblable à celui d'un humain, en exploitant un **ensemble multi-modèles d'IA hybride**. Le système permet à l'utilisateur de prendre des décisions de trading plus informées, basées sur les données et testées de manière systématique, sur les **ETF du NASDAQ et du Pétrole (WTI)** via **Trading 212**.

## 3. Fonctionnement Attendu
L'utilisateur exécutera un seul script depuis la ligne de commande. Le script :
1. Récupérera les dernières données de marché pour un ticker spécifié (NASDAQ ou WTI), en utilisant un **cache Parquet local avec invalidation automatique** (stale si > 2 jours).
2. Traitera les données pour calculer un large éventail d'indicateurs techniques.
3. Injectera ces informations dans **neuf modèles d'IA parallèles** :
    - **Modèle Classique** (Scikit-Learn : RandomForest, GradientBoosting, LogisticRegression).
    - **TimesFM 2.5** (Google Research) : prévision probabiliste de séries temporelles.
    - **TensorTrade / PPO** : agent de Reinforcement Learning via stable-baselines3 (gymnasium).
    - **LLM Texte** (Gemma 4:e4b via Ollama) : analyse contextuelle enrichie par recherche web (Crawl4AI).
    - **LLM Visuel** (Gemma 4:e4b) : analyse directe du graphique technique.
    - **Sentiment Analysis** : hybride Alpha Vantage + AlphaEar (tendances finance multi-sources).
    - **Hyperliquid** : sentiment décentralisé (Funding Rate, Open Interest sur perps WTI).
    - **Vincent Ganne Model** : validation géopolitique et cross-asset (exclusif Nasdaq, signal BUY uniquement).
    - **Oil-Bench Model** : expert fondamental EIA pour le pétrole (stocks, importations, raffineries, STEO).
4. Combinera les sorties via un **moteur de décision pondéré** (75% cognitif / 25% quantitatif) avec ajustement adaptatif des poids selon la précision récente.
5. Affichera la décision, l'analyse du LLM et les principales métriques de performance d'un backtest dans la console.
6. Générera des graphiques visualisant la performance de la stratégie.
7. Optionnellement, exécutera l'ordre sur **Trading 212** (mode démo ou réel) avec vérification du cash, prix live T212, et sizing fractionné.

## 4. Objectifs d'Expérience Utilisateur
- **Simplicité** : Le système doit être facile à exécuter avec une seule commande (`uv run main.py`).
- **Clarté** : La sortie doit fournir à la fois un signal direct et le raisonnement derrière celui-ci (analyse LLM + journal CSV détaillé par modèle).
- **Transparence** : Les résultats du backtesting et les métriques de performance doivent être transparents, permettant à l'utilisateur de comprendre la performance historique et les risques de la stratégie.
- **Résilience** : Le système doit tolérer les pannes réseau (circuit breakers yfinance, fallback MA50, cache auto-invalidation) et continuer à fonctionner même si certains modèles individuels échouent (Graceful Degradation).
- **Automatisation** : Le script `schedule.py` permet une exécution autonome continue (8h30-18h00, intervalle 30 min) avec dashboard live.

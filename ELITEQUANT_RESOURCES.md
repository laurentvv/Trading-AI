
# Ressources EliteQuant pertinentes pour Trading-AI

Suite à l'analyse du répertoire [EliteQuant](https://github.com/EliteQuant/EliteQuant), voici une sélection de bibliothèques, modèles, idées et sources de données qui pourraient enrichir l'architecture hybride de **Trading-AI**. Seules les solutions en Python (ou architectures inspirantes) et les ressources gratuites/open-source ont été retenues, conformément aux directives.

## 1. Modèles de Machine Learning et Deep Learning (Série Temporelle et RL)

Le système actuel utilise *TimesFM* et *Scikit-learn*. Les ressources suivantes de la section "Quantitative Model" pourraient être intégrées pour étendre nos moteurs de décision :

* **[Tensortrade](https://github.com/tensortrade-org/tensortrade)** :
  * *Pourquoi c'est intéressant* : C'est une bibliothèque Python open-source de Reinforcement Learning (RL) spécialement conçue pour le trading. Elle permet de construire des environnements de trading compatibles avec OpenAI Gym.
  * *Idée d'intégration* : Ajouter un agent RL comme signal supplémentaire dans le consensus (à côté des modèles prédictifs et d'analyse de sentiment actuels).

* **[gym-trading](https://github.com/hackthemarket/gym-trading) / [QLearning_Trading](https://github.com/ucaiado/QLearning_Trading)** :
  * *Pourquoi c'est intéressant* : Implémentations d'environnements d'apprentissage par renforcement pour le trading algorithmique.
  * *Idée d'intégration* : Inspirer le développement d'un nouveau moteur de backtesting ou d'entraînement dans `src/` qui utiliserait le Q-learning sur les données historiques WTI ou NASDAQ.

* **[bulbea](https://github.com/achillesrasquinha/bulbea)** :
  * *Pourquoi c'est intéressant* : Bibliothèque Python basée sur le Deep Learning pour la modélisation et la prédiction du marché boursier.
  * *Idée d'intégration* : Fournir des indicateurs techniques profonds en complément de ceux générés par `TechnicalAnalyzer`.

* **[DeepDow](https://github.com/jankrepl/deepdow)** :
  * *Pourquoi c'est intéressant* : Optimisation de portefeuille via Deep Learning.
  * *Idée d'intégration* : Plutôt que de simplement prendre des décisions "Acheter/Vendre", ce modèle pourrait aider à optimiser l'allocation du capital initial ou en cours pour la partie "Portfolio Manager" (dans `src/t212_executor.py` ou futur équivalent global).

## 2. Bibliothèques Quantitatives et Analyse de Séries Temporelles

Le projet utilise des données OHLCV classiques. Ces outils peuvent améliorer l'extraction de features.

* **[pyflux](https://github.com/RJT1990/pyflux)** :
  * *Pourquoi c'est intéressant* : Bibliothèque d'analyse de séries temporelles en Python.
  * *Idée d'intégration* : Améliorer les signaux d'entrée pour TimesFM ou Scikit-Learn avec une modélisation bayésienne des volatilités de marché.

* **[arch](https://github.com/bashtage/arch)** :
  * *Pourquoi c'est intéressant* : Modélisation ARCH/GARCH en Python, excellent pour prédire la volatilité (un facteur crucial pour l'Oil/WTI et le NASDAQ).
  * *Idée d'intégration* : Créer un nouveau module d'évaluation du risque (`risk_analyzer.py`) qui utiliserait la volatilité GARCH pour adapter la confiance globale (Global Confidence) du consensus.

* **[Statsmodels](https://www.statsmodels.org)** :
  * *Pourquoi c'est intéressant* : L'outil standard en Python pour l'estimation de modèles statistiques.
  * *Idée d'intégration* : Ajouter des tests de stationnarité (ADF) ou de co-intégration avant d'accepter les signaux de `TimesFM`.

## 3. Sources de Données (Gratuites)

Actuellement, l'outil utilise yfinance, EIA et Alpha Vantage.

* **[IEX Trading (via IEX Cloud Free Tier)](https://iextrading.com/trading/market-data/)** :
  * *Pourquoi c'est intéressant* : Données de marché fiables. Même avec des quotas stricts sur le niveau gratuit, c'est une très bonne source alternative de fallback en cas de panne de Yahoo Finance (`yfinance`).
  * *Idée d'intégration* : Créer un module `data_fetcher_iex.py` comme fallback pour sécuriser l'acquisition des données pour le NASDAQ.

* **[SEC EDGAR API](https://sec-api.io/)** :
  * *Pourquoi c'est intéressant* : Pour accéder aux dépôts (filings) d'entreprises.
  * *Idée d'intégration* : Si l'architecture décide d'étendre l'analyse (NASDAQ) au-delà des prix purs vers l'analyse fondamentale, combiner cette API avec les LLMs actuels (Qwen/Ollama) pour extraire le sentiment institutionnel sur le secteur technologique (NASDAQ).

## 4. Concepts et Architectures (Inspirés d'autres langages/systèmes)

* **[Arctic (par Man AHL)](https://github.com/manahl/arctic)** :
  * *Pourquoi c'est intéressant* : Datastore de haute performance pour séries temporelles et données tick-by-tick (historiquement sur MongoDB, aujourd'hui ArcticDB en Python/C++).
  * *Idée d'intégration* : Le projet Trading-AI utilise actuellement SQLite (`model_performance.db`, `trading_history.db`). Si le système évolue pour ingérer des données intra-day massives (tick data) plutôt que Daily, remplacer SQLite par ArcticDB permettrait un gain de performance massif pour pandas.

* **[QuantConnect (Lean) / Zipline / Backtrader]** *(Plateformes)* :
  * *Idée d'intégration* : `Trading-AI` dispose de son propre moteur de simulation (ex: `backtest_prod.py`). S'inspirer de la boucle événementielle (Event-Driven Architecture) de Zipline ou Lean (C#) permettrait de réduire le *Look-ahead bias* de notre backtesting.

---

### Conclusion et Plan d'Action Recommandé

Pour enrichir `Trading-AI` tout en gardant sa contrainte de gratuité et d'environnement Python local :
1. **Priorité 1 :** Étudier l'intégration de `arch` ou `pyflux` pour modéliser le risque et la volatilité, augmentant ainsi la robustesse du score de "Global Confidence".
2. **Priorité 2 :** Expérimenter `tensortrade` pour ajouter une couche "Apprentissage par Renforcement" comme quatrième pilier de décision (en plus de LLM, Scikit, TimesFM).
3. **Priorité 3 :** Envisager l'implémentation de la logique de datastore inspirée de `Arctic` si les données historiques s'accumulent au-delà des simples cours journaliers.

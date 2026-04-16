# Plan d'Intégration des Données EIA et du Modèle Oil-Bench

Ce document décrit comment intégrer les données fondamentales de l'EIA (Energy Information Administration) et implémenter un nouveau modèle de décision inspiré de "oil-bench" pour le trading du WTI dans le système `trading-ai`.

## 1. Obtention de la Clé API EIA (Gratuite)

Pour accéder aux données fondamentales du marché de l'énergie (comme les stocks de pétrole brut américain), il faut utiliser l'API de l'EIA.

1. Allez sur le site officiel de l'EIA : [https://www.eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php)
2. Remplissez le formulaire d'inscription (Email, Prénom, Nom, Nom de l'entreprise/Projet).
3. Acceptez les termes et conditions.
4. Vous recevrez immédiatement votre clé API par email.
5. Ajoutez cette clé dans votre fichier `.env` :
   ```env
   EIA_API_KEY=votre_cle_api_ici
   ```

## 2. Nouveaux Modules à Développer

L'intégration nécessitera la création de deux nouveaux fichiers dans le dossier `src/` :

### A. `src/eia_client.py`
Ce module sera responsable de la communication avec l'API de l'EIA.
- **Objectif** : Récupérer les données clés (ex. inventaires hebdomadaires de pétrole brut).
- **Fonctions prévues** :
  - `get_us_crude_inventories()`: Récupère la dernière donnée des stocks de brut.
  - Formater les données pour qu'elles soient facilement lisibles par le LLM.

### B. `src/oil_bench_model.py`
Ce module contiendra la logique décisionnelle, en agissant comme un Analyste Quantitatif en Matières Premières.
- **Condition d'exécution** : Ce modèle ne s'exécutera **que** si le ticker analysé est lié au pétrole (ex: `CL=F` pour le WTI). Il sera ignoré pour le NASDAQ (`QQQ`), etc.
- **Collecte de données** :
  - *Prix* : Ticker WTI et le DXY (`DX-Y.NYB`) via `yfinance`.
  - *Fondamentaux* : Appel à `eia_client.py`.
  - *News* : Utilisation des modules existants basés sur DuckDuckGo (`web_researcher.py` / `news_fetcher.py`).
- **Logique** :
  - Construction d'un prompt complet ("Vous êtes un analyste expert...").
  - Envoi à Ollama local via `llm_client.py`.
  - Le LLM retournera une allocation de portefeuille de 0 à 100%.
- **Traduction en Signal** :
  - > 50% = Signal ACHAT (BUY)
  - < 50% = Signal VENTE (SELL)
  - = 50% = NEUTRE (HOLD)

## 3. Intégration au Système Existant

Le fichier principal d'analyse (`src/enhanced_trading_example.py` ou `src/main.py`) sera modifié :

1.  **Vérification du Ticker** : Avant de lancer l'analyse, vérifier le nom du ticker.
2.  **Exécution conditionnelle** :
    ```python
    if "CL=" in ticker or "WTI" in ticker:
        oil_model = OilBenchModel()
        oil_decision = oil_model.analyze()
        consensus_results['Oil_Bench_LLM'] = oil_decision
    ```
3.  **Vote du Consensus** : Le signal généré par le `OilBenchModel` sera intégré aux résultats. Le `EnhancedDecisionEngine` prendra ensuite ce nouveau vote en compte avec les autres (Scikit-learn, TimesFM, Sentiment) pour produire la décision finale.

## 4. Prochaines Étapes
1. Créer le compte EIA et ajouter `EIA_API_KEY` au `.env`.
2. Développer et tester unitairement `eia_client.py`.
3. Développer `oil_bench_model.py` et affiner le prompt LLM.
4. Connecter le tout au flux de `main.py`.

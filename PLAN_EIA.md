# Plan d'Intégration des Données EIA et du Modèle Oil-Bench (TERMINÉ)

Ce document décrit comment intégrer les données fondamentales de l'EIA (Energy Information Administration) et implémenter un nouveau modèle de décision inspiré de "oil-bench" pour le trading du WTI dans le système `trading-ai`.

## Status : ✅ COMPLÉTÉ

## 1. Obtention de la Clé API EIA (Gratuite) ✅

### La clé à été ajoutée `.env` :
   ```env
   EIA_API_KEY=cle_ok
   ```

## 2. Modules Développés ✅

### A. `src/eia_client.py` ✅
Module responsable de la communication avec l'API v2 de l'EIA.
- **Données exploitées** :
  - `get_crude_inventories()`: Stocks hebdomadaires (WSTK) avec tendance 4 semaines.
  - `get_crude_imports()`: Importations mensuelles US par origine.
  - `get_refinery_utilization()`: Taux d'utilisation des raffineries (moyenne nationale).
  - `get_steo_series()`: Prévisions Short-Term Energy Outlook (Prix, Production, Demande).

### B. `src/oil_bench_model.py` ✅
Logique décisionnelle imitant un Analyste Quantitatif en Matières Premières.
- **Signal** : Basé sur une allocation (0-100%) générée par Gemma 4.
- **Niveaux** : STRONG_BUY (>=75%), BUY (>=55%), SELL (<=45%), STRONG_SELL (<=25%).

## 3. Intégration au Système ✅

Le flux `main.py` intègre désormais automatiquement le modèle `OilBench` dès qu'un ticker lié au pétrole est détecté.

## 4. Historique des Étapes ✅
1. Création du compte EIA et configuration `.env`. (FAIT)
2. Développement et tests unitaires de `eia_client.py`. (FAIT)
3. Développement de `oil_bench_model.py` et affinement du prompt. (FAIT)
4. Connexion au flux `main.py` et validation via simulation. (FAIT)
5. Ajout des métriques avancées (Raffineries, Imports). (FAIT)

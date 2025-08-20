# Contexte Actif

## 1. Focus Actuel du Travail
Le projet est en **Phase 3 : Finalisation et Documentation**.
Le focus immédiat est sur **la Mise à jour de toute la documentation du projet pour refléter les dernières améliorations, y compris l'intégration des données macroéconomiques et le moteur hybride à 3 modèles**.

## 2. Changements Récents
- **Moteur Hybride à 3 Modèles Intégré** : `src/main.py` a été mis à jour avec succès pour orchestrer le modèle classique, le LLM textuel et le nouveau LLM visuel, combinant leurs sorties en une décision finale.
- **Client LLM Mis à Jour** : Le module `src/llm_client.py` supporte les modèles textuels et visuels.
- **Générateur de Graphiques Créé** : Le module `src/chart_generator.py` est terminé.
- **Correction de Bug (2025-08-18)** : Correction d'un crash causé par une `ALPHA_VANTAGE_API_KEY` manquante. Le système charge désormais la clé depuis un fichier `.env`.
- **Intégration des Données Macroeconomiques (2025-08-19)** : Implémentation réussie de la récupération robuste des données depuis FRED via `pandas-datareader` avec mise en cache locale. Ces données sont désormais intégrées comme caractéristiques dans le modèle quantitatif classique.
- **Données Historiques Étendues (2025-08-20)** : Le système récupère désormais l'historique complet des données disponibles pour QQQ, augmentant considérablement le jeu de données pour l'entraînement et le backtesting.

## 3. Prochaines Étapes
1.  Finaliser les mises à jour de la documentation dans `README.md`, `QWEN.md`, et tous les fichiers pertinents de `memory-bank`.
2.  Effectuer un test de bout en bout final du système complet pour s'assurer que tous les composants fonctionnent de manière fluide.
3.  Soumettre le projet final.

## 4. Décisions et Considérations Actives
- Les graphiques seront générés en interne en utilisant `mplfinance` pour la fiabilité.
- Le graphique affichera 6 mois de données quotidiennes avec des chandeliers, des MM 50/200, le Volume, le RSI et le MACD.
- Le nouveau signal visuel constitue un troisième vote, égal aux autres, dans le moteur hybride.
- Les données macroéconomiques (taux d'intérêt, IPC, chômage, PIB) font désormais partie intégrante du jeu de caractéristiques du modèle quantitatif classique.
- Le système exploite un vaste jeu de données historiques (de 1999 à aujourd'hui) pour un entraînement et un backtesting robustes.

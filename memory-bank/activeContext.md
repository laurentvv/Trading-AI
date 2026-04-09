# Contexte Actif

## 1. Focus Actuel du Travail
Le projet est en **Phase 2 : Apprentissage Initial**.
Le focus immédiat est sur **la surveillance de la stabilité du système et l'acquisition de données**.

## 2. Changements Récents
- **Moteur Hybride à 3 Modèles Intégré** : `src/main.py` a été mis à jour avec succès pour orchestrer le modèle classique, le LLM textuel et le nouveau LLM visuel, combinant leurs sorties en une décision finale.
- **Client LLM Mis à Jour** : Le module `src/llm_client.py` supporte les modèles textuels et visuels.
- **Générateur de Graphiques Créé** : Le module `src/chart_generator.py` est terminé.
- **Correction de Bug (2025-08-18)** : Correction d'un crash causé par une `ALPHA_VANTAGE_API_KEY` manquante. Le système charge désormais la clé depuis un fichier `.env`.
- **Intégration des Données Macroeconomiques (2025-08-19)** : Implémentation réussie de la récupération robuste des données depuis FRED via `pandas-datareader` avec mise en cache locale. Ces données sont désormais intégrées comme caractéristiques dans le modèle quantitatif classique.
- **Données Historiques Étendues (2025-08-20)** : Le système récupère désormais l'historique complet des données disponibles pour QQQ, augmentant considérablement le jeu de données pour l'entraînement et le backtesting.
- **Corrections du Planificateur Intelligent (2025-09-12)** : Correction de la logique de transition de phase et de la persistance de la date de démarrage du projet.
- **Gestion de la Configuration du Scheduler (2025-09-12)** : Ajout d'un fichier de configuration `scheduler_config.json` pour permettre un contrôle précis du comportement du planificateur, notamment les transitions de phase. La documentation (`GEMINI.md`, `README.md`, `techContext.md`) a été mise à jour pour refléter ce changement.
- **Correction du Bug de Persistance de Phase (2025-09-12)** : Corrigé un bug où la transition de phase du scheduler n'était pas immédiatement sauvegardée dans la base de données. La méthode `_transition_to_next_phase` appelle maintenant une nouvelle fonction `_persist_phase_transition` pour assurer la persistance immédiate.

## 3. Prochaines Étapes
1.  Continuer la surveillance en Phase 2 jusqu'à son achèvement (22 septembre 2025).
2.  Transition automatique vers Phase 3 : Optimisation.
3.  Surveillance des performances et ajustements si nécessaire.

## 4. Décisions et Considérations Actives
- Les graphiques seront générés en interne en utilisant `mplfinance` pour la fiabilité.
- Le graphique affichera 6 mois de données quotidiennes avec des chandeliers, des MM 50/200, le Volume, le RSI et le MACD.
- Le nouveau signal visuel constitue un troisième vote, égal aux autres, dans le moteur hybride.
- Les données macroéconomiques (taux d'intérêt, IPC, chômage, PIB) font désormais partie intégrante du jeu de caractéristiques du modèle quantitatif classique.
- Le système exploite un vaste jeu de données historiques (de 1999 à aujourd'hui) pour un entraînement et un backtesting robustes.
- Le système est en production depuis le 25 août 2025 et fonctionne correctement avec les corrections récentes.

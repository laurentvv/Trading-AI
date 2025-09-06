# Propositions d'améliorations pour le système de décision sur le Nasdaq QQQ

Ce fichier suit les idées et leur statut pour améliorer le système de trading AI.

**Mise à jour :** La fiabilisation du système via un nouveau planificateur a été réalisée. Le système est maintenant stable et collecte des données de manière autonome.

**Prochaine étape :**
- Utiliser SHAP pour expliquer les décisions du modèle classique.

## Liste des propositions

1.  **Fiabilisation du Planificateur (Scheduler)**
    *   **Description :** Remplacer le planificateur défaillant pour assurer l'exécution automatique et fiable des analyses.
    *   **Statut :** **FAIT** - Un nouveau planificateur robuste (`src/scheduler.py`) a été créé, corrigeant les erreurs critiques qui empêchaient le fonctionnement du système.
    *   **Priorité :** Critique

2.  **Incorporation de Données Macroéconomiques**
    *   **Description :** Intégrer des données macroéconomiques (taux d'intérêt, ISM PMI, etc.).
    *   **Statut :** **FAIT** - Implémentation de base réalisée. Un système de cache a été ajouté. La logique de récupération des données macro est en place en utilisant `pandas-datareader` pour accéder aux données FRED.
    *   **Priorité :** Haute

3.  **Explication des Décisions (XAI)**
    *   **Description :** Utiliser SHAP pour expliquer les décisions du modèle classique.
    *   **Statut :** En attente
    *   **Priorité :** Haute

4.  **Optimisation Automatique des Poids**
    *   **Description :** Trouver les pondérations optimales pour les modèles via un algo d'optimisation.
    *   **Statut :** En attente
    *   **Priorité :** Moyenne

5.  **Modèle de Risque Dynamique**
    *   **Description :** Ajuster dynamiquement les seuils/pondérations en fonction de la volatilité ou du VIX.
    *   **Statut :** En attente
    *   **Priorité :** Moyenne

6.  **Interface Utilisateur Web Simple**
    *   **Description :** Créer une interface web avec Streamlit/Gradio pour visualiser les décisions.
    *   **Statut :** En attente
    *   **Priorité :** Faible

7.  **Alertes et Notifications**
    *   **Description :** Envoyer des alertes pour les signaux "FORTS".
    *   **Statut :** **COMMENCÉ** - Le nouveau planificateur inclut une fonction `_check_performance_alerts` qui peut être étendue pour cette fonctionnalité.
    *   **Priorité :** Faible

8.  **Analyse Multi-Horizons**
    *   **Description :** Générer des signaux pour 3j, 1 semaine, 1 mois.
    *   **Statut :** En attente
    *   **Priorité :** Faible

9.  **Analyse de Sentiment Basée sur les Réseaux Sociaux**
    *   **Description :** Analyser le sentiment à partir de Twitter/X ou Reddit pour un sentiment du marché plus réactif.
    *   **Statut :** Inutile
    *   **Priorité :** Moyenne

10. **Comparaison avec un Benchmark Sectoriel**
    *   **Description :** Comparer les performances avec VGT ou SPY.
    *   **Statut :** Inutile
    *   **Priorité :** Faible

11. **Support pour d'Autres Actifs**
    *   **Description :** Rendre le ticker cible configurable.
    *   **Statut :** Inutile
    *   **Priorité :** Faible

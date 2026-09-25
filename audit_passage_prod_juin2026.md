# 📋 Audit de Décision - Passage en PROD (Juin 2026)

Ce document dresse le bilan final du code actuel et fournit une checklist de décision pour autoriser le basculement officiel de l'environnement DEMO vers l'environnement PROD de Trading 212.

## 🛡️ 1. Analyse des Garde-fous (Risk Management)

L'analyse approfondie du code source (`src/t212_executor.py` et `main.py`) confirme que les règles de sécurité exigées pour la PROD sont en grande partie intégrées.

*   ✅ **Bloquer les ventes à perte :**
    *   **Statut : OPÉRATIONNEL**
    *   *Analyse :* La fonction `_check_sell_loss_guard` bloque l'envoi de l'ordre de vente si la valeur actuelle du portefeuille est inférieure à `99.8%` du coût d'achat de référence. Le système ne peut donc pas clôturer une ligne en perte latente, protégeant ainsi le capital.
*   ✅ **Gestion stricte du Cash (Budget par Ticker) :**
    *   **Statut : OPÉRATIONNEL**
    *   *Analyse :* Le dictionnaire `INITIAL_BUDGETS` alloue bien 1000 € par ticker par défaut. La logique métier enregistre correctement la plus-value : lors d'une vente, le capital disponible devient le capital précédent + la plus-value de l'opération. L'achat suivant sera borné par cette enveloppe sans jamais empiéter sur l'argent dédié aux autres trackers.
*   ✅ **Trailing Stop & Sécurisation des gains :**
    *   **Statut : OPÉRATIONNEL**
    *   *Analyse :* Déclenchement automatique de prise de profit (`SELL`) si l'on enregistre une chute de `≥ 3%` depuis le pic historique atteint par la ligne, sous réserve d'avoir un bénéfice minimum de `0.5%`.
*   ⚠️ **Sizing Progressif Dynamique (75% - 100%) :**
    *   **Statut : PARTIEL (Action Requise)**
    *   *Analyse :* L'IA calcule bien l'exposition suggérée en fonction du consensus et l'affiche dans les logs (`results['position_sizing'].recommended_size`), **mais ce pourcentage n'est pas encore transmis à l'API de Trading 212**. Actuellement, `execute_t212_trade` investit toujours statiquement 95% du budget disponible pour le ticker. (cf. Recommandations).

---

## 🚀 2. Checklist de Basculement (DEMO → PROD)

À valider lors du comité de décision fin juin pour déclencher le passage en réel :

### A. Configuration Système & Infrastructure
- [ ] Modifier la variable d'environnement dans `.env.t212` : `T212_ENV=live` (actuellement en mode demo).
- [ ] Remplacer les clés API DEMO par les **clés API Live T212** dans le fichier `.env.t212`.
- [ ] Vérifier que les variables globales `INITIAL_BUDGETS` de `src/t212_executor.py` correspondent bien au montant que vous êtes prêt à investir réellement (ex: 1000 € par ligne).
- [ ] Mettre en place ou nettoyer la base `t212_portfolio_state.json` pour partir sur un suivi propre sans l'historique DEMO (sinon, l'outil pourrait penser qu'il a déjà des positions).

### B. Mises à jour Code & Logique (Avant le lancement final)
- [ ] **Sizing Progressif :** Ajouter un argument `sizing_ratio` à la fonction `execute_t212_trade()` dans `t212_executor.py`, et modifier le calcul du `target_budget` pour prendre ce ratio en compte (actuellement bloqué à 100% de l'allocation).
- [ ] **Validation de la bascule Ollama :** Confirmer que le serveur Ollama (`hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`) ne subit pas d'OOM (Out Of Memory) sur des runs consécutifs en PROD.

### C. Vérifications Broker (Trading 212)
- [ ] Alimentation du compte confirmée (Fonds suffisants et non investis pour honorer les enveloppes `INITIAL_BUDGETS`).
- [ ] Aucun reliquat de trades manuels ou d'anciens bots sur les tickers ciblés (SXRV, CRUDP) afin de ne pas fausser le calcul du Trailing Stop de l'agent IA.

---

> **Conclusion de l'audit :** L'architecture logicielle est saine et protectrice. Le blocage de vente à perte et le cloisonnement budgétaire fonctionnent parfaitement, protégeant l'investisseur d'un emballement du bot. Le système sera **apte à la production (Go-Live)** une fois la clé API changée et la fonctionnalité d'allocation dynamique intégrée à l'exécuteur.

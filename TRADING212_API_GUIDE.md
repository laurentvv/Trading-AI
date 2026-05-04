# Guide d'Utilisation de l'API Trading 212 (v0)

Ce document récapitule les fonctionnalités validées et la manière de les utiliser avec Python pour l'automatisation de vos transactions.

## 1. Configuration de l'Environnement

Le projet utilise un fichier `.env.t212` pour stocker les identifiants de manière sécurisée.

### Variables requises (`.env.t212`) :
- `T212_API_KEY` : Votre **API Key ID** généré dans l'application.
- `T212_API_SECRET` : Votre **API Secret** généré dans l'application.
- `T212_ENV` : `"demo"` (Argent virtuel) ou `"live"` (Argent réel).

---

## 2. Découvertes Techniques Majeures

### Achat par Valeur (€) vs Quantité
⚠️ **Important :** L'API Trading 212 ne supporte **PAS** l'achat direct par montant (ex: "Acheter pour 1000€"). Elle n'accepte que le paramètre `quantity` (Nombre d'actions).
- **Solution implémentée :** Le système calcule dynamiquement la fraction d'action nécessaire : `Quantité = Budget / Prix Réel`.
- **Précision :** Pour l'ETF Nasdaq (`SXRVd_EQ`), la précision est limitée à **4 décimales**.

### Actions Fractionnées
Le système gère nativement les fractions. Lors d'une vente (`SELL`), le script interroge l'API pour récupérer la quantité exacte possédée (ex: `1.8176`) et passe un ordre de vente pour la **totalité** afin de liquider proprement la position.

### Sécurité et Robustesse
- **Vérification du Portefeuille** : Le système interroge systématiquement votre cash disponible et vos positions ouvertes **avant** d'envoyer un ordre d'achat ou de vente.
- **Gestion des Erreurs API** : Un mécanisme de **Retry automatique** est implémenté pour gérer les erreurs `TooManyRequests` (Code 429), garantissant que les ordres passent même en cas de congestion de l'API.
- **Résilience des champs API** : L'API T212 peut omettre certains champs (ex: `averagePrice`) dans les réponses positions. Le système utilise un fallback défensif (`currentValue / quantity`) pour calculer le prix d'entrée lors de la synchronisation du portefeuille local.

---

## 3. Workflow d'Exécution IA

Le système suit un budget dédié défini dans `t212_portfolio_state.json`.

### Hiérarchie de Récupération du Prix
Le système utilise une cascade de sources pour obtenir le prix le plus précis possible :

1. **Trading 212 Live** (`get_t212_price()`) : Interroge `GET /equity/positions` pour trouver le `currentPrice` de l'ETF en EUR. Instantané (~0.2s). Uniquement disponible si une position est ouverte sur le ticker.
2. **MarketDataManager** (yfinance) : Télécharge les 5 derniers jours via `yf.download()` avec timeout 10s.
3. **yfinance History** : `yf.Ticker().history(period="5d")` en dernier recours.
4. **Erreur** : Aucune source disponible → l'exécution est annulée.

### Flux de Décision
1. **Signal BUY :**
   - Tente le prix T212 en priorité.
   - Calcule la quantité pour un budget de **1000€** (avec une marge de sécurité de 1% pour éviter les rejets).
   - Envoie l'ordre `Market` à Trading 212.
2. **Signal SELL :**
   - Identifie toutes les fractions d'actions possédées.
   - Liquide la position totale au prix du marché.
   - Met à jour le capital (Capital Initial + Profit/Perte).

### Résilience Réseau
- **Circuit breaker yfinance** : Les timeouts metadata (`info`) et données (`download`) sont gérés par des trackers séparés. Après 3 échecs consécutifs, les appels sont bloqués pendant 120s.
- **Timeout 10s** sur tous les appels yfinance (avant : 30s+ sans limite).
- **News API** (Alpha Vantage) : timeout 10s sur les requêtes HTTP.

---

## 4. Commandes Utiles

| Action | Commande |
| :--- | :--- |
| **Analyse + Exécution T212** | `uv run main.py --ticker QQQ --t212` |
| **Test de connexion** | `python test_t212.py` |
| **Audit des accès** | `python full_api_audit.py` |
| **Recherche d'ETFs en €** | `python search_nasdaq_eur.py` |

---

## 5. Limites et Sécurité

1. **Rate Limiting :** L'API est sensible. Le système inclut des pauses pour éviter l'erreur `TooManyRequests`.
2. **Marché Fermé :** Les ordres passés hors session (avant 15h30 pour le Nasdaq) restent en "Pending" sur Trading 212.
3. **Fichier de suivi :** `t212_portfolio_state.json` est le "journal de bord" de l'IA. Ne pas le supprimer manuellement si une position est active.

---
*Dernière mise à jour : 4 mai 2026.*

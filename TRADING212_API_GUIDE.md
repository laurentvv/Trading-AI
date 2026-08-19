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

### Pas d'API OHLCV / Candles
L'API Trading 212 v0 ne propose **aucun endpoint de données historiques** (candles, OHLCV). 12 endpoints potentiels ont été testés (`/instruments/{id}/candles`, `/market-data/`, etc.) — tous retournent 404. Le système utilise donc **Yahoo Finance pour l'historique** et **T212 uniquement pour le prix live** (via `/equity/positions`).

### Achat par Valeur (€) vs Quantité
⚠️ **Important :** L'API Trading 212 ne supporte **PAS** l'achat direct par montant (ex: "Acheter pour 1000€"). Elle n'accepte que le paramètre `quantity` (Nombre d'actions).
- **Solution implémentée :** Le système calcule dynamiquement la fraction d'action nécessaire : `Quantité = Budget / Prix Réel`.
- **Précision :** Pour l'ETF Nasdaq (`SXRVd_EQ`), la précision est limitée à **4 décimales**.

### Actions Fractionnées
Le système gère nativement les fractions. Lors d'une vente (`SELL`), le script interroge l'API pour récupérer la quantité exacte possédée (ex: `1.8176`) et passe un ordre de vente pour la **totalité** afin de liquider proprement la position.

### Sécurité et Robustesse
- **Vérification du Portefeuille** : Le système interroge systématiquement votre cash disponible et positions ouvertes **avant** d'envoyer un ordre d'achat ou de vente.
- **Gestion des Erreurs API** : Un mécanisme de **Retry automatique** est implémenté pour gérer les erreurs `TooManyRequests` (Code 429), garantissant que les ordres passent même en cas de congestion de l'API.
- **Résilience des champs API** : L'API T212 peut omettre certains champs (ex: `averagePrice`) dans les réponses positions. Le système utilise un fallback défensif (`currentValue / quantity`) pour calculer le prix d'entrée lors de la synchronisation du portefeuille local.

### Sécurité des Ordres — idempotence par réconciliation (GO-gate 1, 2026-08-19)

La documentation officielle précise que `POST /equity/orders/market` **n'est pas idempotent** : un retry après perte de réponse peut créer un ordre dupliqué. Le système applique donc :

1. **Timeout explicite** (15 s) sur chaque POST d'ordre (`post_order_market`).
2. **Retry sûr uniquement** : un `429`/`TooManyRequests` (ordre rejeté = non exécuté) est retenté.
3. **Réconciliation avant tout retry réseau** : après un timeout/erreur réseau, `GET /equity/positions` est consulté — si la position est apparue (BUY) ou disparu (SELL), l'ordre est considéré exécuté et **aucun re-POST** n'est émis (anti double-achat).
4. **Confirmation de fill** : un code 2xx signifie « accepté », pas « exécuté » — l'état local et la DB ne sont écrits qu'après observation du fill (`averagePricePaid`, prix réel — plus le prix signal Yahoo).

### Protections côté Broker — stop mouvant + take-profit (GO-gate 2, 2026-08-19)

- Chaque ordre market BUY part avec un **`takeProfit` attaché** (+8 %, prix absolu, 2 décimales). Si l'API refuse ce champ non documenté, le fallback « ordre nu + stop dédié » s'active automatiquement.
- Après fill confirmé, un **ordre stop dédié** est placé : `POST /equity/orders/stop` avec `{ticker, quantity: -qty, stopPrice, timeValidity: "GOOD_TILL_CANCEL"}` à −10 % du prix de fill réel.
- **Ratchet (stop mouvant)** : à chaque cycle (~30 min), si le plus-haut de la position (`highest_value`) progresse, le stop est **annulé** (`DELETE /equity/orders/{id}`) puis **replacé plus haut** à `peak × 0.90` — strictement croissant, jamais abaissé. Si le replacement échoue après suppression, un stop d'urgence est replacé à l'ancien niveau (jamais de position volontairement sans stop).
- La conséquence : une position réelle reste protégée chez le broker même si la machine/scheduler meurt — les stops logiciels (hard stop −10 %, trailing −3 %, time-stop 15 j) restent actifs en parallèle comme défense interne.
- *Validation live à effectuer sur démo : `uv run python tests/check_t212_stops.py` (consigner le bilan ici).*

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

### Synchronisation Réelle du Portefeuille
Le système synchronise désormais son état directement depuis T212 :
- **Source primaire** : `sync_state_from_t212()` interroge `/equity/positions` (positions ouvertes) et `/equity/history/orders` (P&L réalisé, matching FIFO des lots).
- **Fallback offline** : L'état est sauvegardé dans `t212_portfolio_state.json` après chaque sync.
- **Fonctions utilitaires** : `get_t212_positions()`, `get_t212_account_summary()`, `get_t212_order_history()`.

### Injection du Prix Live dans l'Analyse
`_inject_t212_live_price()` (dans `src/data.py`) patche automatiquement la dernière barre OHLCV des ETFs tradeables avec le prix live T212 après le chargement des données Yahoo. Ne s'applique qu'aux tickers mappés (`SXRV.DE`, `SXRV.FRK`, `CRUDP.PA`).

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
*Dernière mise à jour : 19 août 2026 (GO-gates 1-3 : idempotence par réconciliation, fills confirmés, stop mouvant broker).*

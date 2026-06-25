# Plan de Décision Final — Fin Juin 2026

**Période de validation :** 2026-05-29 → 2026-06-30 (mode DEMO T212)
**Date de rédaction :** 2026-06-25
**Objectif :** Décider le passage du mode DEMO au mode RÉEL sur Trading 212.

> ⚠️ **État au 2026-06-25 (données prod réelles)** : ce plan est pré-rempli avec
> les métriques observées pendant la validation. La décision finale sera prise
> au **2026-06-30** après revue des 5 derniers jours. Les chiffres ci-dessous
> doivent être **réactualisés** avant signature (commande : `uv run python
> audit_prod_logs.py` + bloc-notes en bas de ce fichier).

---

## Résumé exécutif (à remplir le 30/06)

| Critère | Seuil de passage en RÉEL | État au 25/06 | Verdict |
|---------|--------------------------|---------------|---------|
| Stabilité système | 0 crash / timeout FinAcumen sur 7j | À confirmer | ⏳ |
| Drawdown max portefeuille | < 20% | CRUDP -18.04% | ⚠️ limite |
| Win rate consensus | > 30% | 21.5% (sous le seuil) | ❌ |
| Dérive pondération | stable (pas d'emballement) | à vérifier | ⏳ |
| Biais directionnel | distribution BUY/SELL équilibrée | SXRV 0 SELL = biais fort | ❌ |

**Recommandation préliminaire : NE PAS passer en mode RÉEL le 30/06** tant que
le biais directionnel (0 SELL sur SXRV.DE) et le win rate (< 30%) ne sont pas
corrigés. Voir détail section par section ci-dessous.

---

## 1. Validation globale du fonctionnement

### 1.1 Stabilité technique
**Critères d'acceptation :**
- [ ] `schedule.py` tourne sans crash sur 7 jours consécutifs
- [ ] `main.py` : 0 `CYCLE_TIMEOUT_SECONDS` déclenché (timeout 15 min)
- [ ] FinAcumen : `status: success` sur les 2 tickers chaque nuit (corrigé le 23/06)
- [ ] Ollama : 0 erreur "Could not find valid JSON" (défense bi-couche active)
- [ ] Cache parquet : auto-refresh fonctionnel (staleness > 1 jour)

**Données observées au 25/06 :**
- Journal : 610 lignes, période 06-01 → 06-25, 0 gap de date → cycles réguliers ✅
- TensorTrade : 226 500 timesteps cumulés (entraînement progressif OK) ✅
- FinAcumen : `success` depuis le 25/06 (réparation déployée) ✅ — **à confirmer sur 5 nuits**
- 15 alertes `daily_loss` (toutes sur CRUDP.PA) → voir §4

### 1.2 Fraîcheur des données
**Critères :**
- [ ] Caches prix couvrent la veille (17 bars en juin = OK)
- [ ] Pas de parquet EIA > 7 jours de retard
- [ ] `search_queries` : dernier `cached_at` < 24h

**Observé :** prix OK juin 2026 ✅ ; EIA affiche epoch-1970 (artefact d'index,
données présentes) ⚠️ cosmétique.

---

## 2. Validation par modèle de décision

Source : `logs_prod/model_performance.db` (5306 entrées, 587 avec outcome réel).

### 2.1 Performance individuelle (période 29/05 → 25/06)

| Modèle | Prédictions | Win rate | Confiance moy | Verdict |
|--------|------------|----------|---------------|---------|
| vincent_ganne | 303 | **23.8%** | 62.3% | meilleur (mais BUY-only) |
| classic | 587 | 21.5% | 63.7% | neutre |
| llm_text | 587 | 21.5% | 70.1% | neutre |
| llm_visual | 587 | 21.5% | 76.7% | neutre |
| sentiment | 587 | 21.5% | 54.1% | neutre |
| timesfm | 587 | 21.5% | 23.8% | neutre, faible confiance |
| tensortrade | 587 | 21.5% | 87.9% | très haute confiance ⚠️ |
| grebenkov | 587 | 21.5% | 100.0% | confiance artificielle |
| oil_bench | 284 | 19.0% | 31.8% | faible |
| hmm_model | 403 | **11.7%** | 44.4% | ❌ **défaillant** |

> ⚠️ **Le win rate uniforme à 21.5%** pour 8 modèles est suspect : il reflète
> probablement un *outcome labeling* corrélé (tous évalués sur les mêmes trades
> réels, peu nombreux). La période ne contient que 7 transactions T212 —
> l'échantillon est **trop petit** pour discriminer les modèles de façon fiable.

### 2.2 Décision par modèle (au 30/06)
- [ ] `hmm_model` : win rate 11.7% → **candidat à la désactivation** (poids → 0) ou investigation
- [ ] `tensortrade` : confiance 87.9% systématique → vérifier qu'elle n'écrase pas le consensus (calibration)
- [ ] `grebenkov` : confiance 100% constante → confiance artificielle, recalibrer ou neutraliser
- [ ] Ré-évaluer après 5 jours supplémentaires (30/06) si on étend la validation

---

## 3. Choix des poids de départ pour la future PROD

### 3.1 Poids actuels (`src/config_weights.py`)
```
classic:0.13  llm_text:0.21  llm_visual:0.19  sentiment:0.16  timesfm:0.20
vincent_ganne:0.05  oil_bench:0.05  tensortrade:0.05  grebenkov:0.05  hmm_model:0.05
```

### 3.2 Recommandation de poids PROD (proposition à valider)
Sur la base des win rates observés (en gardant la prudence : échantillon faible) :

| Modèle | Poids actuel | Poids proposé | Rationale |
|--------|-------------|---------------|-----------|
| llm_text | 0.21 | **0.22** | backbone cognitif, stable |
| llm_visual | 0.19 | **0.18** | légère baisse (win rate moyen) |
| timesfm | 0.20 | **0.15** | confiance très faible (23.8%) |
| classic | 0.13 | **0.13** | inchangé, quantitatif |
| sentiment | 0.16 | **0.14** | légère baisse |
| vincent_ganne | 0.05 | **0.06** | meilleur win rate (mais BUY-only, Nasdaq only) |
| oil_bench | 0.05 | **0.04** | win rate 19% |
| tensortrade | 0.05 | **0.04** | confiance sur-calibrée |
| grebenkov | 0.05 | **0.02** | confiance artificielle 100% |
| hmm_model | 0.05 | **0.02** | win rate 11.7% ❌ |

> **Action 30/06 :** confronter ces propositions aux poids *adaptatifs* réellement
> appliqués par `AdaptiveWeightManager` (récupérés dans `realtime_metrics`).
> Si l'adaptatif a déjà dérivé de la base, trancher : figer la base ou laisser l'adaptif.

### 3.3 Décision
- [ ] Figer les poids dans `src/config_weights.py` avant le passage RÉEL
- [ ] Désactiver l'ajustement adaptatif pendant les 2 premières semaines RÉEL (stabilisation) — *à débattre*

---

## 4. Gestion du portefeuille

### 4.1 État au 25/06 (DEMO)
- **SXRV.DE** : flat 1000€ (position fermée le 01/06, plus de re-entry)
- **CRUDP.PA** : 74.45 unités @ 13.42€ (achat 09/06), valeur actuelle **819.64€ (-18.04%)**

### 4.2 Performance
- **Alpha vs buy&hold : +0.00%** sur les deux tickers → la stratégie de signaux
  ne bat **pas** le buy&hold (SXRV.DE : 0 SELL → équivalent buy&hold ; CRUDP.PA :
  la position reste ouverte sur une baisse).
- **Drawdown CRUDP.PA : -18.04%** (proche du seuil critique 20%).
- **15 alertes `daily_loss`** toutes sur CRUDP.PA (-11% à -14% par jour sur
  plusieurs séances) → le risk manager n'a pas déclenché de stop-loss suffisant.

### 4.3 Décisions de gestion (au 30/06)
- [ ] **Biais directionnel SXRV.DE** : 302 BUY / 0 SELL / 1 HOLD. Le système ne
  sait **jamais sortir** d'une position Nasdaq en hausse. Investiguer le
  pipeline SELL (le risk_manager produit bien des SELL sur CRUDP, jamais sur SXRV).
- [ ] **Stop-loss / trailing-stop** : vérifier pourquoi CRUDP.PA a perdu -18%
  sans déclenchement de protection. Le `AdvancedRiskManager` est-il armé ?
- [ ] Diversification : 2 tickers seulement → concentration. Envisager un 3e/4e
  ticker en PROD RÉEL pour réduire le risque idiosyncratique.

---

## 5. Autres points à examiner

### 5.1 Biais HOLD/BUY (problème structurel détecté)
- SXRV.DE : `Risk_Adjusted` = 153 HOLD / 97 BUY / 53 STRONG_BUY / **0 SELL**
- CRUDP.PA : 196 HOLD / 83 BUY / 25 STRONG_BUY / 3 SELL
→ **Le système ne produit quasi jamais de SELL.** C'est un biais haussier
structurel qui empêche de protéger le capital en baisse. **Bloquant pour le
passage en RÉEL.**

### 5.2 FinAcumen (réparé le 23/06)
- Maintenant `success` en prod (HOLD 0.75 / BUY 0.85 le 25/06) ✅
- **Mais NON branché au consensus temps réel** — il ne fait qu'alimenter le
  morning brief. Décision à prendre : le câbler comme 11ᵉ vote avant le RÉEL,
  ou le laisser en advisory (brief) pour la V1 RÉEL.

### 5.3 Dépendances externes
- [ ] Ollama : doit être up sur le serveur PROD RÉEL (FinAcumen + LLM en dépendent)
- [ ] yfinance : fragilité circuit-breaker — avoir un plan B prix
- [ ] T212 : valider `T212_ENV=live` dans `.env.t212` avant le 1er trade réel

### 5.4 Sécurité / secrets
- [ ] Vérifier qu'aucun secret n'est commité (`.env*` gitignored)
- [ ] Rotation clé API T212 avant passage RÉEL

---

## 6. Décision Finale : passage DEMO → RÉEL

### 6.1 Conditions de go/no-go (toutes doivent être ✅)

| # | Condition | État 25/06 |
|---|-----------|-----------|
| C1 | Stabilité système 7j sans crash | ⏳ (à confirmer) |
| C2 | FinAcumen success 5 nuits consécutives | ⏳ (depuis 25/06) |
| C3 | Win rate consensus > 30% | ❌ (21.5%) |
| C4 | Drawdown max < 15% | ❌ (CRUDP -18%) |
| C5 | Au moins 1 SELL déclenché sur SXRV.DE | ❌ (0 SELL) |
| C6 | Stop-loss fonctionnel vérifié | ❌ (à investiguer) |

**Recommandation au 25/06 : 🛑 NO-GO.** C3, C4, C5, C6 ne sont pas remplis.
Étendre la validation de **2 à 4 semaines** pour corriger le biais SELL et
ré-armer le stop-loss avant tout argent réel.

### 6.2 Si GO au 30/06 (ou plus tard) — paramètres PROD RÉEL

**Cash de départ proposé :**
- **Total : 2 000 €** (1 000 € par ticker, identique à la DEMO pour continuité)
- *Option conservative : 500 € par ticker (1 000 € total)* pour limiter le risque
  pendant les premières semaines RÉEL.

**Budget par ticker (`INITIAL_BUDGETS`) :**
```
SXRVd_EQ: 1000  (ou 500 en conservative)
CRUDl_EQ: 1000  (ou 500 en conservative)
```

**Frais :** T212 0.1% par trade (déjà modélisé dans backtest).

**Limite de perte (kill-switch manuel) :**
- Stop global : -10% du capital total → pause immédiate, revue manuelle
- Stop par ticker : -15% → fermeture automatique de la position

**Stratégie de montée en charge :**
1. Semaine 1 RÉEL : 500 €/ticker, monitoring quotidien
2. Semaine 2 RÉEL : 1 000 €/ticker si tout est vert
3. Semaine 4 RÉEL : revue et ajustement des poids

---

## 7. Conclusion et suivi pendant la PROD RÉEL

### 7.1 Conclusion (à rédiger le 30/06)
> _À compléter après décision._

### 7.2 Suivi RÉEL (checklist hebdomadaire)
- [ ] **Lundi :** `uv run python audit_prod_logs.py` → revue verdict + FinAcumen
- [ ] **Lundi :** revue `trading_journal.csv` (distribution signaux, drift)
- [ ] **Mercredi :** revue `model_performance.db` (win rate par modèle)
- [ ] **Vendredi :** revue portefeuille (drawdown, P&L réalisé vs DEMO)
- [ ] **Alerte auto :** toute `daily_loss` critique → investigation sous 24h

### 7.3 Critères de rollback (DEMO ← RÉEL)
- Drawdown total > 10% → retour DEMO immédiat
- 3 `daily_loss` critiques consécutifs sur un ticker → pause
- Instabilité système (Ollama/FinAcumen down > 1 cycle) → pause

### 7.4 Métriques de succès RÉEL (à 30 jours)
- Sharpe > 0.5
- Max drawdown < 10%
- Win rate trades fermés > 40%
- 0 intervention manuelle d'urgence

---

## Notes de mise à jour

| Date | Auteur | Mise à jour |
|------|--------|-------------|
| 2026-06-25 | ZCode | Rédaction initiale (pré-rempli avec données prod 25/06) |
| 2026-06-30 | _à faire_ | Réactualiser toutes les métriques + décision GO/NO-GO finale |

**Commande pour réactualiser les données avant le 30/06 :**
```bash
uv run python audit_prod_logs.py
# puis : .venv\Scripts\python.exe -m pytest tests/ -q   (sanity)
```

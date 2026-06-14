# Plan : Analyse critique de la rentabilité du système + Roadmap

## Objectif
Produire un **rapport critique unique** (remplaçant/complétant `RAPPORT_AUDIT_GLOBAL.md`, jugé trop flatteur) + une **roadmap d'améliorations priorisées P0/P1/P2** prête à implémenter. Angle dominant : **rentabilité** (espérance de gain, rapport signal/bruit, capacité à générer un edge net de frais). Les ajouts sont livrés en **spécifications implémentables** (fichier, complexité, risque, verdict viable/non-viable).

Contexte clé : aucune donnée de performance réelle exploitable (`logs_prod/` absent, `trading_journal.csv` ≈ 3 lignes). Le jugement de rentabilité se fera donc par **raisonnement structurel** (edge attendu vs frais/drawdown) + recommandations de **mesure** pour combler le manque.

## Livrable final
Un fichier : `docs/RAPPORT_RENTABILITE_ET_ROADMAP.md` (structuré en 6 sections ci-dessous). Aucune modification du code source dans ce plan — c'est une analyse/document. La roadmap listera les actions à exécuter **ensuite**, sur validation.

---

## Section 1 — Analyse critique des 12 modèles de décision

Pour chaque modèle : rôle, qualité d'implémentation, **utilité réelle (valeur prédictive)**, biais, calibration. Constats déjà collectés (à formaliser) :

| # | Modèle | Fichier | Verdict préliminaire |
|---|---|---|---|
| 1 | Classic (RF/GB/LR) | `classic_model.py` | Solide (TimeSeriesSplit, cache MD5). Biais d'entraînement mineur (shuffle=False OK). |
| 2 | TimesFM 2.5 | `timesfm_model.py` | Wrapper OK mais **calibration défaillante** : seuil en vol quotidienne vs rendement à 5j (incohérence d'horizon), **quantiles ignorés**, confiance arbitraire (`×50`), état position non persisté. |
| 3 | TensorTrade PPO | `tensortrade_model.py` | **Valeur prédictive quasi nulle** : 2000+500 timesteps (2-3 ordres sous le minimum), apprentissage incrémental = catastrophic forgetting, biais beta (rester investi). Bruit dans le consensus. |
| 4 | OilBench (Gemma) | `oil_bench_model.py` | Bug critique : `headlines=None` en prod (`enhanced_trading_example.py:627`) → news jamais injectées. **Jitter `random()`** non reproductible (`:199`). Discontinuité de confidence au seuil BUY/HOLD. Asymétrie bullish des seuils. |
| 5 | Text LLM (Gemma) | `llm_client.py` | Fort (fusionneur contextuel) mais `temperature=0.4` élevée, biais bullish unilatéral (règle funding contrarian BUY sans symétrique), `failed:True` non consommé → HOLD bias silencieux. |
| 6 | Visual LLM (Gemma) | `llm_client.py` | OK, temp 0.1. Dépend de la qualité du chart. |
| 7 | Sentiment | `sentiment_analysis.py` + `news_fetcher.py` | **Modèle fantôme** : AlphaEar = constante `0.1`, sentiment Google News ignoré, AV rate-limited → vote HOLD ~100% du temps à confidence 0.5. |
| 8 | Hyperliquid | (dans features/prompts) | Signal contrarian réel (funding/OI) mais biais unilatéral documenté. |
| 9 | Vincent Ganne | `enhanced_decision_engine.py:101` | Détecteur de *market bottom* Nasdaq, BUY-only, hard-block si WTI≥94$. Conceptuel/intéressant mais seuils hardcodés datés (WTI max 94). |
| 10 | Grebenkov ARP | `grebenkov_model.py` | Mathématiquement sérieux (RIE filter, ARP, ATR-adaptive). Un des rares modèles avec fondement théorique solide. |
| 11 | HMM | `hmm_model.py` | Baum-Welch/Viterbi implémentés main (slow, mais OK lookback 252j). Détection de régime. |
| 12 | Web Researcher | `web_researcher.py` | Enrichissement intermittent (timeout fréquent), `verbose=True` en prod. Valeur indirecte. |

**Conclusion section 1** à formaliser : sur 12 modèles, ~4-5 ont une valeur prédictive réelle (Classic, TimesFM à recalibrer, Grebenkov, Text LLM, HMM), ~3 sont dégradés/buggés (OilBench, Sentiment, Visual intermittent), ~1 est du bruit pur (TensorTrade), le reste est utilitaire.

---

## Section 2 — Compilation des avis et décision finale (consensus)

Analyse de `enhanced_decision_engine.py` + `config_weights.py` :

- **Agrégation** : `weighted_score += signal_value × confidence × model_weight` (`:623`). Score ajusté par régime + volatilité + RSI.
- **Seuils adaptatifs** (`:331-338`) : `buy=0.12`, `strong_buy=0.35`. Très **asymétriques** (BUY à 0.12, SELL à -0.15) → biais bullish structurel.
- **Consensus score** = moyenne accord signal + alignement confiance. **Disagreement factor** pénalise les conflits.
- **Super-Consensus Boost** (`:649-658`) : +0.20 si Classic+TimesFM d'accord → favorise les signaux techniques, mais les deux modèles sont corrélés (mêmes features).
- **Poids de base** (`config_weights.py`) : LLM (40% combiné) dominant, quant (33%), spéculatifs 25%. Cohérent avec la philosophie "cognitive".
- **Régime adjustments** : trending booste grebenkov/timesfm, ranging booste classic/tensortrade. Mais TensorTrade boosté = booster du bruit.

**Problèmes critiques pour la rentabilité** :
1. Le consensus **ne consomme pas `failed:True`** → un modèle qui crash vote HOLD à pleine importance → biais HOLD qui tue l'edge.
2. La pondération par `confidence` est **auto-référentielle** : chaque modèle définit sa propre échelle de confiance (TimesFM sature à 2%, TensorTrade ~0.4, sentiment 0.65 au seuil). Comparer ces confidences entre modèles est **non valide** — il faudrait une calibration (Platt/isotonique) normalisée.
3. Les seuils BUY/SELL asymétriques + le `_apply_risk_management` (MIN_CONFIDENCE 0.20/0.40) créent une **région morte** où le système ne trade presque jamais.

---

## Section 3 — Gestion du portefeuille (sizing + exécution)

Analyse de `t212_executor.py` + `advanced_risk_manager.py` :

- **Sizing all-in** (`t212_executor.py:515`) : `budget × 0.95` à chaque BUY. Aucun sizing par risque (Kelly calculé mais **non utilisé** côté T212 — le risk_manager calcule un `recommended_size` ignoré par l'exécuteur).
- **Budgets hardcodés 1000€** indépendants du capital réel du compte.
- **Contradiction trailing-stop / loss-guard** : le trailing (3% from peak, +0.5% profit) est court-circuité par `_check_sell_loss_guard` (bloque vente < -0.2%) → sortie impossible dans certains scénarios.
- **Pas de vérification de fill** : prix enregistré = prix estimé (yfinance), pas le fill réel T212.
- **Prix source hétérogène** : `get_t212_price` ne marche que sur positions ouvertes → sinon yfinance (potentiellement USD/stale).
- **Ordres market uniquement** → slippage non maîtrisé en marché volatil.
- **Rate limiting naïf** (backoff linéaire, pas de `Retry-After`).

**Conclusion section 3** : la gestion du portefeuille est le **maillon le plus faible** pour la rentabilité. Le sizing all-in + les contradictions stop/guard + l'absence de vérif fill = espérance de gain dégradée mécaniquement, indépendamment de la qualité des signaux.

---

## Section 4 — Analyse de viabilité / rentabilité (angle dominant)

Structure du raisonnement (puisqu'aucun track record) :

1. **Bilan des forces/faiblesses pour l'edge** :
   - Forces : diversification modèle, consensus anti-hallucination, risk management centralisé, données fondamentales (EIA).
   - Faiblesses fatales à l'edge : consensus par confiance non calibrée, TensorTrade = bruit, sentiment fantôme, HOLD bias silencieux, sizing all-in, contradictions stop/guard.
2. **Estimation du rapport signal/bruit** : ~5/12 modèles utiles, pondération non calibrée → dilution de l'edge par le bruit.
3. **Analyse coûts vs edge** : frais T212 0.1% × 2 par cycle, fréquence de trading, trailing stop à 0.5% profit → l'espérance par trade doit dépasser ~0.3% net pour être positive (hors drawdown).
4. **Verdict de viabilité** : **Non viable en l'état pour dégager un alpha net**, mais **viable après la roadmap P0** (qui cible les biais structurels et le bruit). Architecturalement sain, économiquement immature.
5. **Recommandations de MESURE obligatoires** avant tout jugement définitif (combler le manque de track record) — cf. section 6.

---

## Section 5 — Ajouts viables (spécifications implémentables)

Chaque ajout : fichier cible, complexité (S/M/L), risque, verdict viable/non-viable. Seront évalués :

1. **Calibration des confidences (Platt scaling)** — normaliser les confiances de chaque modèle sur une échelle commune. Cible `enhanced_decision_engine.py`. Complexité M. Risque faible. **Viable (P1).**
2. **Consommation du flag `failed:True`** — exclure/pénaliser les modèles en échec du consensus. Cible `enhanced_decision_engine.py:619`. Complexité S. **Viable (P0).**
3. **Recalibration TimesFM** (quantiles + seuil `×√horizon`) — `timesfm_model.py:131`. Complexité M. **Viable (P0).**
4. **Suppression/Rentraînement TensorTrade** — soit retirer du consensus, soit ≥10⁵ timesteps + replay périodique. Cible `tensortrade_model.py`. Complexité L. **Viable mais décision tranchée (P1).**
5. **Sizing par volatilité (vol-target)** — remplacer all-in par `capital × target_vol / asset_vol`. Cible `t212_executor.py:515` + `advanced_risk_manager.py`. Complexité M. **Viable (P0).**
6. **Résolution contradiction trailing/loss-guard** — unifier la logique de stop. Cible `t212_executor.py:423-580`. Complexité S. **Viable (P0).**
7. **Vérification de fill via orderId** — polling du statut après ordre market. Cible `t212_executor.py:535`. Complexité M. **Viable (P1).**
8. **Correction OilBench** (`headlines` réels + jitter déterministe + confidence continue). Cible `oil_bench_model.py` + `enhanced_trading_example.py:627`. Complexité S. **Viable (P0).**
9. **Sentiment réel** (LLM/lexique au lieu de constante AlphaEar). Cible `news_fetcher.py:100`. Complexité M. **Viable (P1).**
10. **Grid search des poids via `backtest_prod.py`** — optimisation offline (déjà en TODO). Complexité M. **Viable (P1) — nécessite d'abord des logs_prod.**
11. **Stop-loss/take-profit broker-side** (STOP_LIMIT chez T212). Complexité M. **À évaluer selon API T212.**
12. **Backtest déterministe du consensus** (replay des signaux historiques hors-prod). Complexité L. **Viable (P2).**

Ajouts « recherche » évalués mais probablement **non-viables ou prématurés** : ajout d'un 13e modèle LLM, passage à GPU, trading intraday (la philosophie "cognitive prudence" + CPU-only les contredit).

---

## Section 6 — Roadmap P0 / P1 / P2 + plan de mesure

Synthèse actionnable dérivée des sections 1-5 :

**P0 (bloque la rentabilité — à faire avant tout jugement) :**
- Consommer `failed:True` dans le consensus
- Recalibrer TimesFM (quantiles + horizon)
- Sizing par volatilité (fin de l'all-in)
- Résoudre contradiction trailing/loss-guard
- Corriger OilBench (headlines + jitter + confidence)
- (Optionnel) Retirer TensorTrade du consensus tant que non réentraîné

**P1 (optimise l'edge) :**
- Calibration Platt des confidences
- Vérification de fill T212
- Sentiment réel
- Grid search des poids (après accumulation de logs_prod)
- Réentraînement TensorTrade complet OU suppression définitive

**P2 (maturité) :**
- Backtest déterministe du consensus
- Stop-loss broker-side
- Index DB + DB_PATH absolu (hygiène)

**Plan de mesure (indispensable, aucune donnée aujourd'hui) :**
- Mettre en place l'accumulation fiable de `logs_prod/trading_journal.csv` sur ≥ 60 jours
- Activer `backtest_prod.py` mensuellement dès qu'assez de signaux
- Définir des seuils de décision (Sharpe > 0.5, alpha > buy&hold, max drawdown acceptable) pour la revue

---

## Méthode d'exécution (en mode Act)
1. Créer `docs/RAPPORT_RENTABILITE_ET_ROADMAP.md` avec les 6 sections formalisées à partir des constats ci-dessus + des rapports des deux sous-agents explore déjà produits.
2. Ajouter des références `file:line` précises pour chaque constat.
3. Inclure un tableau de synthèse (modèles × verdict) et un tableau de roadmap (action × priorité × complexité × fichier).
4. Aucune modification de code source (ce plan est un livrable documentaire). La roadmap sera exécutée sur validation, plan par plan.

## Hors périmètre (explicite)
- Ne pas implémenter les correctifs P0/P1/P2 dans ce plan (livrable = analyse + roadmap).
- Ne pas supprimer `RAPPORT_AUDIT_GLOBAL.md` (le nouveau rapport le complète/corrige).
- Ne pas modifier le code source.

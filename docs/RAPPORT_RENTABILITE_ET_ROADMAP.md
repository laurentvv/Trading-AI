# Rapport critique de rentabilité & Roadmap d'améliorations (P0/P1/P2)

> **Statut du document** : Analyse critique — *supersede et corrige* `RAPPORT_AUDIT_GLOBAL.md` (racine du dépôt), jugé trop flatteur et structurellement incomplet (cf. §0.2).
> **Angle dominant** : **rentabilité** — espérance de gain nette de frais, rapport signal/bruit, capacité à produire un *edge*.
> **Date** : 2026-06-14 — période de validation en cours (2026-05-29 → 2026-06-30).
> **Périmètre** : document d'analyse uniquement. Aucune modification de code source. La roadmap (§6) sera exécutée plan par plan sur validation.

---

## 0. Avertissement méthodologique

### 0.1 Données de performance : indisponibles à la rédaction, attendues fin Juin 2026

À la date de rédaction (2026-06-14), le jugement de rentabilité ne peut pas encore s'appuyer sur un *track record* :

- `logs_prod/` est **absent** à ce jour. Ce dossier n'est **pas** une sortie automatique du pipeline : c'est un **snapshot copié manuellement par l'opérateur** depuis la prod (log, DB, journal, sorties morning brief) pour l'analyse et la validation hors-ligne. Aucune copie n'a encore été réalisée.
- En prod, `trading_journal.csv` ne contient que **3 lignes** (en-tête + 2 enregistrements) — statistiquement nul.
- Le système est en « période de validation » (2026-05-29 → 2026-06-30) selon `memory-bank/activeContext.md`.

**Toutefois, cette absence est transitoire.** La phase de test sur 1 mois (mai→juin 2026) est en cours : les artefacts de prod (log, DB, journal, sorties morning brief) s'accumulent à leurs emplacements naturels pendant l'exécution. Fin Juin 2026, l'opérateur **copiera manuellement un mois complet** de ces artefacts dans `logs_prod/` pour constituer le premier *track record* réel analysable — permettant de passer du raisonnement structurel à des **métriques empiriques** (Sharpe, alpha, drawdown, calibration par modèle).

**Conséquence (à la rédaction)** : le verdict de rentabilité présenté ici repose sur du **raisonnement structurel** (edge attendu vs frais/drawdown, qualité du pipeline signal → exécution). Il devra être **révisé fin Juin 2026** à l'aune des données réelles accumulées. Le §6.4 définit le point de contrôle correspondant.

**Instrumentation déjà en place** (la phase de validation est déjà outillée, pas à construire) :
- **Log live** `trading.log` écrit par `src/bootstrap.py:5-23` (RotatingFileHandler 5 Mo × 5 backups) — trace d'exécution en temps réel.
- **Analyseur live** `morning_brief/tools/analyze_trading_logs.py` — parse le log live (erreurs, warnings, déconnexions API, slippage, `health_score`). Outil du morning brief, **non dédié à `logs_prod/`**.
- **Archive `logs_prod/`** : **copie manuelle** (par l'opérateur) des artefacts de prod — `trading_journal.csv` (signaux), `trading_history.db` (trades T212 réels), `performance_monitor.db` (métriques temps réel par ticker) + sorties morning brief (cf. `tests/check_db.py:30-31`, `backtest_prod.py:29,237,307`). Une première copie complète (mois de Juin) sera faite fin Juin 2026.
- **Outils d'analyse d'archive existants** : `morning_brief/tools/audit_portfolio_performance.py` (lit `performance_monitor.db` : PnL, drawdown, alertes) et `backtest_prod.py` (replay des signaux `logs_prod/trading_journal.csv` contre prix + frais T212). Ces outils tournent sur le **snapshot** `logs_prod/`, pas sur un flux auto.

### 0.2 Correction de `RAPPORT_AUDIT_GLOBAL.md`

Le rapport d'audit précédent (racine) est **maintenu** pour l'historique mais ses conclusions sont **explicitement invalidées** sur les points suivants :

| Sujet | Conclusion de `RAPPORT_AUDIT_GLOBAL.md` | Réalité vérifiée dans le code |
|---|---|---|
| TensorTrade PPO | « bonne intégration » | Valeur prédictive quasi nulle : 2000+500 *timesteps* (`tensortrade_model.py:17-18`), oubli catastrophique par *fine-tuning* incrémental (`:204`). **Bruit dans le consensus.** |
| Sentiment | « approche pragmatique » | **Modèle fantôme** : sentiment AlphaEar = constante `0.1` (`news_fetcher.py:100`), sentiment Google News jamais agrégé (`:124`, `:150-155`). Vote HOLD ~constant. |
| OilBench | Intégration fonctionnelle | Bug de câblage : `headlines=None` en production (`enhanced_trading_example.py:627`) + jitter `random()` non reproductible (`oil_bench_model.py:199`). |
| Sizing | Non identifié comme risque | **All-in** : `budget × 0.95` à chaque achat (`t212_executor.py:515`); le `recommended_size` du risk manager est calculé puis ignoré (`enhanced_trading_example.py:656`). |

Le présent rapport est la **source de vérité** courante pour l'évaluation de la rentabilité.

---

## 1. Analyse critique des 12 modèles de décision

Chaque modèle est évalué sur : rôle, qualité d'implémentation, **valeur prédictive réelle**, biais, calibration. Références `file:line` vérifiées dans le code.

| # | Modèle | Fichier | Verdict | Confiance* |
|---|---|---|---|---|
| 1 | Classic (RF/GB/LR) | `classic_model.py` | **Solide** — TimeSeriesSplit 5 folds (`:124`,`:180`), cache MD5 (`:22`), `shuffle=False`. Biais d'entraînement mineur. | ~0.5–0.7 (proba) |
| 2 | TimesFM 2.5 | `timesfm_model.py` | **Wrapper OK, calibration défaillante** — confiance `×50` saturée à 2 % (`:135`), seuil en vol. *quotidienne* vs rendement à 5 j (`:100-105` vs `:132`), **quantiles ignorés** (`:127`), état position non persisté (`:44`). | 0 → 1.0 (saturation) |
| 3 | TensorTrade PPO | `tensortrade_model.py` | **Bruit pur** — 2000+500 *timesteps* (`:17-18`), *fine-tuning* incrémental = oubli catastrophique (`:204`), biais « rester investi » (`:91` vs `:95`). | ~0.33–0.45 (softmax) |
| 4 | OilBench (Gemma) | `oil_bench_model.py` | **Dégradé par bugs** — `headlines=None` en prod (`enhanced_trading_example.py:627`), jitter `random()` (`:199`), **discontinuité de confiance** au seuil BUY/HOLD (`:186` vs `:201`). | HOLD ~0.30 ; BUY seuil 0.10 |
| 5 | Text LLM (Gemma) | `llm_client.py` | **Fort mais biaisé** — fusionneur contextuel, `temperature=0.4` élevée (`:203`), `failed:True` non consommé (`:64-68`). Biais *bullish* unilatéral (règle funding *contrarian* BUY, cf. §2). | auto-déclarée 0–1 |
| 6 | Visual LLM (Gemma) | `llm_client.py` | **OK** — `temperature=0.1`, dépend de la qualité du chart (`:216`). Dégrade en HOLD sur timeout (`enhanced_trading_example.py:498`). | auto-déclarée 0–1 |
| 7 | Sentiment | `sentiment_analysis.py` + `news_fetcher.py` | **Modèle fantôme** — AlphaEar = constante `0.1` (`:100`), Google News sentiment `0.0` non agrégé (`:124`,`:150-155`), Alpha Vantage *rate-limité* (`:75`). Vote HOLD ~constant à confiance ≥0.5 (`:19`). | ≥ 0.5 (plancher) |
| 8 | Hyperliquid | (dans prompts/features) | Signal *contrarian* réel (funding/OI) mais **biais unilatéral** (règle BUY uniquement, `enhanced_decision_engine.py:224-228`). | n/a (enrichissement) |
| 9 | Vincent Ganne | `enhanced_decision_engine.py:101` | Détecteur de *market bottom* Nasdaq, **BUY-only**, hard-block WTI≥94 $ (`:263`). Seuils hardcodés datés. | 0–1 (score/max) |
| 10 | Grebenkov ARP | `grebenkov_model.py` | **Sérieux** — filtre RIE (Bun et al. 2016, `:42`), ARP, seuils ATR-adaptatifs (`:186`,`:207`). Fondement théorique solide. | ~0–1 |
| 11 | HMM | `hmm_model.py` | **Correct** — Baum-Welch/Viterbi implémentés main (`:31`,`:76`), 2 états, *lookback* 252 j (`:136`). Lent mais valide. Détection de régime. | ~0–1 |
| 12 | Web Researcher | `web_researcher.py` | **Enrichissement intermittent** — `verbose=True` en prod (`:275`), `temperature=0.4` (`:204`), timeouts fréquents. Valeur indirecte. | n/a (contexte) |

*\*Confiance* = échelle brute produite par chaque modèle **avant** toute normalisation. L'incomparabilité de ces échelles est le défaut central analysé au §2.

### Détails des modèles à valeur prédictive dégradée

**#2 TimesFM — calibration incohérente.** Le seuil d'action `_adaptive_threshold` (`timesfm_model.py:100-105`) est dérivé de la **volatilité quotidienne** (`realised_vol * 0.5`, `:105`) mais comparé à l'`expected_return` calculé sur l'**horizon complet** (5 jours par défaut, `:132`). Un rendement à 5 j présente naturellement un écart-type ~√5× supérieur à la vol. quotidienne : le seuil est donc structurellement trop bas → **sur-signalement**. Par ailleurs, la `confidence = min(1.0, abs(expected_return) * 50)` (`:135`) sature à 1.0 pour tout mouvement attendu ≥ 2 %, soit l'immense majorité des cas. Enfin, `_positions` est un dict en mémoire (`:44`), non persisté ; en cas de redémarrage du processus entre cycles, l'état revient à `FLAT` et le filtre position-aware (`:146-148`) **convertit tout signal SELL en HOLD** — biais *bullish* latent dépendant du mode de lancement.

**#3 TensorTrade — bruit dans le consensus.** 2000 *timesteps* initiaux + 500 de *fine-tuning* (`tensortrade_model.py:17-18`) représentent 2 à 3 ordres de grandeur sous le minimum viable pour PPO (>10⁵). Le *fine-tuning* incrémental à chaque cycle (`:204`) provoque un **oubli catastrophique** (la politique dérive sur la dernière fenêtre vue). La structure de récompense pénalise le *flat* (`-0.01`, `:95`) et récompense le *holding* (`price_change/atr`, `:91`) → biais bêta structurel (« rester investi »). La confiance produite est une probabilité softmax ~0.33–0.45 (`:243`), qui sera **écrasée** par les confidences saturées de TimesFM/LLM dans le consensus.

**#4 OilBench — bug de câblage + discontinuité.** `enhanced_trading_example.py:627` appelle `oil_model.analyze(ticker=..., headlines=None)` alors que les *headlines* sont disponibles à ce point (récupérées en Phase C, `enhanced_trading_example.py:451`). Le prompt OilBench reçoit donc toujours « No recent oil-specific news available » (`oil_bench_model.py:130`). La discontinuité de confiance est plus subtile : à `allocation=55` (BUY), `confidence = (55-50)/50 = 0.10` (`:189`) tandis qu'à `allocation=54.99` (HOLD), `confidence = 0.3 + random*0.05 ≈ 0.32` (`:201`). **Un vote HOLD pèse ~3× plus qu'un vote BUY faible** — inversion de la sémantique de confiance. Le `import random` *in-function* (`:199`) brise en outre la reproductibilité des *backtests*.

**#7 Sentiment — modèle fantôme.** `news_fetcher.py:100` retourne une **constante** `0.1` comme sentiment AlphaEar (« neutral/positive default »), indépendante du contenu réel. `fetch_google_news_rss` renvoie toujours `0.0` (`:124`) et n'est agrégée que comme source de *headlines*, jamais comme score (`:150-155`). Alpha Vantage est *rate-limité* (`time.sleep(12)`, `:75`). Conséquence : `get_sentiment_decision_from_score` (`sentiment_analysis.py:7`) reçoit presque toujours un score ≈ 0.1–0.13 → vote **HOLD à confiance ≥ 0.5** (`:19`). Le « modèle sentiment » est donc un vote HOLD à poids 16 % (`config_weights.py:10`), contribution nulle à l'edge.

### Synthèse section 1

Sur 12 modèles : **~5 ont une valeur prédictive réelle** (Classic, TimesFM à recalibrer, Grebenkov, Text LLM, HMM), **~3 sont dégradés/buggés** (OilBench, Sentiment, Visual intermittent), **~1 est du bruit pur** (TensorTrade), le reste est utilitaire (Vincent Ganne, Hyperliquid, Web Researcher). Le consensus agrège donc **~40 % de bruit/de votes morts** à pleine pondération.

---

## 2. Compilation des avis et décision finale (consensus)

Source : `enhanced_decision_engine.py` + `config_weights.py`. Entrée principale : `make_enhanced_decision()` (`enhanced_decision_engine.py:541`).

### 2.1 Mécanique d'agrégation

- **Score pondéré** : `weighted_score += signal_value × decision.confidence × model_weight` (`enhanced_decision_engine.py:619-623`). Le `model_weight` provient des poids normalisés (somme ramenée à 1.0, `:569-571`).
- **Poids de base** (`config_weights.py:6-17`) : `llm_text` 0.21, `timesfm` 0.20, `llm_visual` 0.19, `sentiment` 0.16, `classic` 0.13, puis 0.05 chacun pour `vincent_ganne`, `oil_bench`, `tensortrade`, `grebenkov`, `hmm_model`. **Les LLM (texte+visuel) dominent à ~35 %** post-normalisation, le quantitatif (classic+timesfm+tensortrade) ~33 %, le sentiment 14 %, les spéculatifs ~18 %.
- **Ajustements de régime** (`enhanced_decision_engine.py:343-346`) : *trending* booste grebenkov/timesfm ; *ranging* booste classic/**tensortrade** → on **booste le bruit** en marché *ranging*.
- **Seuils adaptatifs asymétriques** (`enhanced_decision_engine.py:331-338`) : `buy=0.12`, `strong_buy=0.35`, `sell=-0.15`, `strong_sell=-0.45`. Le seuil d'achat est **plus proche de zéro** que le seuil de vente (0.12 < 0.15) → **biais bullish structurel** dans le mapping final.
- **Super-Consensus Boost** (`enhanced_decision_engine.py:649-658`) : +0.20 si Classic et TimesFM d'accord. Mais ces deux modèles consomment des **features corrélées** → le « consensus » est en partie une redondance, pas une diversification.
- **Risk management** (`enhanced_decision_engine.py:469-491`) : `MIN_CONFIDENCE_FOR_ACTION=0.20` (achat), `MIN_CONFIDENCE_FOR_SELL=0.40` (`:302-303`). Exigence de confiance **2× plus élevée pour vendre** qu'acheter → seconde source de biais bullish.

### 2.2 Trois défauts critiques pour la rentabilité

**Défaut A — Confiance non calibrée (auto-référentielle).**
Chaque modèle définit sa propre échelle de confiance, et ces échelles sont **incomparables** :

| Modèle | Échelle de confiance | Au seuil d'action |
|---|---|---|
| TimesFM (`:135`) | `min(1.0, |ret|×50)` | saturée à 1.0 dès 2 % |
| TensorTrade (`:243`) | softmax 3 actions | ~0.40 |
| Sentiment (`:19`) | `0.5 + |score|` | **0.50 minimum** (même sans info) |
| OilBench HOLD (`:201`) | `0.3 + random*0.05` | ~0.32 |
| Classic | `predict_proba` | ~0.5–0.7 |

Multiplier `signal × confidence × weight` revient à comparer des quantités sans unité commune. Un TimesFM BUY à confiance 1.0 contribue **~5–10×** la magnitude d'un TensorTrade BUY à 0.4, indépendamment de la qualité réelle du signal. **Le consensus est donc dominé par les modèles à confiance saturée, pas par les modèles les plus précis.** Solution : calibration (Platt/isotonique) normalisée — cf. §5.1.

**Défaut B — `failed:True` non consommé (biais HOLD par déflation du score).**
Lorsqu'un LLM échoue, `_fallback_decision` (`llm_client.py:60-72`) retourne `{signal: "HOLD", confidence: 0.0, failed: True}`. La docstring (`:64-68`) **avertit explicitement** que ce flag n'est pas consommé par l'agrégateur. Tracé dans `make_enhanced_decision` : la décision est lue via `getattr(dec, "signal"/"confidence")` (`enhanced_decision_engine.py:509-514`), le champ `metadata.failed` n'est jamais inspecté.

Effet réel (vérifié) : un modèle échoué contribue **0 au score pondéré** (HOLD×0.0×weight), mais **ses poids ne sont pas renormalisés**. Si le Text LLM (poids ~21 %) échoue, le `weighted_score` perd jusqu'à ~0.42 de magnitude potentielle et se rapproche de zéro → atterrissage dans la **bande HOLD [-0.15, 0.12]**. Parallèlement, `_calculate_consensus_score` (`:400-425`) compte ce HOLD(0) comme un signal « d'accord », ce qui **gonfle artificiellement** le `consensus_score` puis la `final_confidence`. Résultat net : un échec LLM intermittents **décale silencieusement la décision vers HOLD** tout en **surévaluant la confiance** du résultat. C'est un biais HOLD + une métrique de confiance trompeuse.

**Défaut C — Région morte de trading.**
La combinaison seuils asymétriques (`:331-338`) + `MIN_CONFIDENCE_FOR_SELL=0.40` (`:302`) + confiance non calibrée crée une **région morte** où le système émet rarement un SELL exécutable : il faut à la fois un score ≤ -0.15 (plus dur que +0.12) ET une confiance ≥ 0.40 (double de l'achat). Le système est structurellement un **acheteur réticent à vendre** — incompatible avec une gestion asymétrique du risque.

---

## 3. Gestion du portefeuille (sizing + exécution)

Source : `t212_executor.py` + `advanced_risk_manager.py`. **C'est le maillon le plus faible pour la rentabilité.**

### 3.1 Sizing « all-in » avec risk manager ignoré

- L'exécuteur calcule `target_budget = min(available_cash, portfolio["cash"]) × 0.95` (`t212_executor.py:515`) → **95 % du cash alloué à chaque BUY**, indépendamment du signal, de la confiance ou de la volatilité.
- Le `advanced_risk_manager.calculate_position_sizing()` (`advanced_risk_manager.py:350`) calcule un `recommended_size` (Kelly fractionnel `:346`, ajustement force/risque `:397-404`). Ce `recommended_size` est **calculé et journalisé** (`enhanced_trading_example.py:656-666`) puis **jamais transmis** à l'exécuteur : `execute_t212_trade()` (`t212_executor.py:668-674`) ne reçoit que `(signal, confidence, ticker, ...)`, **aucune taille**. Deux systèmes de sizing parallèles qui ne se connectent jamais.
- Budgets **hardcodés à 1000 €** par ticker (`t212_executor.py:88-92`), déconnectés du capital réel du compte (récupéré via `get_t212_account_summary`, `:134`, mais non utilisé pour le sizing).

### 3.2 Contradiction « trailing stop » vs « loss guard » — aucun repreneur de pertes

Deux gardes de vente coexistent avec des logiques **incompatibles** :

1. **Trailing stop** (`_evaluate_trailing_stop`, `t212_executor.py:423-453`) : déclenche SELL si `drop_from_peak ≥ 3 %` **ET** `profit_margin > 0.5 %` (`:448`). → ne sort **que les positions en gain**.
2. **Loss guard** (`_check_sell_loss_guard`, `t212_executor.py:565-580`) : **bloque** toute vente si `current_value < reference_cost × 0.998` (≈ -0.2 %, `:575`). → interdit de **réaliser une perte**, même minime.

Lecture croisée vérifiée dans `_execute_sell_order` (`:634-666`) : tout SELL passe par `_check_sell_loss_guard` (`:648`), qui retourne `None` (vente avortée, silencieuse) dès que la position est sous l'eau de plus de 0.2 %. Conséquence structurelle :

> **Le système ne dispose d'aucun mécanisme pour sortir d'une position perdante.** Le trailing stop ne sort que les gagnants (≥ 0.5 % de profit), le loss guard interdit de sortir les perdants (< -0.2 %). Toute position en drawdown est **verrouillée jusqu'à retour à l'équilibre** — comportement de type martingale, **fatal pour l'espérance de gain** et le drawdown maximal.

### 3.3 Autres défauts d'exécution

- **Pas de vérification de fill** : le prix enregistré est le prix **estimé** (yfinance, `get_real_price_eur` `:349`), pas le fill réel T212 (`_execute_buy_order` `:535` enregistre `current_price` avant confirmation). L'écart de slippage réel n'est jamais mesuré.
- **Prix source hétérogène** : `get_t212_price` ne fonctionne que sur positions ouvertes (`:329-346`) ; sinon repli yfinance (potentiellement USD/stale).
- **Ordres market uniquement** (`:532`, `:653`) → slippage non maîtrisé en marché volatil ; pas de STOP_LIMIT broker-side.
- **Rate limiting naïf** : *backoff* linéaire `(attempt+1)×2` sans lecture de `Retry-After` (`safe_request`, `:389-408`).

### 3.4 Coût par trade vs edge

`backtest_prod.py` applique `T212_FEE_RATE = 0.001` (0.1 %) à l'achat **et** à la vente (`:20`, `:122-123`, `:132-134`) → **0.2 % de frais aller-retour**. Le trailing stop exige `profit_margin > 0.5 %` (`t212_executor.py:448`) pour déclencher une vente « gagnante » → un trade « réussi » ne **nettoie que 0.3 %** avant slippage et spread. Marge si fine que **tout slippage ou erreur de prix (stale yfinance) la transforme en perte nette**.

---

## 4. Analyse de viabilité / rentabilité (verdict)

### 4.1 Bilan forces / faiblesses pour l'edge

**Forces** :
- Diversification multi-modèle (12 sources) et consensus anti-hallucination.
- *Risk management* centralisé (Kelly, métriques de risque) — conceptuellement présent.
- Données fondamentales réelles (EIA via `eia_client`), données alternatives (Hyperliquid).
- Deux modèles à fondement théorique solide (Grebenkov ARP, HMM).
- Défense JSON double-couche robuste (`AGENTS.md` §2.1, ADR-001).

**Faiblesses fatales à l'edge** (vérifiées) :
1. Consensus par confiance **non calibrée** (§2.2 A) → dilution de l'edge par les modèles à confiance saturée.
2. TensorTrade = **bruit** à 5 % de poids (§1 #3).
3. Sentiment = **vote HOLD constant** à 16 % de poids (§1 #7).
4. Biais HOLD silencieux via `failed:True` ignoré (§2.2 B).
5. Sizing **all-in** + risk manager ignoré (§3.1).
6. **Aucun repreneur de pertes** — positions perdantes verrouillées (§3.2).
6 biais. Seuils BUY/SELL asymétriques + `MIN_CONFIDENCE_FOR_SELL` 2× (§2.1, §2.2 C).

### 4.2 Rapport signal/bruit estimé

~5 modèles utiles sur 12, pondérés par une confiance non calibrée où le bruit (TensorTrade, Sentiment) et les votes morts (HOLD constants) pèsent ~20–30 % du poids total. **L'edge des bons modèles est dilué par le bruit agrégé** plutôt que concentré.

### 4.3 Coûts vs edge

Frais 0.2 % aller-retour (`backtest_prod.py:20`), trailing stop à 0.5 % de profit min (`t212_executor.py:448`), slippage market non maîtrisé, prix stale (yfinance). L'**espérance par trade doit dépasser ~0.3–0.5 % net** pour être positive — seuil que la dilution du consensus et l'absence de stop-loss rendent peu probable en l'état.

### 4.4 Verdict de viabilité

> **Non viable en l'état pour dégager un alpha net de frais.** Le pipeline est **architecturalement sain** (orchestration parallèle, défense JSON, données fondamentales) mais **économiquement immature** : l'edge potentiel des 5 bons modèles est détruit par (a) l'agrégation non calibrée, (b) le bruit pondéré, (c) le sizing all-in, et surtout (d) l'impossibilité structurelle de couper les pertes.
>
> **Viable après exécution de la roadmap P0** (§6), qui cible précisément ces biais structurels. Ce verdict structurel devra être **confirmé ou nuancé fin Juin 2026** dès que `logs_prod/` fournira le premier mois complet de données réelles (cf. §6.4).

---

## 5. Ajouts viables (spécifications implémentables)

Chaque ajout : fichier cible, complexité (S/M/L), risque, verdict. Évalués ci-dessous dans l'ordre de la roadmap.

| # | Ajout | Fichier cible | Complexité | Risque | Verdict |
|---|---|---|---|---|---|
| 1 | Consommer `failed:True` (exclure/penaliser modèles en échec + renormaliser poids) | `enhanced_decision_engine.py:619` (loop), `:506-526` (lecture) | **S** | Faible | **Viable — P0** |
| 2 | Recalibrer TimesFM (utiliser les **quantiles** `:127` + seuil `×√horizon` au lieu de vol. quotidienne `:100`) | `timesfm_model.py:127`,`:135`,`:100-105` | **M** | Moyen | **Viable — P0** |
| 3 | Sizing par volatilité (*vol-target*) : remplacer all-in par `capital × target_vol / asset_vol` | `t212_executor.py:515` + `advanced_risk_manager.py` ; câbler `recommended_size` via `execute_t212_trade` (`:668`) | **M** | Moyen | **Viable — P0** |
| 4 | Résoudre contradiction trailing/loss-guard : **ajouter un vrai stop-loss** (sortie perdante) et unifier la logique de stop | `t212_executor.py:423-580` | **S–M** | Moyen (risque de sur-trading) | **Viable — P0** |
| 5 | Corriger OilBench : passer `headlines` réelles + jitter déterministe (seed) + confiance continue (lissage au seuil) | `oil_bench_model.py:199-201`,`:186-189` + `enhanced_trading_example.py:627` | **S** | Faible | **Viable — P0** |
| 6 | (Optionnel) Retirer TensorTrade du consensus tant que non réentraîné | `config_weights.py:14` (poids→0) ou court-circuiter `enhanced_trading_example.py:513` | **S** | Faible | **Viable — P0** |
| 7 | Calibration Platt/isotonique des confidences sur échelle commune | `enhanced_decision_engine.py` (preprocessing des `decisions`) | **M** | Moyen | **Viable — P1** |
| 8 | Vérification de fill T212 (polling `orderId` après ordre market) | `t212_executor.py:535` | **M** | Faible | **Viable — P1** |
| 9 | Sentiment réel (LLM/lexique au lieu de la constante `0.1`) | `news_fetcher.py:100` | **M** | Moyen | **Viable — P1** |
| 10 | *Grid search* des poids via `backtest_prod.py` (optimisation offline) | `backtest_prod.py` + `config_weights.py` | **M** | Faible (offline) | **Viable — P1 — nécessite `logs_prod` d'abord** |
| 11 | Réentraînement TensorTrade complet (≥10⁵ timesteps) OU suppression définitive | `tensortrade_model.py:17-18` | **L** | Moyen | **Viable — P1 (décision tranchée)** |
| 12 | Stop-loss/take-profit broker-side (STOP_LIMIT T212) | `t212_executor.py` (nouveau type d'ordre) | **M** | À évaluer vs API T212 | **À valider — P2** |
| 13 | *Backtest* déterministe du consensus (replay signaux historiques hors-prod) | nouveau / `backtest_prod.py` | **L** | Faible (offline) | **Viable — P2** |

### Ajouts évalués comme non-viables ou prématurés

- **13ᵉ modèle LLM supplémentaire** : dilue davantage le consensus avant calibration ; non-viable tant que §5.7 n'est pas fait.
- **Passage GPU** : contredit la philosophie « CPU-only, cognitive prudence » ; le goulet est le calibration/sizing, pas le calcul.
- **Trading intraday** : incohérent avec un cycle de 15 min (`main.py`) et l'absence de stop-loss broker-side ; prématuré.

---

## 6. Roadmap P0 / P1 / P2 + plan de mesure

### 6.1 P0 — Bloque la rentabilité (à faire avant tout jugement définitif)

| Action | Réf. | Effet attendu |
|---|---|---|
| Consommer `failed:True` + renormaliser poids | `enhanced_decision_engine.py:506-526`,`:619` | Fin du biais HOLD par déflation du score |
| Recalibrer TimesFM (quantiles + `√horizon`) | `timesfm_model.py:127`,`:135`,`:100` | Confiance comparable + fin du sur-signalement |
| Sizing par volatilité (fin de l'all-in) | `t212_executor.py:515` + câblage `recommended_size` | Maîtrise du risque par trade |
| Ajouter un vrai stop-loss (sortie perdante) | `t212_executor.py:423-580` | Fin du verrouillage des pertes (martingale) |
| Corriger OilBench (headlines + jitter + confiance) | `oil_bench_model.py:199-201` + `enhanced_trading_example.py:627` | Modèle réellement informé par les news |
| Retirer TensorTrade du consensus (temporaire) | `config_weights.py:14` | Retrait du bruit pondéré |

### 6.2 P1 — Optimise l'edge

- Calibration Platt/isotonique des confidences (§5.7).
- Vérification de fill T212 via `orderId` (§5.8).
- Sentiment réel LLM/lexique (§5.9).
- *Grid search* des poids **après** accumulation de `logs_prod` (§5.10).
- Réentraînement TensorTrade ≥10⁵ timesteps **OU** suppression définitive (§5.11).

### 6.3 P2 — Maturité

- *Backtest* déterministe du consensus (§5.13).
- Stop-loss/take-profit broker-side STOP_LIMIT (§5.12).
- Hygiène : index DB, `DB_PATH` absolu, unification de `_THINKING_TOKENS` (cf. `AGENTS.md` §6).

### 6.4 Plan de mesure (indispensable — premier point de contrôle fin Juin 2026)

Aucun verdict de rentabilité n'est définitif sans *track record*. La phase de validation en cours (2026-05-29 → 2026-06-30) **produira le premier jeu de données réel fin Juin 2026** (~1 mois complet d'exécution). Mesures obligatoires :

0. **Réutiliser l'instrumentation existante** (ne pas reconstruire). Rappel : `logs_prod/` est une **copie manuelle** opérateur des artefacts de prod — ces outils tournent donc sur le *snapshot*, pas sur un flux auto :
   - *Suivi live (en prod)* : `analyze_trading_logs.py` sur `trading.log` (santé opérationnelle, slippage, déconnexions API) — inspection quotidienne pendant la phase de validation, avant toute copie.
   - *Métriques portefeuille (snapshot)* : `audit_portfolio_performance.py` sur `logs_prod/performance_monitor.db` (PnL, drawdown, alertes par ticker).
   - *Backtest rentabilité (snapshot)* : `backtest_prod.py` sur `logs_prod/trading_journal.csv` (replay signaux vs prix + frais T212 0.1 %).
1. **Point de contrôle n°1 — fin Juin 2026** : **copier manuellement** les artefacts de prod du mois écoulé dans `logs_prod/` (`trading.log`, `trading_journal.csv`, `trading_history.db`, `performance_monitor.db`, sorties morning brief), puis lancer `backtest_prod.py` + `audit_portfolio_performance.py` pour la **première mesure empirique**. Vérifier la complétude du logging (signal, confiance, prix estimé **et** prix fillé). À ce stade, ~1 mois de données permet déjà un premier *sanity check* (fréquence de trading, ratio win/loss brut, amplitude des signaux), mais **reste insuffisant** pour une statistique robuste.
2. **Accumulation jusqu'à ≥ 60 jours** : prolonger l'enregistrement au-delà de fin Juin pour atteindre un échantillon statistiquement significatif (objectif : 2 mois pleins), notamment avant tout *grid search* de poids (§5.10).
3. **Activation mensuelle de `backtest_prod.py`** dès que le mois de Juin est bouclé (`uv run backtest_prod.py`), puis chaque mois.
4. **Seuils de décision** pour la revue de validation : `Sharpe > 0.5`, `alpha > buy & hold`, drawdown max acceptable défini explicitement, et **win/loss ratio net de frais > 1**.
5. **Métriques par modèle** (via `adaptive_weight_manager.py` SQLite / `model_performance.db`) : précision directionnelle, calibration (Brier score), contribution à l'edge — pour alimenter la calibration P1 et le *grid search*.

> **Lacune outil identifiée** : aucun analyseur ne compare aujourd'hui le **prix estimé (yfinance)** au **prix fillé réel (T212)** — `backtest_prod.py` et l'exécuteur utilisent le prix estimé (cf. §3.3). Le point de contrôle n°1 devra ajouter cette réconciliation (relié à l'ajout §5.8 « vérification de fill »).

> **Révision du rapport** : ce document devra être **révisé après le point de contrôle fin Juin 2026** dès que les données réelles d'un mois complet seront disponibles — le verdict structurel du §4.4 sera alors confirmé, nuancé ou invalidé par les métriques empiriques.

---

## Annexe — Références `file:line` consolidées

| Constat | Référence |
|---|---|
| Score pondéré du consensus | `enhanced_decision_engine.py:619-623` |
| Seuils adaptatifs asymétriques | `enhanced_decision_engine.py:331-338` |
| `MIN_CONFIDENCE_FOR_SELL` 2× / `MIN_CONFIDENCE_FOR_ACTION` | `enhanced_decision_engine.py:302-303` (usage `:474`,`:479`) |
| Super-Consensus Boost | `enhanced_decision_engine.py:649-658` |
| Régime adjustments (boost TensorTrade en ranging) | `enhanced_decision_engine.py:343-346` |
| `failed:True` non consommé (avertissement docstring) | `llm_client.py:60-72` (docstring `:64-68`) |
| Text LLM `temperature=0.4` | `llm_client.py:203` |
| TimesFM confiance `×50` | `timesfm_model.py:135` |
| TimesFM quantiles ignorés | `timesfm_model.py:127` |
| TimesFM seuil vol. quotidienne vs horizon | `timesfm_model.py:100-105`,`:132` |
| TimesFM `_positions` non persisté | `timesfm_model.py:44` (filtre `:143-148`) |
| TensorTrade timesteps insuffisants | `tensortrade_model.py:17-18` (fine-tuning `:204`) |
| TensorTrade biais « rester investi » | `tensortrade_model.py:91` vs `:95` |
| OilBench `headlines=None` en prod | `enhanced_trading_example.py:627` |
| OilBench jitter `random()` | `oil_bench_model.py:199-201` |
| OilBench discontinuité de confiance | `oil_bench_model.py:186-189` vs `:201` |
| Sentiment AlphaEar = constante | `news_fetcher.py:100` |
| Sentiment Google News non agrégé | `news_fetcher.py:124`,`:150-155` |
| Sentiment plancher de confiance 0.5 | `sentiment_analysis.py:19` |
| Sizing all-in | `t212_executor.py:515` |
| Risk manager `recommended_size` ignoré | `enhanced_trading_example.py:656` (vs `t212_executor.py:668`) |
| Trailing stop (sort les gagnants) | `t212_executor.py:423-453` (seuil `:448`) |
| Loss guard (bloque les perdants) | `t212_executor.py:565-580` (seuil `:575`) |
| Budgets hardcodés 1000 € | `t212_executor.py:88-92` |
| Frais T212 0.1 % × 2 | `backtest_prod.py:20` (achat `:122`, vente `:132`) |
| Web Researcher `verbose=True` en prod | `web_researcher.py:275` |
| HMM main-implementé | `hmm_model.py:31`,`:76` (lookback `:136`) |
| Grebenkov RIE / ATR-adaptatif | `grebenkov_model.py:42`,`:186`,`:207` |
| Classic TimeSeriesSplit / cache MD5 | `classic_model.py:124`,`:180`,`:22` |
| Log live (RotatingFileHandler → `trading.log`) | `src/bootstrap.py:5-23` |
| Analyseur log live (health/slippage/disconnects) | `morning_brief/tools/analyze_trading_logs.py` |
| Audit portefeuille depuis `performance_monitor.db` | `morning_brief/tools/audit_portfolio_performance.py` |
| Archive `logs_prod/` (copie manuelle opérateur : journal + DB trades + DB perf) | `tests/check_db.py:30-31`, `backtest_prod.py:29`,`:237`,`:307` |

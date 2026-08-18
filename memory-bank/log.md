# Journal d'Exécution (Append-Only)

> **Rôle** — Journal chronologique **en mode ajout seul**. Rien ne doit y être effacé.
> Chaque événement majeur y est empilé. Une entrée est ajoutée au début et à la fin de chaque action.
> (Historique produit → voir `changelog.md`. Tableau de bord sprint → voir `progress.md`.)
> Voir `AGENTS.md §1` pour la discipline des 4 fichiers et le protocole de gestion d'erreurs.

---

## [2026-06-30] init | Initialisation du système de mémoire déterministe (4 fichiers)
- Mise en place de la discipline des 4 fichiers dans `memory-bank/` (voir `AGENTS.md §1`).
- Création de `feature_list.json` (22 fonctionnalités cartographiées, F-01 → F-22).
- Création de `contract.md` (critères d'acceptation automatisés + protocole d'évaluation).
- Création de ce `progress.md` (tableau de bord sprint : période de validation PROD).
- Création de ce `log.md` (journal d'exécution append-only).
- Renommage de l'ancien `progress.md` (167 lignes d'historique corrections 2025-2026) → `changelog.md`.
- Refonte/allègement de `AGENTS.md` (racine) avec intégration du Principe Fondamental + directives opérationnelles.

## [2026-06-30] gen | Synchronisation de l'état initial
- `feature_list.json` : F-19 (validation PROD) marqué `in_progress` ; F-20/F-21 marqués `pending`.
- `progress.md` : objectif sprint aligné sur la période de validation 2026-05-29 → 2026-06-30.

## [2026-07-03] fix | Refonte reset_for_fresh_test.py (PR #75/#76/#77)
- Wipe par pattern (extensions gitignored) au lieu d'une liste explicite fragile.
- Bug `trading_history.db` sauté quand `cache_moved=True` corrigé (cause de désync).
- Bug `_safe_to_wipe` bloquant les sous-dossiers WIPE_DIRS de dossiers keep corrigé.
- Fallback copy+truncate pour fichiers verrouillés Windows (scheduler.log).
- Ledger Gemini wipé par défaut (démo) ; flag `--keep-quota-ledger` pour PROD payante.

## [2026-07-09] fix | 4 bugs PROD : phantom trades, T212 precision, win_rate, EIA (PR #78/#79)
- **#1 CRITIQUE** : `write_db=not is_t212` — la simulation n'écrit plus en DB en mode T212.
- **#2** : table `TICKER_QUANTITY_PRECISION` (OD7Fd_EQ=2, SXRVd_EQ=4, fallback=2).
- **#3** : sentinelle `-1.0` cohérente pour win_rate "non calculable" ; 72 fausses alertes éliminées.
- **#4** : `audit_prod_logs.py` utilise la colonne `period` pour EIA (au lieu de `df.index` → 1970).
- Nouveau script `clean_phantom_trades.py` pour nettoyage ciblé post-fix.
- Validation PROD : 0 transaction fantôme, 0 fausse alerte, state synced broker, 0 nouvelle erreur.

## [2026-07-09] gen | Mise à jour doc (AGENTS.md §2.2/§6/§7, changelog.md, log.md)
- `AGENTS.md §2.2` : 3 nouveaux invariants (write_db, TICKER_QUANTITY_PRECISION, win_rate sentinel).
- `AGENTS.md §6.1/§6.2` : bugs July 2026 résolus documentés ; ADR-002 Suite renuméroté.
- `AGENTS.md §7` : ajout `reset_for_fresh_test.py` et `clean_phantom_trades.py`.
- `changelog.md §5` : 2 entrées corrections récentes (reset refonte + 4 bugs).
- Phase de validation longue durée lancée (jusque fin juillet) avant passage PROD.

## [2026-07-15] fix | Audit PROD 2026-07-15 : 3 bugs (risk manager, biais SELL, EIA)
- **Audit initial** : `uv run python audit_prod_logs.py` → verdict WARN. Cohérence DB/broker parfaite (bug phantom-trades confirmé résolu), fix win_rate/precision effectifs. Mais 3 anomalies de comportement :
  - `Risk_Level = VERY_HIGH` sur 100% des 294 cycles → neutralise SXRV.DE (147/147 Risk_Adjusted=HOLD).
  - 0 signal SELL sur 294 cycles malgré ~400 votes SELL individuels.
  - `eia_crude_imports.parquet` dégénéré (1 ligne @ 2026-04-01).
- **Causes racines confirmées par reproduction sur logs_prod/data_cache** :
  - #1 : `advanced_risk_manager.py` applique des seuils échelle volatilité (0.01–0.04) à un score composite 0–1 → tout score > 0.04 = VERY_HIGH. Liquidity_risk structurel ~0.74 (pattern_risk faux pour ETF).
  - #2 : abstention HOLD non renormalisée + vincent_ganne 100% BUY sur non-oil + seuil SELL -0.15 inatteignable (score bearish max -0.139).
  - #3 : `get_crude_imports` sans facets → payload 1 ligne ; TTL basé sur mtime masque le problème.
- **Action en cours** : 4 correctifs ciblés + test de régression PROD (plan approuvé, approche « ciblée et sûre »).

## [2026-07-15] fix | 4 correctifs implémentés et validés (96/96 tests OK)
- **#1 Risk manager** (`advanced_risk_manager.py:94-100`) : seuils rescalés échelle composite 0-1 (VERY_LOW 0.20 → VERY_HIGH inf), au lieu de l'échelle volatilité (0.01-0.04) qui forçait VERY_HIGH sur 100% des cycles. SXRV.DE (score ~0.42) sort de VERY_HIGH → ses BUY ne sont plus neutralisés.
- **#2 Liquidity risk** (`advanced_risk_manager.py:236-242`) : `pattern_risk = 1-|corr(volume,returns)|` plafonné à 0.5 et sous-pondéré (40% vs 60% pour volume_risk). Corrige l'inflation structurelle des ETF (~0.74 → ~0.50).
- **#3a Consensus renormalisé** (`enhanced_decision_engine.py:511-540`) : le score pondéré est normalisé sur le poids des modèles votants (non-HOLD), plus sur tous les modèles. Les abstentions ne diluent plus le score. **Quorum guard** : si < 25% du poids a voté, fallback sur score brut (évite qu'un modèle isolé déclenche un signal fort — 47/294 cycles PROD concernés).
- **#3b Seuil SELL** (`enhanced_decision_engine.py:344`) : -0.15 → -0.10 (rend SELL atteignable ; le cycle le plus bearish PROD passait de -0.139 à -0.32 avec la renormalisation).
- **#3c Vincent Ganne** (`enhanced_trading_example.py:651-662`) : désactivé sur les tickers non-oil (vote BUY/STRONG_BUY 147/147 sur SXRV.DE = bruit macro sans lien avec l'ETF equity). `effective_vg_indicators = None`.
- **#4 EIA crude_imports** (`eia_client.py:134-170`) : ajout `facets[process][]=MCO` + tri asc ; refus de cacher une payload < 3 lignes (empêche le mtime de masquer une dégénérescence).
- **Test de régression** `tests/test_prod_regression.py` (6 tests) : rejoue les vraies données logs_prod/ → assert SXRV.DE < 50% VERY_HIGH, SELL atteignable, EIA pas de cache dégénéré.
- **Validation** : 96/96 tests OK (suite mockée complète + régression). `test_bias_removal.py` corrigé par le quorum guard (1 SELL seul → HOLD, pas STRONG_SELL).
- **Rappel PROD** : supprimer `logs_prod/data_cache/eia/eia_crude_imports.parquet` après `git pull` pour forcer le re-fetch EIA.

## [2026-07-23] eval | ⛔ Kronos (shiyu-coder/Kronos) TESTÉ ET REJETÉ — NE PAS RÉINTÉGRER

> **⚠️ AVERTISSEMENT PERMANENT AUX AGENTS FUTURS** : l'intégration du modèle de fondation **Kronos** (`NeoQuasar/Kronos-base`, repo `shiyu-coder/Kronos`, AAAI 2026, licence MIT) a été **implémentée entièrement puis rejetée** après un backtest 10 ans sur les tickers PROD. **Ne pas réitérer cette expérience** sauf si les conditions de la section "Pour rouvrir" ci-dessous sont réunies.

### Contexte
Demande explicite d'intégrer Kronos comme 12ᵉ modèle de décision (module `KronosModel(BaseModel)`, vendor `vendor/kronos/`, poids `0.06`, ADR-004). Implémentation complète et techniquement fonctionnelle : 23 tests mockés + 52 tests de régression OK, inférence live validée (~5s/cycle sur CPU), bug `DatetimeIndex.dt` corrigé.

### Pourquoi rejeté — preuve par backtest 10 ans
Bench `tests/bench_kronos_10y.py` (créé puis supprimé) sur données réelles yfinance (5 ans dispo, ETF n'existent que depuis 2021), rebalance hebdomadaire, frais T212 0.1% :

| Stratégie | SXRV.DE | CRUDP.PA | (ETF haussiers : +100% / +143% sur la période) |
|---|---|---|---|
| Buy & Hold | **+99.7%** | **+142.9%** | référence |
| Kronos | +17.2% | **+0.0%** | loin du marché |

**Cause racine** : Kronos-base est pré-entraîné sur des **K-lines intraday 5-min de marchés émergents** (XSHG/A-shares). Appliqué en **daily sur des ETF européens**, il produit des signaux **massivement baissiers** (biais mean-reversion inadapté) :
- SXRV.DE : 688 STRONG_SELL / 420 HOLD / **seulement 80 signaux acheteurs** sur 1268 bars.
- CRUDP.PA : **0 signal d'achat** sur 1279 bars (1160 HOLD, 119 SELL) → 100% cash, 0% de rendement.
- Magnitudes prédites irréalistes (±27% sur 5 jours) → le modèle extrapole mal hors de son domaine d'entraînement.

### Décision
Branch `feat/kronos-decision-module` **supprimée**, tous les artefacts retirés (module, tests, vendor, setup, ADR-04). Retour propre à `main`.

### Pour rouvrir (conditions strictes)
Ne réintégrer Kronos **que si** l'une de ces conditions est démontrable :
1. **Fine-tuning** sur données daily ETF européens (le modèle pré-entraîné brut est inadapté, mais l'architecture pourrait l'être après réentraînement — coût important).
2. Usage sur tickers **intraday 5-min** proches du domaine d'entraînement (marchés émergents), pas sur nos ETF daily.
3. Test préalable d'un **autre checkpoint** (Kronos-small 24.7M, ou un futur modèle daily) avec le même bench 10 ans → ne réintégrer que si Kronos bat Buy&Hold ou au moins Hold-only sur les deux tickers.

**Leçon générale** : tout nouveau modèle de décision doit passer le **bench 10 ans vs Buy&Hold ET Hold-only** sur les tickers PROD avant d'être activé. Un poids conservateur + quorum guard limitent les dégâts mais ne transforment pas un signal non pertinent en valeur ajoutée.

## [2026-07-27] fix | Rampe douce win_rate remplaçant le gate dur 45% + reset complet PROD

### Contexte — audit logs_prod (snapshot 2026-07-27 11:09)
L'audit des derniers logs PROD a révélé que **8 modèles sur 10 étaient neutralisés** (`Poids forcé à 0.0`) par le gate dur `win_rate < 45%` (`adaptive_weight_manager.py:670`) :
- classic 32.9% · hmm 33.2% · llm_text 26.3% · llm_visual 28.2% · oil_bench 25.8% · sentiment 27.5% · tensortrade 34.6% · timesfm 32.4% → **tous bloqués**.
- Seuls survivants : vincent_ganne (60.4%), grebenkov (44.1%, juste sous le seuil), council.
- **Conséquence** : l'ensemble s'effondre à 3 votants, la diversité disparaît, et un BUY fragile passe sur CRUDP.PA (confiance 0.19, council SELL) via le `OIL SPECIAL RISK MODE`.

### Cause racine
Le `win_rate` (ADR-002) est une **précision directionnelle + dead-zone** (`HOLD_NEUTRAL_RETURN_THRESHOLD=0.005`) : un signal n'est compté correct que si la direction **et l'amplitude** dépassent la dead-zone. Sur des ETF européens faiblement volatils (std return ~2%, 26.6% des jours dans la dead-zone), cette métrique est **structurellement < 45%** — même pour des modèles directionnellement corrects. Re-construction depuis la DB PROD :
- tensortrade : 35.5% (gate) vs **51.4%** en précision directionnelle pure (delta +15.9%).
- oil_bench : 27.3% vs **40.1%** (+12.8%) · hmm : 34.1% vs 44.5% (+10.3%) · grebenkov : 45.3% vs 51.9%.
- llm_text (27.1%) et sentiment (28.3%) : réellement mauvais (delta ~0%).

Un **gate dur à 45% sur cette métrique** était donc la mauvaise forme : il tuait de bons modèles sur marchés peu volatils. Le gate binaire n'existait dans aucun code commité (note PR #86 : *"the 'Bloquage' log message exists in no committed code; PROD is running an uncommitted revision"*).

### Correctif — rampe douce (pénalité continue)
- `src/adaptive_weight_manager.py` : 2 constantes module `WIN_RATE_SOFT_FLOOR=0.25`, `WIN_RATE_SOFT_CEIL=0.50`. Le bloc gate dur `if win_rate < 0.45: weight = 0.0` devient un **facteur multiplicatif linéaire** dans la bande [FLOOR, CEIL] : `(wr − FLOOR)/(CEIL − FLOOR)`, clampé [0, 1]. Sous le FLOOR → 0 (vraiment mauvais tué) ; au-dessus du CEIL → 1.0 (pas de pénalité).
- **Effet attendu sur les win_rate PROD actuels** : vincent_ganne 60%→1.00, grebenkov 44%→0.77, tensortrade 35%→0.38, classic 33%→0.32, hmm 33%→0.33, timesfm 32%→0.30, llm_visual 28%→0.13, sentiment 28%→0.10, llm_text 26%→0.05, oil_bench 26%→0.03 → **~7 votants significatifs** au lieu de 3 ; les modèles réellement mauvais restent quasi-éteints.
- La renormalisation existante (`total_smoothed`) est conservée — elle s'applique après le facteur.
- **Non-redondant** avec `calculate_performance_score` (qui normalise et n'éteint jamais) : la rampe agit en Stage 3 post-smoothing, le score est un ranking relatif continu en Stage 1.

### Test de régression
`tests/test_adaptive_weight_manager.py::test_soft_winrate_ramp_preserves_diversity` : 3 modèles (strong 0.90 / weak 0.30 / garbage 0.10) → assert weak survie (poids > 0, régression vs gate dur), ranking strong ≥ weak > garbage préservé, garbage (sous FLOOR) = 0.0.

### Validation
36/36 tests OK (`tests/test_adaptive_weight_manager.py` + `test_enhanced_decision_engine.py` + `test_prod_regression.py`), dont le quorum guard (#86), le SELL reachable, l'EIA non-dégénéré, le risk manager. **0 régression.**

### Décision associée — reset complet PROD
Vu l'accumulation de bugs (gate non-commité, EIA dégénéré, modèles 404, win_rate pollué par l'historique), l'utilisateur a décidé un **reset complet** via le script existant `reset_for_fresh_test.py` (pas de nouveau script). Wipe total : models entraînés, caches prix, EIA, portfolio, win_rate DB, gemini quota (DEMO). Le prochain cycle retélécharge et réentraîne depuis zéro.

### Déploiement PROD (à exécuter par l'utilisateur)
Le `git pull` 2026-07-27 a réussi (fast-forward 32df1dd → 50906b6 — corrigeait déjà les modèles 404 + EIA #86). Après merge de ce fix :
1. `git pull` sur PROD (récupère la rampe douce).
2. `uv run python reset_for_fresh_test.py --dry-run` puis `--yes` (DEMO → wipe quota OK ; si PAID, passer `--keep-quota-ledger`).
3. Relancer le pipeline. Premier cycle plus lent (re-download/re-train).
4. Surveiller : `Réduction de ...` (INFO) au lieu de `🚨 Bloquage` ; ~7 votants ; diversité restaurée.

### Leçon générale
Un **gate dur sur une métrique à dead-zone** = effondrement d'ensemble sur marchés peu volatils. Toujours préférer une pénalité continue, et calibrer les seuils par rapport à la **distribution empirique** de la métrique, pas une intuition (45% = "bon" est arbitraire pour cette métrique). Documenté dans AGENTS.md §6.3.

## [2026-07-28] fix | Sur-correction ADR-002 : biais bearish (classic dead-band, prompt visual, pipeline sentiment)

### Contexte — audit logs_prod (1 jour, 30 cycles, 27-28 juillet)
Après le déploiement de la rampe douce win_rate (`f22bbfc`), 1 jour de PROD a révélé que **3 modèles n'émettaient jamais de BUY** : `classic` (0/30), `llm_visual` (0/30), `sentiment` (0/30, toujours HOLD). Le marché baissier de la période (-3.1% le 27, -0.6% le 28) justifie les SELL, mais le 0 BUY est **structurel** (sur un marché haussier, classic n'atteindrait toujours pas 0.58, llm_visual exigerait toujours un "textbook pattern"). Deux investigations root-cause (agents Explore) convergent : le correctif ADR-002 (juin 2026, anti-biais-bullish) a **sur-corrigé** en 3 endroits indépendants, créant un biais bearish symétrique.

### Correctif 1 — `src/classic_model.py` : dead-band + calibration
**Problème** : BUY nécessitait `proba(1) ≥ 0.58` (dead-band `CLASSIC_HOLD_MARGIN=0.08` autour de 0.5) + la calibration **isotonic** aplatit proba vers 0.5 sur un classifieur faible (F1≈0.32, recall≈0.25) → 0.58 inaccessible → 0 BUY / 30 cycles. SELL toujours à 0.650 (cap).
**Fix** : `CLASSIC_HOLD_MARGIN` 0.08 → **0.04** (BUY requiert proba ≥ 0.54, dead-band symétrique 16→8 points) + calibration **isotonic → sigmoid** (Platt scaling, aplatit moins sur classifieur faible). Cap `CLASSIC_CONFIDENCE_CAP=0.65` inchangé (garde le plafond anti-overconfidence ADR-002).

### Correctif 2 — `src/gemini_gateway.py` + `src/llm_client.py` : prompt visual symétrisé
**Problème** : le prompt `_CHART_ANALYST_PROMPT` forçait HOLD sur "ambiguous, mixed, or mostly sideways" (barre basse = presque tout) et exigeait "textbook unmistakable pattern + >0.7 confidence" pour BUY/SELL (barre très haute) + temperature 0.4 (conservateur) → 0 BUY / 30 cycles.
**Fix** : instructions symétrisées — BUY sur "recognizable uptrend (higher lows, breakout above resistance)", SELL sur "recognizable downtrend", HOLD "ONLY when genuinely directionless", "apply the SAME confidence standard to BUY and SELL". Temperature 0.4 → **0.6** (analyse_chart Gemini + Ollama fallback). Identique dans les 2 fichiers.

### Correctif 3 — `src/news_fetcher.py` : 3 bugs sentiment (sentiment_score toujours ~0)
**Bug A — rate-limit silencieux** (ligne 60) : sur `Information`, l'ancien code faisait `break` → retournait 0 sans signaler clairement. **Fix** : WARNING explicite + `continue` (préserve les headlines collectées des requêtes précédentes).
**Bug B — AlphaEar crash silencieux** (ligne 99) : `get_unified_trends()` retourne une STRING markdown, pas une liste de dicts ; itérer dessus avec `.get("title","")` → `AttributeError` → `except Exception` silencieux → AlphaEar n'a jamais contribué. **Fix** : appel `fetch_hot_news(source_id)` qui retourne `List[Dict]` avec clé `"title"` (confirmé dans `news_tools.py:46`), + logger les erreurs.
**Bug C — moyenne non-filtrée** (ligne 68) : l'ancienne boucle sommait `ticker_sentiment_score` de TOUS les tickers mentionnés dans chaque article (SPY, MSFT dans une story oil) → diluait le score vers ~0.10. **Fix** : filtrer les rows dont `ticker` matche le ticker cible. Si 0 match, retourner 0 (no signal) plutôt que la moyenne bruitée.
**Bonus** : branche morte supprimée (`elif gn_headlines` / `else` identiques).

### Test de régression
`tests/test_classic_model.py` : 2 nouveaux tests `test_buy_signal_reachable_on_bullish_features` (features bullish fortes → pred_int == 1) et `test_sell_signal_reachable_on_bearish_features` (features bearish fortes → pred_int == 0). Ce test aurait empêché le bug de passer inaperçu (il aurait échoué sur l'ancien isotonic+0.08). Les tests ADR-002 existants (`TestBaisRemovalInvariants`) pinnaient la *symétrie des thresholds* mais pas la *reachability des signaux* depuis les modèles calibrés/promptés — c'est exactement la faille.

### Validation
45/45 tests OK (`test_classic_model` + `test_llm_prompts` + `test_enhanced_decision_engine` + `test_prod_regression` + `test_adaptive_weight_manager`), dont le quorum guard, le SELL reachable, l'EIA non-dégénéré, la rampe win_rate, et les nouveaux tests BUY/SELL reachable. **0 régression.**

### Déploiement PROD (à exécuter par l'utilisateur)
1. `git pull` sur PROD.
2. **Pas de reset complet** — `model_performance.db` est valide (30 cycles, données réelles). Les win_rate se recalculeront avec les nouveaux comportements.
3. **Invalidation cache classic** : supprimer `data_cache/models/classic_*.pkl` sur PROD (le modèle sera retrained au prochain cycle avec la calibration sigmoid). Ne pas toucher aux autres caches.
4. Relancer. Surveiller : classic doit émettre quelques BUY ; llm_visual idem ; sentiment doit dépasser 0.15 sur certaines journées (quand l'API news renvoie du signal directionnel).

### Leçon générale
Un correctif anti-biais (ADR-002) peut créer un biais **symétrique** s'il sur-corrige. Toujours valider non seulement que l'ancien biais disparaît, mais que la **reachability** des signaux dans les deux directions est préservée — par un test de bout en bout sur fixtures bull ET bear, pas seulement par la symétrie des thresholds. Documenté dans AGENTS.md §6.4.

## [2026-07-31] fix | Audit PROD 31/07 — 3 causes racines (churn + EIA stale + sell-guard)
- **Audit** `logs_prod/` (snapshot 31/07 11:04, verdict WARN) → investigation profonde → 3 bugs distincts.
- **Bug 1 — Cache EIA `crude_imports` stale** (`src/eia_client.py`) : payload de 3 lignes s'arrêtant 2026-04-01 (4 mois stale) mais mtime fraîche → ne se rafraîchit jamais. Cause : guard `len >= 3` ne vérifie que le nombre de lignes, pas la fraîcheur du contenu (dernier `period`). Régression du fix §6.0/§6.3 sous une forme différente. Fix : second critère `MAX_CRUDE_IMPORTS_AGE_DAYS = 70`.
- **Bug 2 — Sell-loss guard neutralisé** (`src/t212_executor.py:801`, CAUSE RACINE du churn) : le guard lit `current_pos.get("averagePrice")` — champ ABSENT du payload T212 réel (le vrai champ est `averagePricePaid`, documenté absent dans `TRADING212_API_GUIDE.md:32` + `memory-bank/changelog.md:119`). Conséquence : `t212_buy_cost=0.0` → `reference_cost = state_buy_budget` (sous-estimé, prix Yahoo signal-time) → ventes en perte de -0.8% à -1% laissées passer. C'est ce qui a permis les 3 round-trips churnés du 30/07 sur CRUDP.PA. Les tests ne l'attrapaient pas : fixture `_pos` injectait `"averagePrice"` (champ fictif). Fix : helper `_get_avg_price()` priorisant `averagePricePaid`, appliqué aux 4 sites de lecture.
- **Bug 3 — Aucun anti-churn** : gap BUY→SELL = 1 cycle (~30 min) systématique, aucun min-holding nulle part. Fix : `_evaluate_min_holding` (4h, BUY→SELL uniquement, bypassé par hard-stop-loss pour préserver la protection du capital).
- **Tests** : extension `test_prod_regression.py` (Bug 1) + `test_stop_loss.py` (Bugs 2 & 3, fixtures corrigées).
- **Leçon générale** : un guard de cache basé sur la mtime + le nombre de lignes ne suffit pas — il faut valider la **fraîcheur du contenu** (valeur métier), car une source en amont peut renvoyer du stale qui passe les guards quantitatifs. De même, un guard de sécurité qui lit un champ API documenté absent est silencieusement neutralisé — toujours valider les guards contre le **vrai payload API**, pas un mock qui invente des champs.

### Résultat
- **43/43 tests verts** (`test_stop_loss` + `test_prod_regression` + `test_t212` + `test_eia_client`) ; **67/67 verts** sur la suite LLM élargie (vérification absence de régression transverse). 8 nouveaux tests de régression.
- **Fichiers modifiés** (4) : `src/eia_client.py`, `src/t212_executor.py`, `tests/test_stop_loss.py`, `tests/test_prod_regression.py`.
- **Déploiement PROD requis** : `git pull` + suppression de `data_cache/eia/eia_crude_imports.parquet` sur la **machine PROD** (le cache stale actuel resterait sinon — le guard fraîcheur ne s'applique qu'aux nouvelles écritures). Pas de reset DB.

## [2026-08-18] fix | Audit PROD 18/08 & Fix Morning Brief output auto-creation
- **Audit `logs_prod/` (2026-07-27 -> 2026-08-18, 588 cycles)** :
  - **Verdict global** : ✅ SAIN. 20 transactions réelles T212 confirmées, consensus équilibré (BUY, SELL, HOLD actifs).
  - **Weekend Council** : 100% opérationnel sans erreur, rapports générés et 11ème voix active.
  - **Cause racine des 22 échecs Morning Brief** : dans `morning_brief/morning_brief.py`, `logging.basicConfig` configurait un `FileHandler` pointant vers `morning_brief/output/morning_brief.log` AVANT que le répertoire `morning_brief/output` ne soit créé (`OUTPUT_DIR.mkdir()`). Sur une machine PROD ou un clone propre (où `output/` est gitignored), l'import/lancement échouait immédiatement avec `FileNotFoundError: [Errno 2] No such file or directory`.
- **Fixes appliqués** :
  1. `morning_brief/morning_brief.py` : définition de `OUTPUT_DIR` et création systématique avec `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` et `(OUTPUT_DIR / "tools").mkdir(parents=True, exist_ok=True)` avant `logging.basicConfig`.
  2. `morning_brief/tools/__init__.py` : `TOOLS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` garanti dans `save_tool_result`.
  3. `schedule.py` (`run_morning_brief`) : création préventive de `morning_brief/output` + fallback header si le markdown de base est manquant pour garantir que FinAcumen peut toujours écrire son rapport.
  4. `src/bootstrap.py` : création automatique du dossier parent pour tout `log_file` personnalisé avant `RotatingFileHandler`.
- **Validation** :
  - Test de régression dédié `tests/test_morning_brief_init.py` (suppression complète de `output/` puis import/init automatique) : PASS.
  - 67/67 tests unitaires : PASS.


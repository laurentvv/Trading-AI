# Audit indépendant de passage en PROD — Trading 212 (compte payant)

**Date :** 2026-08-19
**Auditeur :** ZCode (audit indépendant, généré sans avoir lu `docs/AUDIT_PROD_READINESS_TRADING212.md` ni aucun autre document d'audit préexistant du dépôt)
**Question posée :** le système est-il prêt à passer sur un compte Trading 212 réel (argent payant) ?
**Verdict court : ❌ NO-GO en l'état** — 7 bloquants critiques identifiés, performance réelle non démontrée. Détail et conditions de GO ci-dessous.

---

## 1. Méthode et périmètre

- Lecture intégrale des fichiers critiques (`main.py`, `src/t212_executor.py`, `src/enhanced_decision_engine.py`, `src/llm_client.py`, `src/data.py`, `src/features.py`, `src/advanced_risk_manager.py`, `src/performance_monitor.py`, `src/adaptive_weight_manager.py`, `src/database.py`, tous les sous-modèles, `schedule.py`).
- Analyse quantitative des **données de production réelles** : `trading_journal.csv` (648 décisions), `trading_history.db` (20 ordres réels), `model_performance.db` (5 516 prédictions), `t212_portfolio_state.json`.
- Analyse des **logs** : `trading.log` (racine + `logs_prod/`, ~12 MB cumulés, 27/07 → 19/08), `scheduler.log`, `weekend_council.log`, `analyse_morning.log`.
- Exécution de la suite de tests : **196/196 passés** (78,6 s, sans réseau).
- Vérification d'hygiène git/secrets (`git ls-files`, `.gitignore`).

Historique de production couvert : **24 jours (27/07 → 19/08/2026), 2 tickers (CRUDP.PA, SXRV.DE), compte T212 DEMO exclusivement** (aucune occurrence "LIVE" dans les logs ; 279 déclenchements du module d'ordres, 11 ordres placés, 20 transactions).

---

## 2. Synthèse exécutive

| Domaine | Note | Résumé |
|---|---|---|
| Exécution broker T212 | **D** | Pas de timeout ni idempotence sur le POST d'ordre ; pas de stop-loss côté broker ; fill jamais vérifié |
| Modèle de décision | **C−** | Bug d'unité de volatilité qui amortit le score en permanence ; seuils calibrés *sur* ce biais ; vote SELL fantôme du classic en échec |
| Sous-modèles | **C−** | TensorTrade entraîné à la volée, figé, partagé entre tickers ; TimesFM dégradation silencieuse ; 2 modèles morts sur 11 |
| Données & features | **D** | Données macro **synthétiques** (aléatoires) mises en cache sans TTL ; fallback cache de prix sans limite d'âge |
| Risque & sizing | **C+** | Sorties multiples et inconditionnelles (bon) mais pas de plafond portefeuille global ; bande de perte « invendable » −0,2 %/−10 % |
| Performance réelle observée | **F** | −0,25 € sur 2 000 € en 24 jours ; 9 round-trips (non significatif) ; aucun modèle > 40 % de précision directionnelle |
| Exploitation / scheduler | **C−** | Pas de verrou d'instance (cycles concurrents **observés**) ; aucune relance après crash |
| Tests | **B** | 196/196 verts, bonnes régressions PROD ; mais couche exécution broker non testée (rejets, fills) |
| Hygiène git/secrets | **A** | Aucun secret commité, artefacts runtime correctement ignorés |

**Verdict global : NO-GO.** L'architecture est au-dessus de la moyenne pour un projet personnel (défense en profondeur sur les sorties, isolation `write_db`, quorum de vote, fail-safe HOLD) et le runtime est stable (314/315 cycles réussis). Mais le chemin de l'argent réel contient des défauts qui n'ont pas d'impact en demo et en auraient en live (double ordre sur perte de réponse réseau, position sans protection broker si la machine s'éteint), le moteur de décision souffre d'un bug de calibration identifié comme CRITIQUE, et surtout **24 jours de production n'ont démontré aucun edge** (performance plate, confiance systématiquement < 50 %, 56,9 % de HOLD).

---

## 3. Constats détaillés

### 3.1 Exécution Trading 212 — le chemin de l'argent réel

#### 🔴 C1 (CRITIQUE) — POST d'ordre sans timeout + retry aveugle → risque de double achat
`safe_request` (`src/t212_executor.py:532-551`) n'a **pas de timeout par défaut** et est utilisé pour le POST d'ordre market **sans timeout explicite** (`src/t212_executor.py:836` pour le BUY, `:968` pour le SELL). Deux conséquences en compte réel :
1. Un ordre peut bloquer le cycle indéfiniment (pas de borne réseau).
2. Pire : sur `RequestException` (ex. réponse perdue après que le broker a **reçu et exécuté** l'ordre), la fonction **rejoue le même POST jusqu'à 3 fois** → le même signal peut générer 2 à 3 positions réelles. Il n'existe ni clé d'idempotence, ni vérification de position entre les retries. En demo le risque est financier nul ; en live c'est le scénario classique de double-achat.

#### 🔴 C2 (CRITIQUE) — Aucune protection attachée à l'ordre côté broker
`order_data = {"ticker": ..., "quantity": ...}` (`src/t212_executor.py:835`, `:967`) — ni `stopLoss`, ni `takeProfit`, ni ordre limite. **Toutes** les protections (hard stop −10 %, take-profit +8 %, trailing −3 %, time-stop 15 j) sont évaluées **par le logiciel**, uniquement pendant les heures de marché, uniquement tous les ~30 min, uniquement si la machine/scheduler tourne. Une position réelle reste **sans aucune protection** si : machine éteinte, scheduler mort (cf. I1), panne internet, week-end de gap. Pour un compte payant, c'est disqualifiant en l'état : le stop-loss doit vivre chez le broker.

#### 🔴 C3 (CRITIQUE) — Fill jamais vérifié + comptabilité locale fausse
À réception 200/201/202 (ordre *accepté*, pas nécessairement *exécuté*), l'état `active_position` est écrit immédiatement (`src/t212_executor.py:838-848`) et la DB enregistre une transaction avec le **prix Yahoo au moment du signal**, pas le prix de fill réel (reconnu dans les commentaires du code, `:178-183`). Preuve chiffrée en prod : **écart de ~9,9 €** entre la comptabilité broker (+9,45 € affichés) et le recalcul depuis les prix locaux (−0,42 €) sur 24 jours ; 7 ventes sur 9 loggées « +0.00€ » alors que le P&L réel allait de −2,91 € à +2,66 €. La colonne `T212_Capital` du journal n'est pas une courbe d'equity (valeur de position si ouvert, cash si flat) → faux « drawdown » de −71,6 %. En live, ne pas savoir combien on a réellement gagné/perdu est un bloqueur.

#### 🟠 M1 (MAJEUR) — Chemins de BDD relatifs au répertoire de travail
`DB_PATH = Path("trading_history.db")` (`src/database.py:9`, dupliqué en dur dans `src/performance_monitor.py:816`). Selon le CWD du planificateur, on lit/crée une base différente → historique vide, win rate sentinelle −1.0 permanent. (Mitigation partielle : l'acheteur vérifie la position **broker** avant tout BUY, ce qui évite le double-achat, mais la cohérence des états reste scindée.)

#### 🟡 Points positifs (vérifiés)
- Vente = `-quantityAvailableForTrading` lu chez le broker (`:967`) → **short involontaire impossible** ; SELL sans position = no-op.
- BUY bloqué si position réelle OU locale (`:758`) ; resynchronisation locale depuis le broker.
- Stratégie de sortie **inconditionnelle et hiérarchisée** évaluée avant la logique BUY/SELL (`:1044-1094`) : hard stop −10 % (double couche risk manager + exécuteur, bypass du garde de vente), take-profit +8 %, trailing −3 %, time-stop 15 j, anti-churn 4 h. C'est la partie la plus solide du code.
- Recalage du prix d'entrée sur `averagePricePaid` broker (`_validate_and_recalibrate_entry_price`), écriture atomique du state JSON.
- `T212_ENV` gouverne demo/live partout (`:261`, `:476`, `:1004`) ; le mode est loggé à chaque exécution.
- Gestion du rate limit T212 (retry 429 avec backoff).

### 3.2 Modèle de décision (fusion d'ensemble)

#### 🔴 C4 (CRITIQUE) — Bug d'unité de volatilité : seuils quotidiens vs vol annualisée
L'orchestrateur fournit `current_volatility = returns.std() * np.sqrt(252)` (vol **annualisée**, ~0,15–0,25 pour un ETF actions) — `src/enhanced_trading_example.py:639-640` — mais le moteur compare aux seuils calibrés pour une vol **quotidienne** : `VOLATILITY_HIGH_THRESHOLD = 0.04` (`src/enhanced_decision_engine.py`), donc `score *= 0.8` (`_adjust_for_market_regime`) est **activé en permanence** sur tout actif réel. Conséquences :
- Tous les seuils de décision (BUY +0.12, STRONG +0.35, SELL −0.10) sont effectivement décalés de +25 % en permanence.
- La correction empirique documentée dans le code (« sell loosened from -0.15 to -0.10, le cycle le plus baissier sur 294 lignes PROD n'a atteint que −0.139 ») a **calibré le seuil sur le symptôme du bug** au lieu de corriger la cause.
- Même bug dans `adaptive_weight_manager.detect_market_regime` (seuil 0.03) → régime « volatile/crisis » **permanent** → ajustements de poids appliqués en continu sur des bases fausses.
Le système vit ainsi structurellement dans un régime « haute volatilité » amorti, ce qui explique largement la prépondérance de HOLD (56,9 %) et de confiance faible observées en prod.

#### 🟠 M2 (MAJEUR) — Échec du modèle classic → vote SELL confiants 0.5 (et non HOLD)
En cas d'échec d'entraînement/prédiction, l'orchestrateur initialise `classic_pred, classic_conf = 0, 0.5` (`src/enhanced_trading_example.py:293, 296`), et la conversion `_convert_legacy_inputs` mappe tout ce qui n'est ni 1 ni 2 sur **SELL** (`src/enanced_decision_engine.py:613-616`). Un échec technique du modèle quantitatif injecte donc un **vote baissier fantôme** pondéré 0,13 dans le consensus. C'est le seul modèle dont l'échec n'est pas neutre.

#### 🟠 M3 (MAJEUR) — Boucle de dégradation silencieuse des poids adaptatifs
Un modèle en échec retourne `HOLD, 0.0` (exclu du vote, correct) **mais** est enregistré comme « prédiction HOLD » dans la base de performance ; dès que le marché bouge de > 0,5 %, ce HOLD technique est jugé incorrect → win rate dégradé → soft penalty (poids annulé sous 25 % de win rate) → le modèle échouant pèse de moins en moins, **sans aucune distinction entre panne technique et vrai signal**. Les logs montrent la sanction active : 39 occurrences de « Réduction forte de hmm_model/grebenkov/classic : win_rate sous le plancher 25 % ».

#### 🟠 M4 (MAJEUR) — Double comptage du Weekend Council
Le verdict du council est à la fois une **11e voix pondérée** (0.10) dans le vote ET **injecté dans le prompt du LLM texte** via `get_council_verdict_context()` (`src/llm_client.py:210-219`, appelé avant chaque décision). Le poids effectif du council est > 0.10 et non mesurable. Le morning brief (< 24 h) est aussi injecté comme « Extremely Important Context » → boucle de rétroaction du système sur lui-même.

#### 🟠 M5 (MAJEUR) — Surcharges empiriques empilées
Override pétrole : si `oil_bench` < 0.15, on force `oil_bench = llm_text = timesfm = 0.15` avant renormalisation (constantes magiques écrasant la pondération adaptative). Asymétrie BUY +0.12 / SELL −0.10 assumée mais fondée sur le biais C4. Règle prompt « extreme negative funding = contrarian BUY » sans symétrie vendeuse (`src/llm_client.py:311`). Sentiment structurellement incapable de SELL (score jamais observé < −0.15 ; 648/648 HOLD en prod — `src/sentiment_analysis.py`).

#### 🟡 Points positifs
- Vote pondéré normalisé sur les votants non-HOLD + **quorum** (< 25 % du poids ou < 3 votants → score capé, STRONG inatteignable) : bien conçu.
- Fail-safe : tous providers LLM morts → `HOLD 0.0` échoué, jamais dernière valeur ; aucun provider configuré → cycle annulé (`main.py:311-323`).
- Parsing JSON double défense (`json_mode` + extraction glissante + `_find_dict_with_keys`) réellement robuste.
- Aucune position détenue injectée dans les prompts LLM (pas d'ancrage).

### 3.3 Sous-modèles (11 voix)

| Modèle | Constat principal | Sévérité |
|---|---|---|
| classic (RF/GBM/LR) | Échec → SELL 0.5 (M2). Calibration avec CV temporel propre ; cible asymétrique « pas de hausse » ≠ prédiction de baisse, seuil réduit « for more BUY signals » (biais de conception documenté) | 🟠 |
| timesfm | `ppo_model.zip`— non : poids téléchargés HuggingFace ; si l'init échoue → HOLD 0.0 **pour toujours** (singleton, aucune alerte au-delà d'un warning). Filtre anti-re-achat alimenté par une position fantôme (`update_position` jamais appelé). Précision directionnelle prod : 39,1 % mais **0/28 sur ses BUY** | 🟠 |
| tensortrade (PPO) | **`ppo_model.zip` absent du disque** → entraînement from-scratch (2 000 steps) **à la volée dans le cycle de trading** ; un seul zip **partagé entre tous les tickers** (PPO entraîné sur CRUDP.PA sert à SXRV.DE) ; jamais réentraîné. Précision prod : 25,3 % | 🟠 |
| llm_text / llm_visual | Robustes (double défense JSON, timeouts 240/300 s). llm_visual le plus équilibré (248 BUY / 266 SELL / 134 HOLD) ; précision 33,6–36,2 % — pas mieux que le hasard sur 3 classes | 🟡 |
| sentiment | **100 % HOLD sur 648 décisions prod** — 0,16 de poids pour une voix qui ne vote jamais | 🟠 |
| vincent_ganne | Désactivé en dur (`effective_vg_indicators = None`) — N/A 648/648, 0,02 de poids mort | ℹ️ |
| oil_bench | HOLD avec `confidence = 0.3 + random()` — **bruit aléatoire non seedé** dans la confiance ; vote émis même si toutes les données EIA sont « N/A » | 🟠 |
| hmm / grebenkov | Look-ahead propre (sigma `shift(1)` côté grebenkov) ; biais structurel vendeur du HMM 2-états (tout état non haussier → SELL) ; précision prod 27,1 % / 22,3 % | 🟡 |
| council | Double comptage (M4) ; exécution hebdo propre (3 rapports générés, 0 erreur) | 🟠 |

**Bilan : sur 11 voix, 2 mortes (sentiment, vincent_ganne), 1 fantôme en échec (classic), 1 bruitée (oil_bench), 1 figée et partagée entre actifs (tensortrade).** La précision directionnelle observée en prod ne dépasse **40 % pour aucun modèle** (base 31,5 % de jours plats).

### 3.4 Données & features

#### 🔴 C5 (CRITIQUE) — Données macro **synthétiques** fabriquées et mises en cache sans TTL
`src/data.py` (« Method 4: Create realistic default data ») : si FRED/Yahoo/Alpha Vantage échouent, la fonction génère 24 mois de données **aléatoires** (`np.random.normal`, ±2 % autour d'une valeur par défaut, ex. Fed funds 5,25 %) et **les sauvegarde dans le cache Parquet** (`_save_macro_data_to_cache`). Le cache macro n'expire jamais → des taux/CPI **inventés** peuvent alimenter les features macro indéfiniment, avec un seul WARNING au premier échec. En live : décisions prises sur des chiffres fictifs possibles.

#### 🔴 C6 (CRITIQUE) — Fallback sur cache de prix sans aucune limite d'âge
Après 3 échecs yfinance, retour au cache Parquet **quelle que soit son ancienneté** (`src/data.py`, bloc « All download attempts failed, trying to use cached data ») : le contrôle de fraîcheur (1 jour) n'est appliqué qu'à l'entrée, pas au fallback. Ticker délisté ou Yahoo en panne une semaine → le pipeline calcule RSI/MACD/MA200 et décide sur des prix d'une semaine. (Le patch prix live T212 ne corrige que la dernière ligne de 3 tickers.)

#### 🟠 Autres (MAJEUR/MINEUR)
- Cache macro sans expiration du tout (CPI/Fed peuvent rester figés des mois) — MAJEUR.
- Appels réseau sans timeout : Alpha Vantage, FRED, Hyperliquid, SMTP (`src/data.py:411, 549, 650, 691` ; `src/performance_monitor.py:451`) — MAJEUR (cycle bloquable).
- Fuite macro en backtest : valeurs du jour `bfill`-ées sur tout l'historique d'entraînement (`src/features.py` + `select_features`) — en live les colonnes macro sont de facto constantes/désormais mortes, mais tout backtest les incluant est invalide — MAJEUR.
- `VIX` backfillé depuis le futur (`bfill`) en tête de série ; seuil de cible calculé sur l'échantillon complet ; pas de normalisation des features prix — MINEUR.
- Les features techniques elles-mêmes sont **propres** : rolling/ewm/`shift(1)` uniquement, cibles `shift(-N)`, `TimeSeriesSplit` partout — c'est le point le mieux traité du repo.

### 3.5 Risque & sizing

- **Pas de plafond d'exposition portefeuille global** : `max_portfolio_risk` est stocké puis jamais utilisé ; l'exposition réelle = nb tickers × 1 000 € sans contrôle agrégé ni corrélation — MAJEUR.
- **Bande de perte « invendable » entre −0,2 % et −10 %** : le garde de vente bloque toute vente en perte > 0,2 % (`_check_sell_loss_guard`) et l'inertie de sortie convertit les SELL peu confiants ; une position à −4 % ne peut sortir que par le hard stop (−10 %) ou le time-stop (15 j). C'est exactement ce qui a laissé CRUDP dériver à −17 % (incident documenté dans le code) — MAJEUR.
- Sentinelle `win_rate = −1.0` bien gérée par tous les consommateurs internes ; pas de division par zéro ; win rate calculé sur trades réellement fermés — ✔.
- Kelly : win rate ≤ 0 ou sentinelle −1.0 → fraction Kelly 0,1 (logique inversée, impact borné par le cap 1 %) — MINEUR.
- Sizing réel très conservateur en prod : `min(budget, cash broker) × 0,95 × 0,75` → **seuls ~28,5 % de chaque budget de 1 000 € sont déployés** — à connaître avant de passer live (le « 1 000 € par ticker » affiché ne correspond pas à l'exposition réelle).

### 3.6 Poids adaptatifs

- **Doublons d'observations** : pas de contrainte UNIQUE `(date, model_name)` dans `model_performance_history` ; avec un cycle ~30 min, `min_observations = 10` **lignes** est atteint en ~une journée → les poids s'adaptent au bruit d'un seul jour, chaque journée comptant N fois — MAJEUR.
- `max_drawdown` calculé sur une série temporelle **inversée** (`ORDER BY date DESC` puis cumprod) dans le chemin mono-modèle — score faux alimentant 15 % du poids de performance — MAJEUR.
- `return_1d` de la boucle de feedback T212 = rendement de la **durée de détention** (jusqu'à 15 j) étiqueté 1 jour — MINEUR/MAJEUR.
- Régime « crisis » permanent (conséquence du bug C4) — cf. 3.2.

### 3.7 Performance réelle observée (données prod recalculées)

- **PnL total reconstitué : −0,25 € sur 2 000 € en 24 jours (−0,013 %)** ; 9 round-trips fermés, win rate 44,4 %, PnL réalisé local −0,42 € ; détention typique 30–90 min. **Échantillon sans aucune significativité statistique.**
- Répartition des 648 décisions : HOLD 56,9 %, SELL 30,1 %, STRONG_SELL 6,2 %, STRONG_BUY 4,2 %, BUY 2,6 % → **biais vendeur 5,3:1** (héritage des corrections anti-biais-BUY successives).
- **Confiance moyenne 23,05 %, maximum observé 48,84 % — jamais ≥ 50 %** en 24 jours.
- Précision directionnelle (5 516 prédictions) : meilleur modèle 39,1 %, aucun > 40 %, base 31,5 % de jours plats ; rendement moyen J+1 après un BUY **négatif** pour timesfm (−2,33 %), tensortrade, hmm, grebenkov.
- **Depuis la migration du 18/08 : 51 décisions hybrides, 100 % HOLD, 0 ordre envoyé.** Le comportement post-migration en conditions réelles n'est pas démontré.
- `performance_monitor.db` décoratif : `portfolio_value` constant à 1 000 € sur 649 lignes, drawdown/sharpe/returns à 0 — le monitoring « temps réel » ne mesure rien.

### 3.8 Logs & stabilité runtime

- **314/315 cycles exécutés (99,7 %)**, 0 CRITICAL, 0 Traceback dans les logs de trading, durées 2–6 min, auto-récupération effective. Très bon.
- Morning Brief : **22 échecs consécutifs** (28/07→18/08) dans l'ancien environnement (chemin obsolète), corrigé le 19/08 — une défaillance muette de 3 semaines détectée seulement à la migration.
- Providers LLM très dégradés le 18/08 : Gemini 429 ×13, Cerebras 402 ×10, Groq 404 (modèle retiré) — les fallbacks ont tenu, mais la chaîne dépend massivement du gratuit.
- 1 rejet d'ordre réel : vente de quantité 0 → HTTP 400 « Quantity is missing » (31/07) — le cas dégénéré n'est pas gardé en amont.
- Bruit permanent : EIA `crude_imports` périmé refusé ~195 fois, requêtes HF non authentifiées, fontes manquantes.
- Sync T212 intermittent documenté dans les données : positions ouvertes non reflétées dans le capital pendant des heures (04/08→05/08).

### 3.9 Workflow / scheduler / exploitation

- 🔴 **I1 (CRITIQUE)** : aucun verrou d'instance — `start_scheduler.bat` et `schedule.py` ne posent pas de lock ; **des cycles concurrents ont déjà eu lieu** (3 lancements en 61 s le 18/08 avec « Code 120 » ; cycles doublés le 19/08 dans `scheduler.log`). Les verrous de `main.py` sont intra-processus seulement. Deux instances = double analyse, double state, risque d'ordres concurrents.
- 🔴 **I2 (CRITIQUE)** : aucune relance après crash — toute exception non `KeyboardInterrupt` tue le scheduler (`schedule.py:229`), le `.bat` se termine sur `pause`. Plus aucun cycle jusqu'à intervention humaine → positions réelles sans surveillance (cf. C2).
- 🟠 Fenêtre de rattrapage nulle : machine éteinte à 01:00 → Morning Brief perdu pour la journée ; Council perdu pour la semaine.
- 🟠 Council synchrone (timeout 48 h) peut bloquer la boucle du scheduler.
- 🟡 Bon : anti-chevauchement intra-processus, `CYCLE_TIMEOUT` 40 min avec `cancel_event` persistant qui empêche un thread orphelin de passer un ordre T212 après timeout (`main.py:442-476`) — défense bien pensée. `write_db = not is_t212` respecté (`main.py:349`) : seule l'exécution broker écrit en DB — invariant confirmé.

### 3.10 Tests & hygiène

- **196/196 tests passés** (79 s, mockés) ; excellentes régressions rejouant de vrais incidents PROD (stop-loss, précision quantité, cache EIA dégénéré, renormalisation du score).
- **Trous critiques** : aucun test du rejet d'ordre (branches `else` de `_execute_buy/_sell`), aucune vérification de fill après 2xx, ordres partiels non couverts, `test_full_cycle.py` invisible de pytest, zéro test du scheduler.
- Hygiène git **conforme** : aucun secret/DB/artefact tracké ; `.env`, `.env.t212`, `data_cache/`, `logs_prod/` ignorés et vérifiés par `git check-ignore`.

---

## 4. Registre consolidé des constats

| # | Sévérité | Constat | Localisation |
|---|---|---|---|
| C1 | CRITIQUE | POST d'ordre sans timeout + retry aveugle → double achat possible en live | `t212_executor.py:532-551, 836, 968` |
| C2 | CRITIQUE | Aucun stop-loss/take-profit côté broker ; protection seulement logicielle et cyclique | `t212_executor.py:835, 967` |
| C3 | CRITIQUE | Fill non vérifié ; DB enregistre le prix signal (pas le fill) ; P&L/capital affichés faux (écart 9,9 € mesuré) | `t212_executor.py:838-861` ; données prod |
| C4 | CRITIQUE | Bug d'unité de volatilité (annualisée vs quotidienne) → amortissement ×0.8 permanent, seuils calibrés sur le biais | `enhanced_trading_example.py:640` ; `enhanced_decision_engine.py` seuils 0.04 ; `adaptive_weight_manager.py` seuil 0.03 |
| C5 | CRITIQUE | Données macro synthétiques aléatoires persistées en cache sans TTL | `data.py` « Method 4 » |
| C6 | CRITIQUE | Fallback cache de prix sans limite d'âge → décision possible sur données périmées | `data.py` fallback post-échec |
| I1 | CRITIQUE | Pas de verrou d'instance scheduler ; cycles concurrents observés en prod | `schedule.py`, `start_scheduler.bat`, `scheduler.log` |
| I2 | CRITIQUE | Aucune relance après crash du scheduler | `schedule.py:229` |
| M1 | MAJEUR | Échec classic → vote SELL 0.5 fantôme | `enhanced_trading_example.py:293-296` + `enhanced_decision_engine.py:613-616` |
| M2 | MAJEUR | HOLD techniques enregistrés en perf → dégradation en boucle des poids | `enhanced_trading_example.py:747-756` |
| M3 | MAJEUR | TensorTrade : zip absent, entraîné in-cycle, partagé entre tickers, jamais réentraîné | `tensortrade_model.py` |
| M4 | MAJEUR | Double comptage du council (vote + prompt) | `llm_client.py:210-219` |
| M5 | MAJEUR | Pas de plafond d'exposition portefeuille global (`max_portfolio_risk` mort) | `advanced_risk_manager.py:74, 88` |
| M6 | MAJEUR | Bande de perte invendable −0,2 %/−10 % | `t212_executor.py` garde + inertie |
| M7 | MAJEUR | Poids adaptatifs : doublons d'observations, seuil 10 lignes ≈ 1 jour, drawdown sur série inversée | `adaptive_weight_manager.py:185-198, 330-379` |
| M8 | MAJEUR | Timeouts réseau manquants (AV, FRED, Hyperliquid, SMTP) | `data.py:411, 549, 650, 691` |
| M9 | MAJEUR | Chemins BDD relatifs au CWD (state dupliqué possible) | `database.py:9`, `performance_monitor.py:816` |
| M10 | MAJEUR | Fuite macro en backtest (bfill de la valeur du jour) ; cache macro sans expiration | `features.py` + `data.py` |
| M11 | MAJEUR | Performance non démontrée : 9 round-trips, −0,25 €/24 j, aucun modèle > 40 % précision, confiance < 50 % systématique | Données prod |
| m1–m8 | MINEUR | oil_bench confiance aléatoire ; sentiment plancher 0.5 ; news sans filtre de fraîcheur ; RSS « +commodity » codé en dur ; `except: pass` bootstrap ; SMTP sans timeout ; sentinelle −1.0 mêlée aux vraies valeurs ; schéma DB sans unicité | voir §3 |

---

## 5. Conditions de passage en PROD (GO-gates)

**Bloquants absolus (must-fix avant tout dépôt réel) :**
1. Timeout + idempotence sur le POST d'ordre : timeout explicite, PAS de retry automatique d'un POST d'ordre sans re-vérifier la position broker entre les tentatives (C1).
2. Attacher `stopLoss` (et idéalement `takeProfit`) à chaque ordre market côté T212 (C2).
3. Confirmer le fill (re-poll position/commande) avant d'écrire état + DB ; enregistrer le prix de fill réel, pas le prix signal (C3).
4. Corriger le bug d'unité de volatilité puis **recalibrer tous les seuils** (C4) — sans quoi toute la calibration actuelle reste fondée sur un artefact.
5. Supprimer les données macro synthétiques + TTL sur tous les caches ; refuser de trader sur un cache de prix trop vieux (C5, C6).
6. Verrou d'instance scheduler + supervision/relance automatique (watchdog ou tâche planifiée Windows) (I1, I2).
7. **Preuve d'edge en demo** : au minimum 60 à 90 jours de demo consécutifs avec P&L net positif après frais simulés, win rate et drawdown suivis sur une vraie courbe d'equity, et la chaîne de reporting corrigée (P&L des ventes, equity, performance_monitor fonctionnel). Aujourd'hui 24 jours plats ne démontrent rien (M11).

**Fortement recommandés avant mise réelle :**
- Corriger l'échec classic → HOLD neutre (M1) et distinguer HOLD technique vs HOLD de modèle dans la base perf (M2).
- Cycle de vie du PPO TensorTrade (par ticker, réentraînement planifié) ou désactivation pure et simple (M3).
- Dé-doublonner les observations de poids adaptatifs (UNIQUE date+modèle, seuil en jours) et corriger le drawdown inversé (M7).
- Plafond d'exposition global + revoir la bande −0,2 %/−10 % (M5, M6).
- Chemins de BDD absolus (M9), timeouts réseau généralisés (M8).
- Basculer `T212_ENV=live` dans `.env.t212` **seulement** après tout ce qui précède, avec un budget initial volontairement limité (ex. 200–500 €) et une surveillance quotidienne humaine les premières semaines.

---

## 6. Conclusion

Le projet est un système personnel sérieux : défense en profondeur sur les sorties, isolation DB en mode broker, quorum de vote, tests de régression sur incidents réels, hygiène git irréprochable, et une stabilité runtime de 99,7 %. Mais **l'audit indépendant conclut au NO-GO pour un compte payant** : le chemin d'exécution contient quatre défauts inacceptables avec de l'argent réel (double ordre possible, absence de stop broker, fills non confirmés, comptabilité fausse), le moteur de décision repose sur un bug de calibration non identifié qui a contaminé tous les seuils, la couche données peut fabriquer des macro-données fictives, et — point décisif — **la demo de 24 jours n'a démontré aucun avantage statistique** (PnL ~0, confiance jamais ≥ 50 %, aucun modèle au-dessus de 40 % de précision, 100 % HOLD depuis la dernière migration). Passer en payant maintenant reviendrait à payer pour valider un système dont on ne sait pas s'il gagne, avec des risques opérationnels connus et non corrigés.

*Audit généré le 2026-08-19 — lecture croisée code + données + logs + tests, sans consultation des documents d'audit préexistants du dépôt.*

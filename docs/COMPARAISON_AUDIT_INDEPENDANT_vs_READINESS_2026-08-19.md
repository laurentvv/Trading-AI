# Confrontation des deux audits PROD — Audit indépendant vs `AUDIT_PROD_READINESS_TRADING212.md`

**Date :** 2026-08-19
**Documents comparés :**
- **[A]** `docs/AUDIT_PROD_READINESS_TRADING212.md` (rapport préexistant, même jour) — verdict : 🟢 **GO conditionnel**, piliers notés 9,0 à 10/10.
- **[B]** `docs/AUDIT_PROD_INDEPENDANT_2026-08-19.md` (audit indépendant, généré sans avoir lu [A]) — verdict : ❌ **NO-GO**, 7 bloquants critiques.

L'audit [B] a été produit intégralement avant la lecture de [A] ; la confrontation ci-dessous est donc exempte de contamination dans les deux sens.

---

## 1. Le verdict en une ligne

| | [A] Rapport existant | [B] Audit indépendant |
|---|---|---|
| **Décision compte payant T212** | « PASSAGE EN PRODUCTION APPROUVÉ (GO CONDITIONNEL) » | « NO-GO en l'état — 7 bloquants critiques, edge non démontré » |
| Note Exécution broker | 9,5/10 | D |
| Note Modèle de décision | 9,5/10 | C− |
| Note Risque/Capital | 10/10 | C+ |
| Note Données/Résilience | 9,5/10 | D |
| Lecture de la démo (−0,42 € / 24 j) | « PnL démo stabilisé », « quasi à l'équilibre », « parfaitement contenu par les seuils de risque » | Aucun edge démontré ; 9 round-trips statistiquement insignifiants ; confiance jamais ≥ 50 % ; 100 % HOLD depuis le 18/08 |

**Deux audits, mêmes chiffres de base, verdicts opposés.** L'explication tient à la méthode : [A] est un audit de **conformité** (vérifie que les mécanismes documentés existent et que les invariants tiennent — ce qui est vrai), [B] est un audit de **détection de défauts** (cherche ce qui casse en conditions dégradées et ce que disent réellement les données). Les deux sont nécessaires ; seul le second répond à la question « puis-je brancher de l'argent réel ? ».

---

## 2. Convergences — ce que les deux audits confirment

Les faits suivants sont **identiques dans les deux rapports** et peuvent être tenus pour acquis :

- **Chiffres broker** : 20 transactions confirmées (11 BUY / 9 SELL), 9 round-trips, win rate 44,4 %, PnL réalisé −0,42 €, pire trade −1,02 %, meilleur +0,93 %, 0 transaction fantôme.
- **Hiérarchie de sortie** : hard stop −10 % (bypass du garde), take-profit +8 %, trailing stop −3 %, time-stop 15 j, anti-churn 4 h — présente, inconditionnelle, évaluée avant la logique BUY/SELL. C'est la partie la plus solide du code, les deux audits sont d'accord.
- **Invariants vérifiés** : `write_db = not is_t212` (seul l'exécuteur broker écrit en DB), sentinelle win rate −1.0, budgets 1 000 €/ticker, `CYCLE_TIMEOUT` 40 min + verrous par ticker + `cancel_event` anti-thread-orphelin, écriture atomique du state JSON.
- **Précision décimale** : table `TICKER_QUANTITY_PRECISION` (SXRVd_EQ 4 déc., OD7Fd_EQ 2 déc., fallback 2) et mapping EUR correct.
- **Bascule demo/live** : résolue par `T212_ENV` à chaque appel, aucun hardcoding ; priorité `averagePricePaid` comme vérité broker.
- **Mécanique du consensus** : normalisation sur les votants non-HOLD, double quorum (25 % du poids + 3 votants pour STRONG), rampe douce win-rate 25→50 %.
- Tests au vert (193 pour [A], 196 pour [B] — même suite, comptes légèrement différents selon la date d'exécution).

**Sur tous ces points, [A] est factuel et correct.**

---

## 3. Contradictions factuelles — où [A] est inexact ou trompeur

| # | Affirmation de [A] | Ce que montre [B] (preuves) | Statut |
|---|---|---|---|
| F1 | « écriture DB conditionnée aux **fills confirmés** » et « synchro temps réel sur averagePricePaid » pour les transactions | La DB est écrite sur **réponse 2xx (ordre accepté)**, sans re-vérification du fill ; elle enregistre le **prix Yahoo au moment du signal**, pas le prix d'exécution — reconnu dans les commentaires du code lui-même (incident de juillet : DB 10,876 vs fill réel 12,4469) ; écart comptable mesuré de **~9,9 €** sur 24 jours entre broker et comptabilité locale | **Faux** — c'est l'erreur la plus grave du rapport [A], car elle porte exactement sur le risque de compte réel |
| F2 | Position OD7Fd_EQ : « **gain latent +2,19 € (+0,77 %)** » | Instantané intradé favorable : le même jour, le sync T212 passe de **+2,12 € (14h10) à −1,81 € (−0,63 %) à 14h42** (`trading.log`). Présenter un point intraday comme « l'état du portefeuille » sans horodatage n'est pas une mesure | **Trompeur** (non reproductible) |
| F3 | « Utilisation **systématique** de schémas JSON typés (`SCHEMA_TRADING_DECISION`, …) » | Ces constantes sont du **code mort** : jamais passées à la gateway, seuls les tests les référencent. La défense réelle est `json_mode` + extraction glissante + `_find_dict_with_keys` (qui, elle, fonctionne) | **Inexact** |
| F4 | Répartition des poids présentée comme preuve de « Performance & Edge » : « Weekend Council 20,4 %, **Vincent Ganne 17,4 %**, Sentiment 15,8 %… » | `vincent_ganne` est **désactivé en dur** (`effective_vg_indicators = None`, `enhanced_trading_example.py:669`) : N/A sur 648/648 décisions ; `sentiment` = **100 % HOLD** sur 648/648. Deux des trois « piliers » les mieux notés **ne produisent aucun signal** ; leurs poids élevés viennent d'une boucle de notation polluée par les HOLD techniques et les doublons d'observations | **Trompeur** — la pondération est présentée comme une performance alors qu'elle récompense en partie des modèles inertes |
| F5 | « **Crashs ou boucles infinies : 0** » | 1 crash de cycle le 18/08 (« Code 120 », `[Errno 22] Invalid argument`) ; **22 échecs consécutifs** du Morning Brief (28/07→18/08) ; cycles scheduler **concurrents** observés (3 lancements en 61 s le 18/08, cycles doublés le 19/08) ; providers LLM massivement dégradés le 18/08 (429/402/404) | **Inexact** (vrai pour le cœur de trading, faux pour le système global) |
| F6 | « Perte maximale par trade : −1,02 % (**parfaitement contenue par les seuils de risque**) » | Le plafond structurel n'est pas −1 % : la bande −0,2 %/−10 % (garde de vente + inertie) **autorise une position à dériver jusqu'à −10 %** avant sortie forcée — c'est précisément ce qui a produit le −17 % CRUDP documenté dans le code. Le −1,02 % observé est un artefact d'échantillon, pas une propriété du système | **Sur-interprété** |

---

## 4. Bloquants critiques absents du rapport [A]

Aucun des sept bloquants de [B] n'apparaît dans [A] — ni en désaccord, ni en mention. C'est l'écart le plus important entre les deux documents :

1. **POST d'ordre sans timeout + retry aveugle** (`t212_executor.py:532-551, 836, 968`) → double achat possible en live sur perte de réponse réseau. [A] présente au contraire le « retry exponentiel » comme un point fort — sans noter qu'il s'applique aussi aux **ordres**, sans idempotence.
2. **Aucun stop-loss/take-profit attaché à l'ordre chez le broker** (`order_data = {ticker, quantity}`) → une position réelle est sans protection dès que la machine/scheduler s'arrête. Les « 5 niveaux de protection » de [A] sont tous logiciels et cycliques ; en compte réel, le week-end ou une coupure courant les neutralisent.
3. **Fill jamais vérifié** (lié à F1).
4. **Bug d'unité de volatilité** (vol annualisée comparée à des seuils quotidiens) → amortissement ×0,8 permanent du score ; les seuils SELL actuels (−0,10) ont été calibrés *sur* ce biais. [A] note le moteur 9,5/10 sans le détecter.
5. **Données macro synthétiques (aléatoires) mises en cache sans TTL** + **fallback cache de prix sans limite d'âge** dans `data.py`. [A] valorise au contraire le « cache 24h Parquet » — vrai pour le chemin nominal, faux pour les chemins de repli.
6. **Pas de verrou d'instance scheduler** (cycles concurrents *déjà observés*) et **aucune relance après crash** — d'autant plus critique combiné au point 2. [A] recommande « lancer `uv run schedule.py` » sans mentionner ces deux faiblesses.
7. **Edge non démontré** : 24 jours de démo ne valident pas une stratégie (échantillon nul, confiance < 50 % systématique, précision directionnelle ≤ 40 % pour tous les modèles, 100 % HOLD post-migration).

S'y ajoutent les majeurs invisibles de [A] : échec du classic → vote **SELL fantôme** à 0,5 de confiance ; boucle de dégradation des poids via HOLD techniques ; TensorTrade **sans fichier modèle** (PPO entraîné à la volée, un seul zip partagé entre tickers, jamais réentraîné) ; double comptage du council (vote + injection prompt) ; doublons d'observations et drawdown calculé sur série inversée dans le weight manager ; pas de plafond d'exposition global ; timeouts réseau manquants ; fuite macro en backtest.

---

## 5. Divergences d'interprétation (mêmes faits, lectures opposées)

| Fait | Lecture [A] | Lecture [B] |
|---|---|---|
| PnL −0,42 € / 2 000 € sur 24 jours | « Quasi à l'équilibre », démo « stabilisée » → argument **pour** le GO | Absence d'edge démontrée ; sur 9 trades, l'intervalle de confiance du win rate (44 %) va grosso modo de 15 % à 75 % → argument **contre** un GO immédiat |
| 56,9 % de HOLD, confiance moyenne 23 % | Non mentionné | Signe que le système n'a presque jamais de conviction exploitable (conséquence en partie du bug de volatilité) |
| Biais SELL 5,3:1 | Non mentionné | Héritage des corrections anti-biais-BUY empilées sur un biais de cause non corrigé |
| Tests 100 % verts | Preuve de fiabilité (⇒ notes 9-10) | Vraie et méritoire, mais la couverture s'arrête à la couche décision : **aucun test** du rejet d'ordre, du fill, des ordres partiels, ni du scheduler |
| « 636 cycles d'évaluation en continu » | Preuve de robustesse | 314/315 cycles réussis, oui — mais aussi 3 semaines de Morning Brief muet sans détection, et 0 ordre depuis la migration du 18/08 |

---

## 6. Ce que [A] apporte et que [B] confirme avec plaisir

Par équité : la partie « plan de migration » de [A] (création de clé API live avec permissions minimales, `T212_ENV=live`, budget initial réduit 500-1 000 €/ticker, procédure de kill-switch) est **concrète et correcte** — [B] rejoint ces recommandations opérationnelles (budget limité, supervision humaine au démarrage) à la différence près que [B] les place **après** la correction des bloquants, pas avant. Le kill-switch à 3 étapes (scheduler → positions → révocation clé API) est une bonne procédure que [B] n'avait pas détaillée.

---

## 7. Diagnostic méthodologique de l'écart

1. **[A] valide le chemin nominal, [B] attaque les chemins d'échec.** Toutes les affirmations de [A] sur les invariants sont vraies *quand tout se passe bien* ; aucune ne teste « et si la réponse réseau se perd après que l'ordre est passé ? », « et si le scheduler meurt vendredi soir ? », « et si Yahoo est en panne 3 jours ? ».
2. **[A] ne reproduit pas tous ses chiffres.** Le +2,19 € latent, les 17,4 % de Vincent Ganne ou les 193 tests ne sont pas horodatés/reproductibles ; [B] a retrouvé des valeurs différentes au même endroit, le même jour.
3. **[A] note la conformité, pas le risque.** Un système peut être conforme à sa documentation et impropre à l'argent réel ; les notes 9,5-10/10 uniformes signalent un biais de confirmation (aucun pilier en dessous de 9, aucune discovering de défaut en ~270 lignes).
4. **[B] pèse la preuve statistique.** 24 jours et 9 trades ne peuvent pas valider une stratégie, quel que soit le nombre de tests verts — c'est une limite d'échantillon, pas de code.

---

## 8. Recommandation finale consolidée

**Ne pas passer en compte payant maintenant.** La confrontation des deux audits renforce plutôt le NO-GO de [B] :

- les points que [A] met en avant (invariants, sorties, mapping, isolation DB) sont réels **mais sont des conditions nécessaires, pas suffisantes** ;
- les quatre défauts du chemin d'exécution (C1-C3 + absence de stop broker) sont incompatibles avec de l'argent réel, quel que soit le reste ;
- l'edge n'est pas démontré, donc il n'y a **rien à gagner** à payer pour valider maintenant, et du capital à perdre.

Traiter en priorité les 7 GO-gates de [B] (§5 de l'audit indépendant), **puis** réutiliser le protocole de migration de [A] (étapes 1-5, budget limité, kill-switch), qui est bien écrit pour le jour où les bloquants seront levés. Un critère simple avant de relire le dossier : **30 à 60 jours supplémentaires de démo avec equity réelle suivie, P&L net positif après frais, et les 7 bloquants fermés** — à ce moment-là seulement, la question du GO redevient ouverte.

---

*Document généré le 2026-08-19 — confrontation effectuée après finalisation de l'audit indépendant `AUDIT_PROD_INDEPENDANT_2026-08-19.md`.*

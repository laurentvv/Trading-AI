<p align="center">
  <a href="README.md">English</a> |
  <a href="i18n/README_zh.md">中文</a> |
  <a href="i18n/README_hi.md">हिंदी</a> |
  <a href="i18n/README_es.md">Español</a> |
  <a href="i18n/README_fr.md">Français</a> |
  <a href="i18n/README_ar.md">العربية</a> |
  <a href="i18n/README_bn.md">বাংলা</a> |
  <a href="i18n/README_ru.md">Русский</a> |
  <a href="i18n/README_pt.md">Português</a> |
  <a href="i18n/README_id.md">Bahasa Indonesia</a>
</p>

<p align="center">
  <img src="assets/banner.png" alt="Bannière Trading IA Hybride" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Système de Trading IA Hybride 📈</h1>
  <p>
    Un système expert d'aide à la décision pour le trading des ETF NASDAQ et Pétrole (WTI), tirant parti d'une intelligence artificielle hybride à 12 modèles pour des signaux de trading robustes et nuancés.
  </p>
</div>

<div align="center">

[![Statut du projet](https://img.shields.io/badge/status-en--développement-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Version Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Licence](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Table des Matières

- [🌟 À propos du projet](#-à-propos-du-projet)
  - [🚀 Stratégie Dual-Ticker (Analyse vs. Trading)](#-stratégie-dual-ticker-analyse-vs-trading)
  - [🧠 Moteur d'IA Hybride](#-moteur-dia-hybride)
  - [🧘 Philosophie de Décision : "Prudence Cognitive"](#-philosophie-de-décision--prudence-cognitive)
  - [✨ Fonctionnalités clés](#-fonctionnalités-clés)
  - [💻 Stack technologique](#-stack-technologique)
  - [⚙️ Performances & Matériel](#️-performances--matériel)
  - [🧠 Architecture IA & LLM (Gemini + Fallback Local)](#-architecture-ia--llm-gemini--fallback-local)
  - [🧠 FinAcumen (Mémoire Financière)](#-finacumen-mémoire-financière)
- [📂 Structure du projet](#-structure-du-projet)
- [🚀 Démarrage rapide](#-démarrage-rapide)
  - [✅ Prérequis](#-prérequis)
  - [⚙️ Installation](#️-installation)
- [🛠️ Utilisation](#️-utilisation)
  - [Mode Simulation (Paper Trading)](#mode-simulation-paper-trading)
  - [Exécution Réelle (Trading 212)](#exécution-réelle-trading-212)
- [🧪 Backtesting en Production](#-backtesting-en-production)
  - [Fonctionnalités](#fonctionnalités)
  - [Utilisation](#utilisation)
- [🤝 Contribuer](#-contribuer)
- [📜 Licence](#-licence)
- [📧 Contact](#-contact)

---

## 🌟 À propos du projet

Un système expert d'aide à la décision pour le trading des ETF NASDAQ et Pétrole (WTI), exploitant une intelligence artificielle hybride à 12 modèles.

### 🚀 Stratégie Dual-Ticker (Analyse vs. Trading)

Le système **analyse un indice global** (ex. `^NDX` pour le Nasdaq-100, `CL=F` pour le WTI) mais **exécute sur un ETF coté en EUR** (ex. `SXRV.DE`, `CRUDP.PA`). Cette dissociation assure une analyse sur des données de haute fidélité et une exécution réelle sur des actifs accessibles via Trading 212.

### 🧠 Moteur d'IA Hybride

Le moteur combine des modèles hétérogènes en un **consensus pondéré** :

1. **Modèles Scikit-Learn** (RandomForest, GradientBoosting, LogisticRegression) — validés par `TimeSeriesSplit` pour éviter la fuite de données. Signal quantitatif agressif (25 % du poids cognitif).
2. **TimesFM 2.5** (Google Research) — modèle de fondation pour le forecast de séries temporelles.
3. **TensorTrade / PPO** (stable-baselines3) — agent de Reinforcement Learning dans un environnement Gymnasium custom.
4. **Gemma 4 12B** (Ollama) — analyse **textuelle** (macro/news) et **visuelle** (charts techniques) ; la **défense JSON bi-couche** garantit un JSON propre malgré le mode pensée `<|think|>` actif.
5. **Analyse de Sentiment** hybride (Alpha Vantage + AlphaEar + Hyperliquid).
6. **Vincent Ganne Model** — verrou géopolitique (WTI, Brent, Gaz, Urée, DXY) générant des signaux BUY uniquement pour valider les points bas du Nasdaq.
7. **OilBenchModel** — modèle cognitif spécialisé pour le WTI (indicateurs techniques + fondamentaux EIA + sentiment).

### 🧘 Philosophie de Décision : "Prudence Cognitive"

Les modèles cognitifs (Gemma 4, sentiment, Vincent Ganne) détiennent **75 %** du poids de décision, contre **25 %** pour le modèle quantitatif agressif. Cette surpondération délibérée garantit que le contexte qualitatif tempère les signaux quantitatifs. Un signal n'est exécuté que si la confiance globale dépasse **40 %** ; entre 20 % et 40 %, il est rétrogradé en HOLD.

### ✨ Fonctionnalités clés

- **Architecture LLM Hybride Cloud/Local** : intégration `free-llm-api-keys` pour exploiter des « Frontier Models » très intelligents (DeepSeek, Claude, Gemini) pour l'analyse textuelle, avec un fallback 100 % robuste vers l'Ollama local (qui reste le moteur exclusif pour les charts visuels).
- **Approche Dual-Ticker** : analyser l'indice, trader l'ETF.
- **Prix live T212** : récupération en temps réel des prix EUR via l'API Trading 212 (0,2 s), avec fallback yfinance et cache parquet.
- **Spread Brent Dated** : suivi de la tension du marché physique via le spread entre le Brent Spot (Dated) et le Brent à terme.
- **Résilience réseau** : circuit breaker yfinance avec trackers séparés (info vs. download), timeout 10 s sur tous les appels réseau.
- **Auto-invalidation du cache** : le cache Parquet détecte sa péremption (> 2 jours) et force un rafraîchissement. Utilisez `refresh_cache.py` pour un vidage manuel.
- **Parallélisation des appels LLM** : les appels de modèles indépendants (`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`) s'exécutent dans un `ThreadPoolExecutor` pour recouvrir l'inférence Ollama avec les I/O. Chemin critique typiquement 4–6 min sur CPU contre 10+ min en séquentiel.
- **Cache 24h des requêtes de recherche** : la requête de recherche web générée par le LLM est mise en cache sous `data_cache/search_queries/<ticker>_<date>_<price-sig>.json`. Clé par date + signature de prix-action (bucketing log2 du close + bucket RSI), donc un changement de régime l'invalide. Les requêtes de fallback ne sont **jamais** mises en cache (une défaillance transitoire d'Ollama ne peut pas empoisonner le cache pendant 24h).
- **Timeout de cycle strict** : chaque cycle par ticker est encapsulé dans un budget de 15 min (`CYCLE_TIMEOUT_SECONDS` dans `main.py`). Sur timeout, le thread de travail est `shutdown(wait=F)` pour que le ticker suivant démarre immédiatement ; un HOLD est appliqué au ticker expiré. Les futures individuels ont leurs propres timeouts par tâche (recherche 240 s, visuel 300 s, texte 240 s, modèles CPU 180 s chacun, news 90 s, crawl web 30 s).
- **Sécurité anti-thread orphelin** : sur timeout de cycle, un `threading.Event` par ticker est positionné pour que le worker orphelin abandonne avant tout appel `execute_t212_trade` — empêchant des transactions en argent réel après que l'utilisateur ait vu le panneau « HOLD appliqué ». Un `threading.Lock` par ticker sérialise en outre le placement d'ordres T212, éliminant le risque de double-trade sous chevauchement du scheduler ou invocations `--ticker` dupliquées.
- **Sentinelle d'échec LLM** : quand `_query_ollama` épuise toutes ses retries, le dictionnaire de fallback porte un marqueur `"failed": True` pour que la logique de consensus en aval puisse distinguer « le modèle a choisi HOLD » de « le modèle a planté » (actuellement propagé mais non filtré — un suivi connu).
- **Cognition avancée** : utilisation de **Gemma 4 12B** avec **défense JSON bi-couche** :
  1. **Application du schéma côté serveur** (`format: SCHEMA_*` avec `additionalProperties: false`) — la couche porteuse ; passée via le paramètre `format` d'Ollama à chaque site d'appel. Schémas définis dans `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`).
  2. **Suffixe défensif du system prompt** (`"...never add a 'thought' key."`) — seconde ligne redondante-mais-inoffensive, conservée comme belt-and-braces contre toute régression future de la couche schéma.

  Le token de raisonnement `<|think|>` est **actif** dans les quatre system prompts de production (réactivé le 2026-06-06 sur `main` après validation sur la branche `think-mode`). C'est la couche schéma qui neutralise réellement le défaut historique de débris JSON `<|channel>thought` (cause racine mai 2026) : `tests/check_llm_json.py` confirme que les cas schema-strict (`v3_schema`, `v6_schema`, `v7_schema_strict`) produisent un JSON propre même avec `<|think|>` activé, tandis que les variantes loose `format:json` échouent. Voir `docs/ADR-001-think-mode-dual-layer-defence.md` pour l'analyse complète et la procédure de reversal.
- **Agent autonome de Morning Brief** : un workflow nocturne basé sur `smolagents` (`morning_brief/morning_brief.py`) planifié automatiquement à 01:00 via `schedule.py`. Il crawl indépendamment les logs API quotidiens, télécharge les données fondamentales d'inventaire EIA, et arbitre un débat *Bull vs Bear*. Le rapport markdown qui en résulte (`morning_market_brief.md`) est automatiquement injecté dans le system prompt du LLM textuel durant le cycle de trading quotidien, conférant à l'IA principale une mémoire contextuelle profonde et une conscience fondamentale sans ralentir l'exécution en marché live.
- **🏛️ Weekend Council (Mémoire Stratégique)** : une rétrospective LLM multi-personas hebdomadaire (`src/council/weekend_council.py`) s'exécutant chaque **samedi à 01:00** via `schedule.py`. Six personas — chacun sur une **famille de modèle Ollama distincte** (Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / LFM 2.5 / Mistral Nemo 12B) pour une véritable diversité de raisonnement — délibèrent sur un protocole 4 rounds (Problem Restate Gate → Analysis avec STANCE explicite → Débat 1-vs-1 → Synthèse du Juge) avec mécanismes anti-groupthink (quota de dissidence, verdict unresolved-first). Le Juge (Qwen3.5-9B-MTP) émet une stance par ticker qui devient le **11ème vote pondéré** (9,5 %) du consensus temps réel, avec une confiance décroissant linéairement sur 7 jours. Des budgets de tokens généreux (`num_predict` jusqu'à 12000, `num_ctx` jusqu'à 65536) et une fenêtre de scheduler de 48 heures accommodent les thinking models sur CPU. Le council analyse les vraies données PROD : précision des modèles (`model_performance.db`), métriques de portefeuille et alertes critiques (`performance_monitor.db`), et le journal de trading exécuté. Installez les 6 modèles requis avec `uv run python setup_council_models.py`. Voir `docs/ADR-003-weekend-council-11th-voice.md`.
- **News & Sentiment Blockchain** : intégration d'**AlphaEar** et **Hyperliquid** pour capturer le sentiment social et spéculatif.
- **Scheduler automatisé** : script `schedule.py` pour une exécution continue (8:30 – 18:00) sur un serveur.
- **Gestion du risque centralisée** : l'`AdvancedRiskManager` centralise la logique Anti-Loss (Stop-Loss) et Trailing Stop. Les modèles individuels ne gèrent plus ces risques, garantissant une stratégie unifiée et stricte de protection du capital à travers les régimes de marché.
- **Contrats de données stricts** : tous les modèles IA sont entièrement standardisés pour retourner une dataclass fortement typée `ModelResult` (`signal`, `confidence`, `reasoning`), assurant 100 % d'uniformité à travers le moteur de consensus.
- **Santé du code auditée** : le projet maintient un standard de santé de code **Grade B** via des audits automatisés (0 code mort, indice de maintenabilité élevé).
- **Backtesting production** : moteur de backtest autonome (`backtest_prod.py`) rejouant les vrais signaux prod contre les vrais prix avec frais T212 — aucune dépendance externe.
- **Contrôle du dump de debug** : définir `TRADING_DEBUG_DUMP=0` pour désactiver le dump (limité à 5 MB) `data_cache/llm_debug_fail.txt` des échecs LLM.

### 💻 Stack technologique

- **Langage** : `Python 3.12+`
- **Calculs & Données** : `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning** : `scikit-learn`, `shap`
- **IA & LLM** : `google-genai` (Gemini), `requests`, `ollama`
- **Scraping Web & Recherche** : `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualisation** : `matplotlib` (backend Agg pour la thread safety), `seaborn`, `mplfinance`
- **Utilitaires** : `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Performances & Matériel
Le système est conçu pour être **performant sur du matériel grand public** sans nécessiter de GPU dédié.
- **CPU uniquement** : l'inférence LLM (Gemma 4 12B Q6_K via Ollama) et TimesFM tournent entièrement sur CPU. Le débit est de ~3–4 tokens/s sur un CPU moderne 8 cœurs.
- **RAM recommandée** : 16 Go minimum (32 Go suggérés pour faire tourner confortablement Gemma 4 12B à côté de TimesFM et TensorTrade).
- **Concurrence Ollama** : définir `OLLAMA_NUM_PARALLEL=8` (déjà dans le `.env` recommandé) pour que plusieurs appels LLM partagent la charge du modèle. Avec le budget de contexte par défaut de 4 Go, les slots parallèles obtiennent ~512 tokens chacun — Ollama sérialisera si les prompts dépassent le ctx par slot, mais le `ThreadPoolExecutor` garde le recouvrement wall-clock bénéfique pour les étapes liées aux I/O (fetch de news, crawl web, modèles CPU).
- **Temps d'exécution** : ~6 à 9 minutes par ticker sur CPU (à froid), ~3 à 5 minutes par ticker avec un hit du cache de requêtes de recherche. L'exécution par défaut porte sur deux tickers (CRUDP.PA + SXRV.DE), donc prévoir ~15 min au total.
- **Timeout de cycle** : chaque cycle par ticker est borné à 15 min (`CYCLE_TIMEOUT_SECONDS`). En cas de dépassement, un HOLD est appliqué et le ticker suivant démarre immédiatement.
- **Vitesse API** : intégration Trading 212 ultra-rapide (<1 s pour la récupération du prix live).

### 🧠 Architecture IA & LLM (Gemini + Fallback Local)
Le système exploite une architecture multi-niveaux très robuste pour assurer un uptime maximum et une prise de décision intelligente, profondément intégrée dans `main.py` et le Weekend Council.

- **Cascade de Fallback 4-Niveaux** :
  1. **Palier Gemini Payant (`GEMINI_API_KEY_PAY`)** : Plus haute priorité. Utilise des modèles avancés comme Gemini 2.5 Pro pour le raisonnement complexe, la vision de charts techniques et les décisions de trading finales.
  2. **Palier Gemini Gratuit (`GEMINI_API_KEY`)** : Utilisé pour des tâches plus légères à fort volume telles que la synthèse de contexte web.
  3. **Proxies d'API LLM gratuites** : Sauvegarde via `free-llm-api-keys`.
  4. **Ollama Local** : Fallback CPU hors ligne 100 % robuste si tous les services cloud tombent.
- **Protection des coûts** : le palier payant est borné par un budget de coût sur 30 jours glissants (`GEMINI_PAY_MONTHLY_BUDGET_EUR`, défaut 8,6 €/mois) — le coût de chaque appel est calculé à partir de l'usage réel des tokens × le prix du modèle et accumulé ; quand le budget est atteint, les appels basculent vers le palier gratuit / Ollama. Un backstop quotidien (`GEMINI_PAY_DAILY_CAP`, défaut 200) protège contre les boucles incontrôlées.
- **Intégration** : le moteur principal d'exécution quotidienne (`main.py`) utilise Gemini pour le consensus multi-modèles temps réel, tandis que le Weekend Council asynchrone (`council`) intègre Gemini spécifiquement pour certains rôles (comme le Juge et le Sceptique) à côté de divers modèles Ollama locaux.

### 🧠 FinAcumen (Mémoire Financière)
L'architecture FinAcumen a été intégrée pour doter les modèles IA locaux d'une **mémoire d'expérience** et d'outils déterministes. Cela résout le problème de l'amnésie des LLMs.
- FinAcumen fonctionne **de manière asynchrone la nuit** (via `schedule.py`) pour bénéficier de la pleine puissance du CPU sans bloquer les cycles de trading.
- Son rapport qualitatif profond est automatiquement ajouté au **Morning Market Brief** pour guider le LLM de décision tout au long de la journée de trading.

## 📂 Structure du projet

Le projet est organisé de manière modulaire pour une meilleure maintenabilité.

```
Trading-AI/
├── morning_brief/                   # Agent autonome nocturne d'analyse fondamentale approfondie
│   ├── morning_brief.py             # Orchestrateur d'agents et configuration smolagents
│   └── output/                      # Rapports markdown quotidiens générés (morning_market_brief.md)
├── src/                             # Modules cœur
│   ├── adaptive_weight_manager.py   # Pondération dynamique des modèles selon la performance
│   ├── advanced_risk_manager.py     # Gestion du risque Trend-Aware et sizing
│   ├── bootstrap.py                 # Logique d'initialisation cœur
│   ├── chart_generator.py           # Génère les charts techniques pour le LLM visuel
│   ├── classic_model.py             # Ensemble de modèles quantitatifs Scikit-learn
│   ├── config_weights.py            # Configuration des poids de base du moteur hybride
│   ├── data.py                      # Fetch, cache et prétraitement des données
│   ├── database.py                  # Gestion base SQLite pour les métriques
│   ├── eia_client.py                # Client API Energy Information Administration
│   ├── enhanced_decision_engine.py  # Moteur de fusion hybride orchestrant tous les modèles
│   ├── enhanced_trading_example.py  # Scripts d'exemple d'utilisation des modèles
│   ├── features.py                  # Ingénierie de features techniques et macroéconomiques
│   ├── grebenkov_model.py           # Modèle mathématique Trend-Following (Agnostic Risk Parity)
│   ├── hmm_model.py                 # Hidden Markov Model pour la détection de régime
│   ├── llm_client.py                # Intégration Ollama pour l'inférence LLM locale
│   ├── news_fetcher.py              # Crawl et parsing de news financières
│   ├── oil_bench_model.py           # Modèle de trading WTI spécialisé énergie
│   ├── performance_monitor.py       # Suivi de la précision et de l'historique des modèles
│   ├── read_simul.py                # Outils de lecture des sorties de simulation
│   ├── sentiment_analysis.py        # Intégration sentiment Alpha Vantage & AlphaEar
│   ├── t212_executor.py             # Exécution réelle API Trading 212 et portefeuille
│   ├── tensortrade_model.py         # Signal Reinforcement Learning (PPO)
│   ├── timesfm_model.py             # Intégration forecast de séries temporelles TimesFM 2.5
│   └── web_researcher.py            # Scraping macro-économique web avec Crawl4AI
├── data_cache/                       # Tous les caches (gitignoré)
│   ├── *.parquet                     # Données OHLCV par ticker (yfinance)
│   ├── macro/                        # Séries temporelles macro (FRED, multi-sources)
│   ├── search_queries/               # Cache 24h des requêtes de recherche LLM (par ticker+date+price-sig)
│   └── llm_debug_fail.txt            # Dump (limité 5 MB) des échecs LLM — désactiver avec TRADING_DEBUG_DUMP=0
├── tests/                            # Scripts de test et validation
│   ├── test_full_cycle.py            # Test end-to-end T212 achat/attente/vente
│   ├── test_enhanced_decision_engine.py # Tests du moteur de fusion hybride
│   ├── check_llm_json.py             # Diagnostic JSON-schema LLM (teste les 4 sites d'appel Ollama)
│   ├── check_live.py                 # Script de vérification des prix de marché live
│   └── ...                           # Autres tests unitaires et d'intégration
├── i18n/                            # Internationalisation (READMEs traduits)
├── assets/                          # Assets statiques (images, bannières)
├── memory-bank/                     # État déterministe 4-fichiers + contexte long-form (voir AGENTS.md §1)
├── backtest_prod.py                 # Moteur de backtest production autonome
├── main.py                          # Point d'entrée unique (Analyse & Trading)
├── pyproject.toml                   # Dépendances et configuration du projet (uv)
├── refresh_cache.py                 # Utilitaire CLI pour forcer le rafraîchissement du cache Parquet
├── schedule.py                      # Scheduler live pour l'exécution automatisée
├── setup_timesfm.py                 # Script d'installation du vendor TimesFM 2.5
├── .env.example                     # Exemple de variables d'environnement
└── README.md                        # Cette documentation
```

---

## 🚀 Démarrage rapide

Suivez ces étapes pour configurer votre environnement de développement local.

### ✅ Prérequis

- Python 3.12+ (via `uv`)
- [Ollama](https://ollama.com/) installé et en cours d'exécution localement.
- Modèle LLM téléchargé : `ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`
- **Modèles du Weekend Council** (optionnels, mais requis pour la diversité de raisonnement du council) : le council fait tourner chaque persona sur une famille de modèle *différente* (Gemma / GLM / Qwen / LFM). Installez-les tous d'un coup avec `uv run python setup_council_models.py`.

### ⚙️ Installation

1.  **Cloner le dépôt :**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Installer `uv` (si ce n'est pas déjà fait) :**
    Voir [astral.sh/uv](https://astral.sh/uv) pour les instructions d'installation.

3.  **Créer et activer l'environnement virtuel (ÉTAPE CRUCIALE) :**
    Vous devez créer et activer le `.venv` avant d'installer les modèles de fondation.
    ```bash
    uv venv
    source .venv/bin/activate  # Sur Windows, utilisez `.\.venv\Scripts\activate.ps1`
    ```

4.  **Installer les modèles de fondation :**
    Lancez les scripts d'installation pour cloner les modèles dans `vendor/` et appliquer les patches :
    ```bash
    python setup_timesfm.py
    ```

5.  **Initialiser et synchroniser l'environnement :**
    ```bash
    uv sync
    ```

6.  **Installer les navigateurs pour la recherche Web (Crawl4AI) :**
    ```bash
    uv run python -m playwright install chromium
    ```

7.  **Configurer vos clés API :**
    Créez un fichier `.env` à la racine du projet :
    ```
    ALPHA_VANTAGE_API_KEY="VOTRE_CLE"
    EIA_API_KEY="VOTRE_CLE"

    # Optionnel mais fortement recommandé : Intégration Gemini AI
    GEMINI_API_KEY_PAY="VOTRE_CLE_PALIER_PAYANT"  # Pour le raisonnement/vision complexes (Gemini 2.5 Pro)
    GEMINI_API_KEY="VOTRE_CLE_PALIER_GRATUIT"      # Pour les tâches plus légères (synthèse)
    GEMINI_PAY_MONTHLY_BUDGET_EUR=8.6        # Budget de coût sur 30 jours glissants (€) — garde-fou de facturation portant
    GEMINI_PAY_DAILY_CAP=200                 # Backstop : max d'appels API payants par jour
    ```

---

## 🛠️ Utilisation

Le système entraîne ses modèles sur les données les plus récentes à chaque exécution avant de donner une décision.

### Mode Simulation (Paper Trading)

Pour tester le système sans risque avec un capital fictif de 1000 €, utilisez le flag `--simul`. Le système gérera un historique strict d'achats et de ventes.

```sh
# Lancer une analyse simulée (Défaut : SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Lancer sur le Pétrole (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Exécution Réelle (Trading 212)

Le système est désormais **pleinement intégré** à Trading 212 :
- **Vérification du portefeuille** : avant toute action, le robot consulte votre cash et positions réels.
- **Gestion de l'API** : inclut des mécanismes de retry automatique contre les limites de requêtes (Rate Limiting).

```sh
# Lancer l'analyse avec exécution réelle (Démo ou Réel selon .env)
uv run main.py --t212
```

---

## 🧪 Backtesting en Production

Le système inclut un **moteur de backtest production autonome** (`backtest_prod.py`) qui rejoue les vrais signaux prod de `logs_prod/trading_journal.csv` contre les vrais prix des fichiers Parquet de `data_cache/`.

### Fonctionnalités
- **Vrais signaux** : rejoue les décisions exactes du moteur hybride à 12 modèles.
- **Vrais prix** : utilise les vraies données OHLCV des ETF (SXRV.DE, CRUDP.PA) — pas de proxies US.
- **Frais T212** : simule le modèle de frais de Trading 212 de 0,1 % par trade.
- **Comparaison baseline** : calcule automatiquement la performance buy-and-hold comme benchmark.
- **Métriques** : Sharpe Ratio, Drawdown Maximum, Win Rate, Alpha, Rendement Total par ticker.

### Utilisation

```bash
uv run python backtest_prod.py
```

Résultats sauvegardés dans `logs_prod/backtest_report.json` avec courbes d'équity en CSV.

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à forker le projet et à ouvrir une Pull Request.

---

## 📜 Licence

Distribué sous la licence MIT.

---

## 📧 Contact

Lien du projet : [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

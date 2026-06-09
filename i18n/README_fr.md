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
  <img src="assets/banner.png" alt="Hybrid AI Trading Banner" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Système de Trading Hybride par IA 📈</h1>
  <p>
    Un système expert d'aide à la décision pour le trading d'ETFs NASDAQ et Pétrole (WTI), exploitant une intelligence artificielle hybride tri-modale pour des signaux de trading robustes et nuancés.
  </p>
</div>

<div align="center">

[![Project Status](https://img.shields.io/badge/status-in--development-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Table des Matières

- [🌟 À propos du projet](#-à-propos-du-projet)
  - [✨ Fonctionnalités clés](#-fonctionnalités-clés)
  - [💻 Stack technologique](#-stack-technologique)
  - [⚙️ Performances & Matériel](#️-performances--matériel)
- [📂 Structure du projet](#-structure-du-projet)
- [🚀 Démarrage rapide](#-démarrage-rapide)
  - [✅ Prérequis](#-prérequis)
  - [⚙️ Installation](#️-installation)
- [🛠️ Utilisation](#️-utilisation)
  - [Mode Simulation (Paper Trading)](#mode-simulation-paper-trading)
  - [Exécution Réelle (Trading 212)](#exécution-réelle-trading-212)
- [🧪 Backtesting en Production](#-backtesting-en-production)
- [🤝 Contribuer](#-contribuer)
- [📜 Licence](#-licence)
- [📧 Contact](#-contact)

---

## 🌟 À propos du projet

Ce projet est un système expert d'aide à la décision pour le trading d'ETFs, utilisant une approche d'IA hybride tri-modale. Il est conçu pour fournir une analyse complète et robuste en combinant plusieurs perspectives de l'IA.

### 🚀 Stratégie Dual-Ticker (Analyse vs. Trading)
Le système utilise une approche innovante pour maximiser la précision des modèles :
- **Analyse Haute Fidélité** : Les modèles d'IA analysent les **indices de référence mondiaux** (`^NDX` pour le Nasdaq, `CL=F` pour le pétrole brut WTI). Ces indices offrent un historique plus long et des tendances plus "pures", sans le bruit lié aux heures de cotation ou aux frais d'ETFs.
- **Exécution ETF** : Les ordres réels sont placés sur les tickers correspondants sur **Trading 212** (`SXRV.DE`, `CRUDP.PA`), en utilisant les **prix en direct de T212** (via l'API des positions) pour le dimensionnement des positions. L'état du portefeuille est synchronisé directement depuis T212 (`sync_state_from_t212()`), et les prix en direct sont injectés dans le pipeline d'analyse (`_inject_t212_live_price()` dans `src/data.py`).

### 🧠 Moteur d'IA Hybride
Le système fusionne onze signaux distincts :
1.  **Modèle Quantitatif Classique** : Ensemble RandomForest/GradientBoosting/LogisticRegression entraîné sur des indicateurs techniques et macroéconomiques.
2.  **TimesFM 2.5 (Google Research)** : Modèle de fondation de pointe pour les prévisions de séries temporelles.
3.  **TensorTrade / PPO (Apprentissage par Renforcement)** : Agent RL (stable-baselines3) entraînant une politique PPO dans un environnement de trading personnalisé Gymnasium avec persistance entre les cycles.
4.  **Modèle Oil-Bench (Gemma 4 12B (Unsloth))** : Modèle spécialisé dans l'énergie fusionnant les données fondamentales de l'**EIA** (Stocks, Importations, Utilisation des raffineries) et le sentiment pour le trading du WTI.
5.  **LLM Textuel (Gemma 4 12B (Unsloth))** : Analyse contextuelle des données brutes, actualités en temps réel via la compétence **AlphaEar**, et intégration de **recherches web macro-économiques** dynamiques. Il consomme explicitement le rapport nocturne du **Morning Brief** pour acquérir une conscience fondamentale approfondie avant de prendre ses décisions.
6.  **LLM Visuel (Gemma 4 12B (Unsloth))** : Analyse directe des graphiques techniques (`enhanced_trading_chart.png`).
7.  **Analyse de Sentiment** : Analyse hybride combinant Alpha Vantage et les tendances "chaudes" d'**AlphaEar** (Weibo, WallstreetCN).
8.  **Données Décentralisées (Hyperliquid)** : Analyse du sentiment spéculatif sur le Pétrole (WTI) via le *Funding Rate* et l'*Open Interest*.
9.  **Modèle Vincent Ganne** : Analyse géopolitique et multi-actifs (WTI, Brent, Gaz, DXY, MA200) pour la détection de points bas macroéconomiques.
10. **Modèle Grebenkov** : Modèle mathématique de suivi de tendance calibré pour l'analyse multi-actifs utilisant la Parité de Risque Agnostique (Agnostic Risk Parity).
11. **Moteur de Fusion Hybride** : Le méta-modèle orchestrant la pondération dynamique et le consensus cognitif à travers tous les sous-modèles.

L'objectif est de produire une décision finale (`BUY`, `SELL`, `HOLD`) avec une priorité absolue donnée à la **Précision avant tout**.

### 🧘 Philosophie de Décision : "Prudence Cognitive"
Contrairement aux algorithmes de trading classiques qui paniquent dès que la volatilité explose, ce système applique une approche d'investisseur averti :
- **Consensus Fort Requis** : Un modèle quantitatif (Classique) peut crier au loup (`SELL`), mais si les modèles cognitifs (LLM Textuel, Vision, TimesFM) restent neutres, le système préférera `HOLD`.
- **Filtre de Confiance** : Une décision de mouvement (Achat ou Vente) n'est validée que si la confiance globale dépasse un seuil de sécurité (généralement 40 %). En dessous de ce seuil, le système considère le signal comme du "bruit" et reste en attente.
- **Protection du Capital** : En mode de risque `VERY_HIGH`, `HOLD` sert de bouclier. Il évite d'entrer sur un marché instable et d'en sortir prématurément lors d'une simple correction technique si les fondamentaux (Actualités/Vision/Hyperliquid) ne confirment pas un krach imminent.

### ✨ Fonctionnalités clés

- **Approche Dual-Ticker** : Analyser l'indice, trader l'ETF.
- **Prix en Direct T212** : Récupération en temps réel des prix en EUR via l'API Trading 212 (0.2s), avec yfinance en secours et un cache parquet.
- **Spread du Brent Daté** : Surveillance de la tension sur le marché physique via l'écart entre le Brent Spot (Daté) et les Contrats à Terme sur le Brent.
- **Résilience Réseau** : Disjoncteur yfinance avec trackers séparés (infos vs téléchargement), délai d'attente de 10s sur tous les appels réseau.
- **Auto-Invalidation du Cache** : Le cache Parquet détecte automatiquement la péremption (> 2 jours) et force une actualisation. Utilisez `refresh_cache.py` pour un vidage manuel du cache.
- **Parallélisation des Appels LLM** : Les appels indépendants aux modèles (`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`) s'exécutent dans un `ThreadPoolExecutor` pour superposer l'inférence Ollama aux E/S. Chemin critique généralement de 4 à 6 min sur CPU contre 10+ min en séquentiel.
- **Cache de Requête de Recherche de 24h** : La requête de recherche web générée par le LLM est mise en cache sous `data_cache/search_queries/<ticker>_<date>_<signature-prix>.json`. Clé basée sur la date + une signature de l'action des prix (regroupement log2 de la clôture + regroupement RSI), de sorte qu'un changement de régime l'invalide. Les requêtes de secours ne sont **jamais** mises en cache (une défaillance temporaire d'Ollama ne peut pas empoisonner le cache pendant 24h).
- **Délai d'Attente de Cycle Strict** : Chaque cycle de ticker est enveloppé dans un budget de 15 min (`CYCLE_TIMEOUT_SECONDS` dans `main.py`). En cas de dépassement, le thread de travail subit un `shutdown(wait=False)` pour que le ticker suivant commence immédiatement ; HOLD est appliqué au ticker en dépassement de délai. Les futures individuels ont leurs propres délais d'attente par tâche (recherche 240s, visuel 300s, texte 240s, modèles CPU 180s chacun, actualités 90s, crawl web 30s).
- **Sécurité des Threads Orphelins** : Lors d'un dépassement de délai de cycle, un `threading.Event` par ticker est défini afin que le travailleur orphelin s'arrête avant tout appel à `execute_t212_trade` — empêchant les transactions en argent réel après que l'utilisateur a vu le panneau "HOLD appliqué". Un `threading.Lock` par ticker sérialise davantage le placement des ordres T212, éliminant le risque de double transaction lors du chevauchement de l'ordonnanceur ou d'invocations `--ticker` en double.
- **Sentinelle de Défaillance LLM** : Lorsque `_query_ollama` épuise toutes ses tentatives, le dictionnaire de secours porte un indicateur `"failed": True` afin que la logique de consensus en aval puisse distinguer "le modèle a choisi HOLD" de "le modèle a planté" (actuellement propagé mais non filtré — un suivi connu).
- **Cognition Avancée** : Utilisation de **Gemma 4 12B** avec une **double couche de défense JSON** :
  1. **Application stricte du schéma côté serveur** (`format: SCHEMA_*` avec `additionalProperties: false`) — la couche porteuse ; passée via le paramètre `format` d'Ollama à chaque point d'appel. Schémas définis dans `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`).
  2. **Suffixe défensif d'invite système** (`"...never add a 'thought' key."`) — deuxième ligne redondante mais inoffensive, conservée en guise de ceinture de sécurité contre toute régression future de la couche de schéma.

  Le token de raisonnement `<|think|>` est **actif** dans les quatre invites système de production (réactivé le 2026-06-06 sur `main` après validation sur la branche `think-mode`). La couche de schéma est ce qui neutralise réellement le défaut historique de résidus JSON `<|channel>thought` (cause première de mai 2026) : `tests/check_llm_json.py` confirme que les cas de schéma strict (`v3_schema`, `v6_schema`, `v7_schema_strict`) produisent un JSON propre même avec `<|think|>` activé, tandis que les variantes non strictes `format:json` échouent. Voir `docs/ADR-001-think-mode-dual-layer-defence.md` pour l'analyse complète et la procédure d'inversion.
- **Agent Autonome Morning Brief** : Un processus de nuit basé sur `smolagents` (`morning_brief/morning_brief.py`) programmé pour s'exécuter automatiquement à 01h00 via `schedule.py`. Il analyse indépendamment les logs d'API quotidiens, télécharge les données fondamentales des inventaires EIA et arbitre un débat *Bull vs Bear*. Le rapport Markdown généré (`morning_market_brief.md`) est automatiquement injecté dans le Prompt système du LLM Textuel lors du cycle de trading quotidien, offrant à l'IA principale une mémoire contextuelle et une conscience fondamentale approfondies sans ralentir l'exécution sur le marché en direct.
- **Sentiment Actualités & Blockchain** : Intégration d'**AlphaEar** et d'**Hyperliquid** pour capter les sentiments sociaux et spéculatifs.
- **Planificateur Automatisé (Scheduler)** : Script `schedule.py` pour l'exécution continue (8h30 - 18h00) sur un serveur.
- **Gestion des Risques Centralisée** : L'`AdvancedRiskManager` centralise les logiques Anti-Perte (Stop-Loss) et Stop Suiveur (Trailing Stop). Les modèles individuels ne gèrent plus ces risques, assurant une stratégie de protection du capital unifiée et stricte à travers divers régimes de marché.
- **Contrats de Données Stricts** : Tous les modèles d'IA sont entièrement normalisés pour renvoyer une dataclass `ModelResult` fortement typée (`signal`, `confidence`, `reasoning`), garantissant une uniformité de 100 % dans le moteur de consensus.
- **Santé du Code Auditée** : Le projet maintient une norme de santé du code de **Grade B** grâce à des audits automatisés (0 code mort, indice de maintenabilité élevé).
- **Backtesting en Production** : Moteur de backtest autonome (`backtest_prod.py`) rejouant les signaux de production réels face aux prix réels avec les frais de T212 — sans dépendances externes.
- **Contrôle du vidage de débogage** : Définissez `TRADING_DEBUG_DUMP=0` pour désactiver le vidage des défaillances LLM (plafonné à 5 Mo) dans `data_cache/llm_debug_fail.txt`.

### 💻 Stack technologique

- **Langage** : `Python 3.12+`
- **Calculs & Données** : `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning** : `scikit-learn`, `shap`
- **IA & LLM** : `requests`, `ollama`
- **Web Scraping & Recherche** : `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualisation** : `matplotlib` (backend Agg pour la sécurité des threads), `seaborn`, `mplfinance`
- **Utilitaires** : `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Performances & Matériel
Le système est conçu pour être **performant sur du matériel grand public** sans nécessiter de GPU dédié.
- **CPU Uniquement** : L'inférence LLM (Gemma 4 12B Q6_K via Ollama) et TimesFM s'exécutent entièrement sur le processeur (CPU). Le débit est d'environ 3-4 tokens/s sur un processeur moderne à 8 cœurs.
- **RAM Recommandée** : 16 Go minimum (32 Go suggérés pour faire tourner Gemma 4 12B confortablement avec TimesFM et TensorTrade).
- **Concurrence Ollama** : Définissez `OLLAMA_NUM_PARALLEL=8` (déjà dans le `.env` recommandé) pour que plusieurs appels LLM puissent partager la charge du modèle. Avec le budget de contexte par défaut de 4 Go, les slots parallèles obtiennent environ 512 tokens chacun — Ollama sérialisera si les requêtes dépassent le contexte par slot, mais le `ThreadPoolExecutor` garde le chevauchement bénéfique pour les étapes liées aux E/S (récupération d'actualités, crawl web, modèles CPU).
- **Temps d'exécution** : ~6 à 9 minutes par ticker sur CPU (à froid), ~3 à 5 minutes par ticker avec accès au cache de la requête de recherche. L'exécution par défaut traite deux tickers (CRUDP.PA + SXRV.DE), prévoyez donc environ 15 minutes au total.
- **Délai d'Attente de Cycle** : Chaque cycle de ticker est limité à 15 minutes (`CYCLE_TIMEOUT_SECONDS`). En cas de dépassement, HOLD est appliqué et le ticker suivant commence immédiatement.
- **Vitesse de l'API** : Intégration Trading 212 ultra-rapide (< 1s pour la récupération des prix en direct).

---

## 📂 Structure du projet

Le projet est organisé de manière modulaire pour une meilleure maintenabilité.

```
Trading-AI/
├── morning_brief/                   # Agent autonome de nuit pour une analyse fondamentale approfondie
│   ├── morning_brief.py             # Orchestrateur de l'agent et configuration smolagents
│   └── output/                      # Rapports markdown générés quotidiennement (morning_market_brief.md)
├── src/                             # Modules principaux
│   ├── adaptive_weight_manager.py   # Pondération dynamique des modèles selon la performance
│   ├── advanced_risk_manager.py     # Gestion des risques adaptée aux tendances et dimensionnement
│   ├── chart_generator.py           # Génération de graphiques techniques pour le LLM visuel
│   ├── classic_model.py             # Ensemble de modèles quantitatifs Scikit-learn
│   ├── data.py                      # Récupération, mise en cache et prétraitement des données
│   ├── database.py                  # Gestion de la base de données SQLite pour les métriques
│   ├── eia_client.py                # Client de l'API de l'Energy Information Administration (EIA)
│   ├── enhanced_decision_engine.py  # Moteur de fusion hybride orchestrant tous les modèles
│   ├── features.py                  # Ingénierie des caractéristiques techniques et macroéconomiques
│   ├── grebenkov_model.py           # Modèle mathématique de suivi de tendance (Parité de Risque Agnostique)
│   ├── llm_client.py                # Intégration Ollama pour l'inférence LLM locale
│   ├── news_fetcher.py              # Récupération et analyse d'actualités financières
│   ├── oil_bench_model.py           # Modèle de trading WTI spécialisé dans l'énergie
│   ├── performance_monitor.py       # Suivi de la précision et de l'historique des modèles
│   ├── sentiment_analysis.py        # Intégration du sentiment Alpha Vantage & AlphaEar
│   ├── t212_executor.py             # Exécution réelle via l'API Trading 212 et portefeuille
│   ├── tensortrade_model.py         # Signal d'Apprentissage par Renforcement (PPO)
│   ├── timesfm_model.py             # Intégration de TimesFM 2.5 pour les prévisions de séries temporelles
│   └── web_researcher.py            # Scraping web macro-économique avec Crawl4AI
├── data_cache/                       # Tous les caches (ignorés par git)
│   ├── *.parquet                     # Données OHLCV par ticker (yfinance)
│   ├── macro/                        # Séries temporelles macro (FRED, multi-sources)
│   ├── search_queries/               # Cache 24h des requêtes LLM (par ticker+date+sig-prix)
│   └── llm_debug_fail.txt            # Fichier de vidage des erreurs LLM (5 Mo max) — désactivable via TRADING_DEBUG_DUMP=0
├── tests/                            # Scripts de tests et de validation
│   ├── test_full_cycle.py            # Test complet d'achat/attente/vente T212
│   ├── test_enhanced_decision_engine.py # Tests pour le moteur de fusion hybride
│   ├── check_llm_json.py             # Diagnostic du schéma JSON LLM (teste les 4 appels Ollama)
│   ├── check_live.py                 # Script de vérification des prix du marché en direct
│   └── ...                           # Autres tests unitaires et d'intégration
├── i18n/                            # Internationalisation (Fichiers README traduits)
├── assets/                          # Ressources statiques (images, bannières)
├── memory-bank/                     # Mémoire de l'assistant IA et contexte
├── backtest_prod.py                 # Moteur de backtest de production autonome
├── main.py                          # Point d'entrée unique (Analyse & Trading)
├── pyproject.toml                   # Dépendances et configuration du projet (uv)
├── refresh_cache.py                 # Utilitaire CLI pour forcer l'actualisation du cache Parquet
├── schedule.py                      # Planificateur pour exécution automatisée en direct
├── setup_timesfm.py                 # Script d'installation pour le composant externe TimesFM 2.5
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

### ⚙️ Installation

1.  **Cloner le dépôt :**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Installer `uv` (si ce n'est pas déjà fait) :**
    Consultez [astral.sh/uv](https://astral.sh/uv) pour les instructions d'installation.

3.  **Créer et activer l'environnement virtuel (Étape CRUCIALE) :**
    Vous devez créer et activer l'environnement `.venv` avant d'installer les modèles de fondation.
    ```bash
    uv venv
    source .venv/bin/activate  # Sous Windows, utilisez `.\.venv\Scripts\activate.ps1`
    ```

4.  **Installer les modèles de fondation :**
    Exécutez les scripts d'installation pour cloner les modèles dans `vendor/` et appliquer les correctifs :
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

7.  **Configurer vos clés d'API :**
    Créez un fichier `.env` à la racine du projet :
    ```
    ALPHA_VANTAGE_API_KEY="VOTRE_CLE"
    EIA_API_KEY="VOTRE_CLE"
    ```

---

## 🛠️ Utilisation

Le système entraîne ses modèles sur les données les plus récentes à chaque exécution avant de donner une décision.

### Mode Simulation (Paper Trading)

Pour tester le système sans risque avec un capital fictif de 1000 €, utilisez l'option `--simul`. Le système gérera un historique strict des achats et ventes.

```sh
# Lancer une analyse simulée (Par défaut : SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Lancer sur le Pétrole (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Exécution Réelle (Trading 212)

Le système est désormais **pleinement intégré** avec Trading 212 :
- **Vérification du Portefeuille** : Avant toute action, le robot consulte vos liquidités et positions réelles.
- **Gestion des API** : Inclut des mécanismes de relance automatique face aux limites de requêtes (Rate Limiting).

```sh
# Lancer l'analyse avec exécution réelle (Démo ou Réel selon le .env)
uv run main.py --t212
```

---

## 🧪 Backtesting en Production

Le système intègre un **moteur de backtest de production autonome** (`backtest_prod.py`) qui rejoue les signaux de production réels issus de `logs_prod/trading_journal.csv` face aux prix réels des fichiers Parquet de `data_cache/`.

### Fonctionnalités
- **Signaux réels** : Rejoue les décisions exactes du moteur hybride à 11 modèles.
- **Prix réels** : Utilise les véritables données OHLCV de l'ETF (SXRV.DE, CRUDP.PA) — aucun proxy américain.
- **Frais T212** : Simule le modèle de frais à 0,1 % par transaction de Trading 212.
- **Comparatif de référence** : Calcule automatiquement la performance de la stratégie d'achat et conservation (Buy-and-Hold) à titre de référence.
- **Métriques** : Ratio de Sharpe, Drawdown Maximum, Taux de Réussite (Win Rate), Alpha, Rendement Total par ticker.

### Utilisation

```bash
uv run python backtest_prod.py
```

Résultats sauvegardés dans `logs_prod/backtest_report.json` avec CSV des courbes de capitaux (equity curves).

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à faire un fork du projet et à ouvrir une Pull Request.

---

## 📜 Licence

Distribué sous la licence MIT.

---

## 📧 Contact

Lien du projet : [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

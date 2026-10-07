<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_fr.md">Français</a>
</p>

<p align="center">
  <img src="../assets/banner.png" alt="Bannière Trading IA Hybride" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Système de Trading IA Hybride 📈</h1>
  <p>
    <b>Pipeline d'aide à la décision algorithmique multi-modèle à haute conviction pour ETFs NASDAQ et Secteur Énergie.</b><br />
    Guidé par le principe quantitatif fondamental : <i>« En trading, la complexité ne paie pas — commencez par la technique la plus simple et exigez des preuves pour chaque couche de complexité. »</i>
  </p>
</div>

<div align="center">

[![Statut du projet](https://img.shields.io/badge/statut-production%20active-success.svg)](https://github.com/laurentvv/Trading-AI)
[![Version Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Licence](https://img.shields.io/badge/licence-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)
[![Passerelle LLM](https://img.shields.io/badge/passerelle%20LLM-NexusAI--Client-orange.svg)](https://github.com/laurentvv/NexusAI-Client)
[![Modèle de Fondation](https://img.shields.io/badge/prvision-TimesFM%203.0-purple.svg)](https://github.com/google-research/timesfm)

</div>

---

## 📚 Table des Matières

- [🌟 Vision Architecturale](#-vision-architecturale)
  - [Stratégie Dual-Ticker (Analyse sur Indices vs Exécution ETF)](#stratégie-dual-ticker-analyse-sur-indices-vs-exécution-etf)
  - [Ensemble de Modèles Resserré (Réduction de Complexité)](#ensemble-de-modèles-resserré-réduction-de-complexité)
  - [Pivot Énergie : De l'ETC Synthétique à l'ETF Sectoriel Physique](#pivot-énergie--de-letc-synthétique-à-letf-sectoriel-physique)
- [🛡️ Invariants de Production & GO-Gates](#️-invariants-de-production--go-gates)
- [🧠 L'Ensemble Décisionnel à Haute Conviction](#-lensemble-décisionnel-à-haute-conviction)
- [🧪 Moteur de Backtest Vectorisé Pur](#-moteur-de-backtest-vectorisé-pur)
- [📂 Structure du Projet](#-structure-du-projet)
- [🚀 Démarrage Rapide](#-démarrage-rapide)
  - [Prérequis](#prérequis)
  - [Installation](#installation)
  - [Pré-téléchargement du Modèle de Fondation](#pré-téléchargement-du-modèle-de-fondation)
- [🛠️ Exploitation Opérationnelle](#️-exploitation-opérationnelle)
  - [Simulation (Paper Trading)](#simulation-paper-trading)
  - [Exécution Broker Réelle / Démo (Trading 212)](#exécution-broker-réelle--démo-trading-212)
  - [Scheduler Autonome Supervisé](#scheduler-autonome-supervisé)
  - [Conseil Stratégique du Week-end](#conseil-stratégique-du-week-end)
  - [Morning Market Brief](#morning-market-brief)
- [📜 Licence](#-licence)

---

## 🌟 Vision Architecturale

### Stratégie Dual-Ticker (Analyse sur Indices vs Exécution ETF)
Les instruments retail (ETFs) présentent souvent des cotations discontinues, des spreads élargis hors des heures de marché et des anomalies de flux. Le système sépare rigoureusement la perception analytique de l'exécution financière :
1. **Perception sur Indices de Référence** : Les modèles IA analysent les indices sous-jacents mondiaux ultra-liquides (**`^NDX`** pour les technologiques, **`CL=F`** pour les cours mondiaux du pétrole). Ces indices offrent un historique profond sur plusieurs décennies, une volatilité continue et une absence de distorsion horaire.
2. **Exécution sur ETFs UCITS Européens** : Les ordres validés sont exécutés sur des instruments EUR liquides sur **Trading 212** :
   - Indice Tech : **`SXRV.DE`** (iShares Nasdaq 100 UCITS ETF EUR, ticker T212 `SXRVd_EQ`).
   - Secteur Énergie : **`QDVF.DE`** (iShares S&P 500 Energy Sector UCITS ETF EUR Acc, ticker T212 `QDVFd_EQ`).
3. **Prix Réconcilié en Temps Réel** : Les prix d'exécution proviennent directement de l'API Trading 212 (<0.5s), protégeant contre toute clôture périmée.

---

### Ensemble de Modèles Resserré (Réduction de Complexité)
À la suite d'un audit quantitatif rigoureux basé sur l'adage *"En trading, la complexité ne paie pas"*, le système a mis en quarantaine les modèles "zombies" (apprentissage par renforcement PPO suréchantillonné, HMM discrétisé instable, formules macro sans données) pour concentrer l'allocation sur **des moteurs à haute conviction** :

| Moteur Modèle | Technologie | Poids de Base | Rôle & Avantage |
|---|---|:---:|---|
| **TimesFM 3.0** | Google Foundation Model (`timesfm3`) | **25%** | Prévision autorégressive zero-shot de séries temporelles (médiane + 9 quantiles). |
| **Ensemble Quant Classique** | Scikit-Learn (RF, GB, Régression Logistique) | **20%** | Estimation de momentum technique et macroéconomique multi-facteurs. |
| **Modèle Grebenkov** | Trend-Following & Parité de Risque Agnostique | **20%** | Détection mathématique de persistance de tendance. |
| **LLM Textuel Unifié** | Passerelle Cloud `NexusAI-Client` | **15%** | Synthèse des actualités, annonces macro et recherche web dynamique. |
| **LLM Multimodal Vision** | Modèles Cloud Vision Frontière | **10%** | Reconnaissance de figures chartistes et chandeliers japonais (`enhanced_trading_chart.png`). |
| **Conseil Stratégique Week-end** | Débat 6 Personas Multi-Providers | **10%** | Rétrospective stratégique hebdomadaire avec décroissance linéaire sur 7 jours. |
| **Modèle Oil-Bench** | Fondamentaux EIA (Stocks, Imports, Raffinage) | *Dynamique (10%)* | Modèle physique offre/demande activé sur les instruments énergie. |

> **Modèles en Quarantaine** : `tensortrade` (PPO RL), `hmm_model`, `vincent_ganne`, et `sentiment` sont isolés à un **poids de base de 0.0** avec bypass de l'exécution des threads pour éliminer le bruit et préserver les ressources.

---

### Pivot Énergie : De l'ETC Synthétique à l'ETF Sectoriel Physique
Historiquement, l'exposition énergie était confiée à des ETCs synthétiques sur contrats futures WTI (`CRUDP.PA` / `OD7Fd_EQ`). Une analyse empirique walk-forward a mis en lumière deux défauts structurels majeurs :
- **Le Piège de l'Érosion Contango** : Le roulement négatif continu des contrats à terme détruit la valeur dans le temps (**-10.8% de CAGR** avec filtre MA200 et **-68.7% de Drawdown Maximum**).
- **Gel de Cotations** : Les flux de données sur les ETCs européens affichent des trous critiques (81.9% de barres gelées sur `CRUDP.PA`).

**La Solution Retenue — `QDVF.DE` (iShares S&P 500 Energy Sector UCITS ETF EUR)** :
- **Panier d'Actions Physiques** : Investi dans les géants de l'énergie américaine (ExxonMobil, Chevron, ConocoPhillips, EOG).
- **Zéro Érosion Contango** : Performance robuste de **+15.4% CAGR Buy & Hold** sur 11 ans (2 732 séances, 97.6% de données saines).
- **Rendement Réel et Dividendes** : Génération de trésorerie (~3.5% réinvesti).
- **Bêta Direct** : Forte sensibilité aux chocs pétroliers tout en capturant la rentabilité opérationnelle des producteurs.

---

## 🛡️ Invariants de Production & GO-Gates

Sept **GO-Gates** stricts et non négociables garantissent la sécurité opérationnelle :

1. **Idempotence des Ordres (GO-gate 1)** : Les ordres BUY au marché envoient des payloads épurés `{ticker, quantity}` avec un timeout de 15s. L'état courtier est réconcilié avant toute tentative de réémission.
2. **Stop-Loss Broker avec Ratchet (GO-gate 2)** : Chaque position ouverte dispose d'un stop GTC placé chez le courtier (crête × 0.90), qui ne peut être que relevé (ratchet UP).
3. **Confirmation d'Exécution (GO-gate 3)** : Aucune écriture d'état ou de base de données ne s'effectue sans confirmation du fill chez le broker via `/equity/portfolio/positions`.
4. **Volatilité Quotidienne Standardisée (GO-gate 4)** : Calculée en écart-type sur 20 jours (jamais annualisée), parfaitement calibrée aux seuils d'action.
5. **Interdiction des Données Synthétiques (GO-gate 5)** : Données macro synthétiques interdites. Refus strict des caches de prix datant de plus de 3 jours.
6. **Scheduler à Instance Unique (GO-gate 6)** : Verrou atomique `scheduler.lock` avec PID et thread gardien évitant toute double exécution concurrente.
7. **Comptabilité FIFO Réelle (GO-gate 7)** : `equity = initial_budget + P&L réalisé (FIFO) + latent`, tracé dans `trading_journal.csv` (colonne `T212_Equity`).

---

## 🧠 L'Ensemble Décisionnel à Haute Conviction

### 1. Google TimesFM 3.0
- Intégration officielle via le paquet PyPI `timesfm>=3.0.1` (`timesfm3`).
- Poids de fondation `google/timesfm-3.0-pytorch` (~1.3 Go en cache local).
- Inférence autorégressive avec un contexte de 2 048 barres sur CPU (~0.35s par calcul).

### 2. Passerelle Cloud LLM Unifiée (NexusAI-Client)
- Propulsé par **[`NexusAI-Client`](https://github.com/laurentvv/NexusAI-Client)** : zéro dépendance locale (pas d'Ollama ni de modèles GGUF lourds).
- Résilience automatique sans coût à travers plus de 9 fournisseurs cloud :
  - **Gemini Free / Gemini Pro** (Google)
  - **Groq & Cerebras** (Inférence LPU ultra-rapide)
  - **Mistral AI & Cohere**
  - **Nvidia NIM & OpenRouter / OrcaRouter**
- **Défense JSON Double Couche** : Extraction stricte validant le schéma `{signal, confidence, analysis}`.

### 3. Conseil Stratégique du Week-end
- Délibération automatique chaque fin de semaine (`src/council/weekend_council.py`).
- 6 personas spécialisés (Stratège Macro, Risk Manager, Quant, Sceptique Baissier, Tacticien, Analyste Comportemental).
- Chaque persona consulte un **provider cloud distinct** pour garantir une réelle indépendance de raisonnement.
- Décroissance linéaire sur 7 jours injectée comme 11ème vote pondéré dans le consensus temps réel.

---

## 🧪 Moteur de Backtest Vectorisé Pur

Le projet dispose d'une suite de backtest vectorisé en NumPy / pandas (`src/backtest/`) :
- Replay fidèle intégrant les frais Trading 212 (0.1%).
- Comparatif automatisé des stratégies de suivi de tendance (Buy & Hold, MA50, MA200, MA200 hystérésis 1.5%, EMA, RSI, MACD, Bandes de Bollinger).
- Génération automatique de rapports détaillant Sharpe, CAGR, Drawdown Max, Win Rate et Profit Factor.

Pour lancer un comparatif :
```bash
uv run python -m src.backtest.run_benchmark
```

---

## 📂 Structure du Projet

```
Trading-AI/
├── src/                             # Code source de production
│   ├── adaptive_weight_manager.py   # Pondération dynamique bayésienne / win-rate
│   ├── advanced_risk_manager.py     # Sizing et trailing stops trend-aware
│   ├── backtest/                    # Moteur de backtest vectorisé pur
│   ├── chart_generator.py           # Génération des graphiques chandeliers
│   ├── classic_model.py             # Ensemble quantitatif Scikit-Learn
│   ├── config_weights.py            # Poids des modèles et statuts de quarantaine
│   ├── data.py                      # Gestion des flux et caches de marché
│   ├── database.py                  # Persistance SQLite des transactions et états
│   ├── eia_client.py                # Client API EIA v2 pour les fondamentaux pétrole
│   ├── enhanced_decision_engine.py  # Moteur de consensus et filtre de quorum
│   ├── enhanced_trading_example.py  # Orchestrateur de pipeline et worker pool
│   ├── features.py                  # Ingénierie des indicateurs techniques
│   ├── grebenkov_model.py           # Modèle de suivi de tendance Agnostic Risk Parity
│   ├── llm_client.py                # Passerelle LLM NexusAI (Texte & Vision)
│   ├── news_fetcher.py              # Collecte des actualités financières
│   ├── oil_bench_model.py           # Modèle fondamental pétrole EIA
│   ├── performance_monitor.py       # Suivi de performance et calcul du win-rate
│   ├── t212_executor.py             # Exécution Trading 212, ratchet de stop et FIFO
│   ├── timesfm_model.py             # Wrapper du modèle TimesFM 3.0
│   ├── web_researcher.py            # Moteur de requêtes de recherche web
│   └── council/                     # Délibération stratégique multi-agents
├── morning_brief/                   # Synthèse fondamentale matinale autonome
├── memory-bank/                     # Gestion déterministe de l'état (4 fichiers)
├── tests/                           # 430+ tests unitaires, d'intégration et de sécurité
├── main.py                          # Point d'entrée de pipeline en ligne de commande
├── schedule.py                      # Scheduler de production continu
└── scheduler_config.json            # Configuration runtime centralisée
```

---

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.12+
- Gestionnaire d'environnement ultra-rapide [`uv`](https://astral.sh/uv)
- Compte Trading 212 (démo ou réel)
- Clés API cloud configurées dans `.env` (Gemini, Groq, Mistral, Nvidia NIM, etc.)

### Installation
```powershell
# 1. Cloner le dépôt
git clone https://github.com/laurentvv/Trading-AI.git
Set-Location Trading-AI

# 2. Synchroniser l'environnement virtuel avec uv
uv sync

# 3. Installer les dépendances de navigation pour la recherche web
uv run python -m playwright install chromium
```

### Pré-téléchargement du Modèle de Fondation
Avant de démarrer le scheduler, téléchargez le checkpoint Google TimesFM 3.0 (~1.3 Go) :
```powershell
uv run python tests/smoke_timesfm3.py
```

---

## 🛠️ Exploitation Opérationnelle

### Simulation (Paper Trading)
Exécuter un cycle analytique en simulation (capital virtuel de 1 000 € par ticker) :
```powershell
# Tickers par défaut (QDVF.DE et SXRV.DE)
uv run main.py --simul

# Ticker spécifique
uv run main.py --simul --ticker QDVF.DE
```

### Exécution Broker Réelle / Démo (Trading 212)
Exécuter les ordres réels via l'API Trading 212 (`T212_ENV=demo` ou `live` dans `.env.t212`) :
```powershell
uv run main.py --t212
```

### Scheduler Autonome Supervisé
Démarrer le planificateur de production (toutes les 30 minutes de 08:30 à 18:00 CET) :
```powershell
# Exécution directe
uv run schedule.py

# Boucle supervisée avec redémarrage automatique en cas de crash
.\start_scheduler.bat
```

### Conseil Stratégique du Week-end
Déclencher le débat multi-providers à la demande :
```powershell
uv run python -m src.council.weekend_council --days 7
```

### Morning Market Brief
Générer la synthèse matinale des marchés et des données EIA :
```powershell
uv run python morning_brief/morning_brief.py
```

---

## 🧪 Validation & Suite de Tests

Exécuter la suite complète de plus de 430 tests :

```powershell
# Suite complète de tests mockés
.venv\Scripts\python.exe -m pytest tests/ -q --basetemp=data_cache/test_tmp

# Tests ciblés sur les ordres T212 et la sécurité
.venv\Scripts\python.exe -m pytest tests/test_t212_orders.py tests/test_equity_tracking.py tests/test_data_safety.py -v
```

---

## 📜 Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus de détails.
Les poids de Google TimesFM 3.0 sont régis par la licence `timesfm-non-commercial-license-v1.0`.

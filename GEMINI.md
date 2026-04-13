# GEMINI.md

## Project Overview

Ce projet est un système expert d'aide à la décision pour le trading d'ETFs NASDAQ et Pétrole (WTI). Il utilise une approche **IA hybride tri-modale** et une stratégie **Dual-Ticker** unique :
- **Analyse sur Indices** : Le système télécharge et analyse les indices de référence (`^NDX`, `CL=F`) pour obtenir des signaux d'IA plus propres et robustes.
- **Trading sur ETFs** : Les décisions sont appliquées aux ETFs correspondants sur Trading 212 (`SXRV.DE`, `CRUDP.PA`).

Le moteur fusionne un modèle quantitatif classique, un LLM textuel (Gemma 4), un LLM visuel (analyse de graphiques), le modèle de fondation **TimesFM 2.5** de Google Research, et le **Modèle Vincent Ganne** (Géopolitique & Cross-Asset).

### Nouveautés majeures :
- **Modèle Vincent Ganne :** Détection de points bas boursiers via l'analyse du Pétrole (WTI/Brent), du Gaz Naturel (TTF), de l'Urée, du Dollar (DXY) et des moyennes mobiles à 200 jours.
- **Intégration Hyperliquid :** Capture du sentiment spéculatif sur le Pétrole via les données blockchain (*Funding Rate*, *Open Interest*).
- **Gestion des Risques "Trend-Aware" :** Le système adapte ses seuils de confiance selon la tendance du marché (plus agressif en Bull Market).
- **Sizing Progressif :** Exposition dynamique du portefeuille (75% à 100%) basée sur le score de consensus de l'IA.
- **Verrou Géopolitique :** Blocage automatique des achats si les prix de l'énergie (WTI/Brent) dépassent les seuils critiques de stabilité macroéconomique.

## Building and Running

### Prerequisites

- Python 3.12+ (via `uv`)
- Ollama fonctionnant localement avec `gemma4:e4b`
- Clé API Alpha Vantage (pour la macroéconomie et le sentiment)

### Installation

1.  **Installer `uv`** : [astral.sh/uv](https://astral.sh/uv)
2.  **Initialiser l'environnement et TimesFM 2.5** :
    ```bash
    # IMPORTANT : Exécuter ce script en premier pour cloner et patcher TimesFM
    python setup_timesfm.py
    
    # Ensuite, synchroniser l'environnement complet
    uv sync
    ```
    *Cette procédure clone TimesFM dans `vendor/`, applique les correctifs d'API 2.5 et synchronise toutes les dépendances.*
3.  **Configurer l'API** : Créer un fichier `.env` avec `ALPHA_VANTAGE_API_KEY`.

### Running the System

```bash
# Analyse standard (Analyse ^NDX, trading virtuel SXRV.DE)
uv run main.py

# Analyse Pétrole (Analyse CL=F, trading virtuel CRUDP.PA)
uv run main.py --ticker CRUDP.PA

# Exécution réelle sur Trading 212 (Mode DEMO ou REEL via .env)
uv run main.py --t212

# Lancer le scheduler automatique (8h30 - 18h00, Lun-Ven)
uv run schedule.py
```

The system now uses **Gemma 4:e4b** for enhanced cognitive analysis and integrates the **AlphaEar** skill for real-time financial news context and **Hyperliquid** for decentralized sentiment.

The scheduler will run in the background, perform periodic analysis (every 30 minutes), and execute trades on Trading 212. All activities are logged in `scheduler.log`.

## Configuration

The behavior of the Intelligent Scheduler can be customized via a `scheduler_config.json` file placed in the root directory of the project. If this file is not present, the scheduler will use a default configuration.

Creating this file allows you to control parameters such as phase durations, performance targets, and the project's start date.

### Example `scheduler_config.json`

```json
{
    "project_start_date": "2025-08-25T18:05:27.149745",
    "trading_ticker": "QQQ",
    "daily_execution_time": "18:00",
    "weekly_report_day": "friday",
    "monthly_report_day": 28,
    "phase_transitions": {
        "phase_1_duration_days": 7,
        "phase_2_duration_days": 21,
        "phase_3_duration_days": 30,
        "phase_4_duration_days": 120
    },
    "performance_targets": {
        "phase_2": {
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.05,
            "win_rate": 0.45
        },
        "phase_3": {
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.03,
            "win_rate": 0.55
        },
        "phase_4": {
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.02,
            "win_rate": 0.60
        }
    },
    "alerts": {
        "email_notifications": false,
        "performance_alerts": true,
        "phase_completion_alerts": true
    }
}
```

### Key Parameters

*   `project_start_date`: The official start date of the project. This is crucial for calculating phase transitions.
*   `phase_transitions`: Allows you to define the duration (in days) for each of the four project phases. This is the primary way to control the automatic transition between phases.

## Development Conventions

*   **Modularity:** The codebase is organized into a clean, modular structure with a clear separation of concerns.
*   **Logging:** The project uses the `logging` module with **UTF-8 encoding** to support emojis and special characters on Windows.
*   **Data Caching:** Market data is cached locally in Parquet files to speed up subsequent runs.
*   **Documentation-Driven Development:** The `memory-bank/` directory contains documentation about the project's evolution, architecture, and context.
*   **Configuration:** Key parameters and constants are defined at the beginning of the scripts.
*   **Error Handling:** The code includes robust error handling for API requests (Hyperliquid, yfinance, FRED), file operations (atomic JSON writes), and data structure validations (Pandas Series/Float fixes).
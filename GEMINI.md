# GEMINI.md

## Project Overview

Ce projet est un système expert d'aide à la décision pour le trading d'ETFs NASDAQ et Pétrole (WTI). Il utilise une approche **IA hybride tri-modale** et une stratégie **Dual-Ticker** unique :
- **Analyse sur Indices** : Le système télécharge et analyse les indices de référence (`^NDX`, `CL=F`) pour obtenir des signaux d'IA plus propres et robustes.
- **Trading sur ETFs** : Les décisions sont appliquées aux ETFs correspondants sur Trading 212 (`SXRV.DE`, `CRUDP.PA`).

Le moteur fusionne un modèle quantitatif classique, un LLM textuel (Gemma 4), un LLM visuel (analyse de graphiques), le modèle de fondation **TimesFM 2.5** (Google Research), et le **Modèle Vincent Ganne** (Géopolitique & Cross-Asset).

### Nouveautés majeures :
- **Sécurité Anti-Perte & Trailing Stop :** Blocage automatique des ventes à perte et déclenchement de prises de profits (Stop Suiveur 3%) pour sécuriser le cash.
- **Inversion du Risque Pétrole :** Le système reconnaît désormais que la haute volatilité est un signal haussier pour le pétrole et booste les scores d'achat en conséquence.
- **Mémoire de Performance :** Enregistrement de chaque décision individuelle dans `model_performance.db` pour l'ajustement automatique des poids via le Weight Manager.
- **Intégration EIA (Energy Information Administration) :** Analyse automatisée des données fondamentales américaines (Stocks de brut, Importations, Taux d'utilisation des raffineries) et des prévisions STEO.
- **Modèle Oil-Bench (Gemma 4) :** Nouveau modèle spécialisé dans le pétrole, fusionnant les données EIA et le sentiment de marché pour une analyse fondamentale profonde.
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

    # Installer les navigateurs pour la recherche Web
    uv run python -m playwright install chromium
    ```
    *Cette procédure clone TimesFM dans `vendor/`, applique les correctifs d'API 2.5 et synchronise toutes les dépendances.*
3.  **Configurer l'API** : Créer un fichier `.env` avec `ALPHA_VANTAGE_API_KEY` et `EIA_API_KEY`.

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

The behavior of the Intelligent Scheduler and all trading components (Decision Engine, Risk Manager, Weight Manager) is now centralized in a `scheduler_config.json` file. This file serves as the single source of truth for thresholds, risks, and model parameters, allowing for tuning without code changes.

### Centralized Configuration (`scheduler_config.json`)

The system uses an **Injection de Dépendance** pattern where the orchestrator loads the configuration and passes it to all sub-components.

```json
{
    "project_start_date": "2025-08-25T18:05:27.149745",
    "trading_ticker": "QQQ",
    "model_thresholds": {
        "vincent_ganne": {
            "WTI": {"max": 94, "ideal": 80},
            "Brent": {"max": 95, "ideal": 83},
            "Gas": {"max": 55, "ideal": 38}
        }
    },
    "risk_parameters": {
        "max_drawdown_warning": 0.05,
        "max_drawdown_critical": 0.1
    },
    "weight_manager": {
        "regime_thresholds": {
            "high_vol": 0.30
        }
    }
}
```

## Development Conventions

*   **Architecture Découplée :** Utilisation d'une interface `BaseModel` pour tous les modèles IA, permettant d'ajouter de nouveaux signaux sans modifier le moteur de décision.
*   **Standardized Logging :** Remplacement systématique des `print()` par le module `logging` pour une meilleure traçabilité en production.
*   **Modularity:** The codebase is organized into a clean, modular structure with a clear separation of concerns.
*   **Logging:** The project uses the `logging` module with **UTF-8 encoding** to support emojis and special characters on Windows.
*   **Data Caching:** Market data is cached locally in Parquet files to speed up subsequent runs.
*   **Documentation-Driven Development:** The `memory-bank/` directory contains documentation about the project's evolution, architecture, and context.
*   **Configuration:** Key parameters and constants are defined at the beginning of the scripts.
*   **Error Handling:** The code includes robust error handling for API requests (Hyperliquid, yfinance, FRED), file operations (atomic JSON writes), and data structure validations (Pandas Series/Float fixes).(Pandas Series/Float fixes).
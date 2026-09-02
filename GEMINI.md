# GEMINI.md

## Project Overview

Ce projet est un système expert d'aide à la décision pour le trading d'ETFs NASDAQ et Pétrole (WTI). Il utilise une approche **IA multi-modale** et une stratégie **Dual-Ticker** unique :
- **Analyse sur Indices** : Le système télécharge et analyse les indices de référence (`^NDX`, `CL=F`) pour obtenir des signaux d'IA plus propres et robustes.
- **Trading sur ETFs** : Les décisions sont appliquées aux ETFs correspondants sur Trading 212 (`SXRV.DE`, `CRUDP.PA`).

Le moteur fusionne un modèle quantitatif classique, un LLM textuel, un LLM visuel (analyse de graphiques), le modèle de fondation **TimesFM 3.0** (Google Research), et le **Modèle Vincent Ganne** (Géopolitique & Cross-Asset).

### Nouveautés majeures :
- **Architecture LLM Unifiée via NexusAI-Client :** Remplacement complet d'Ollama et de toutes les IA locales par **[`NexusAI-Client`](https://github.com/laurentvv/NexusAI-Client)** avec fallback automatique multi-fournisseurs (Gemini Free/Pro, Groq, Cerebras, Mistral, Cohere, Nvidia NIM, OpenRouter, OrcaRouter, DeepSeek).
- **Sécurité Anti-Perte & Trailing Stop :** Blocage automatique des ventes à perte et déclenchement de prises de profits (Stop Suiveur 3%) pour sécuriser le cash.
- **Inversion du Risque Pétrole :** Le système reconnaît désormais que la haute volatilité est un signal haussier pour le pétrole et booste les scores d'achat en conséquence.
- **Mémoire de Performance :** Enregistrement de chaque décision individuelle dans `model_performance.db` pour l'ajustement automatique des poids via le Weight Manager.
- **Intégration EIA (Energy Information Administration) :** Analyse automatisée des données fondamentales américaines (Stocks de brut, Importations, Taux d'utilisation des raffineries) et des prévisions STEO.
- **Modèle Oil-Bench (NexusAI) :** Modèle spécialisé dans le pétrole, fusionnant les données EIA et le sentiment de marché pour une analyse fondamentale profonde.
- **Modèle Vincent Ganne :** Détection de points bas boursiers via l'analyse du Pétrole (WTI/Brent), du Gaz Naturel (TTF), de l'Urée, du Dollar (DXY) et des moyennes mobiles à 200 jours.
- **Intégration Hyperliquid :** Capture du sentiment spéculatif sur le Pétrole via les données blockchain (*Funding Rate*, *Open Interest*).
- **Gestion des Risques "Trend-Aware" :** Le système adapte ses seuils de confiance selon la tendance du marché (plus agressif en Bull Market).
- **Sizing Progressif :** Exposition dynamique du portefeuille (75% à 100%) basée sur le score de consensus de l'IA.
- **Weekend Council (11ème Voix) :** Délibération rétrospective asynchrone le week-end réunissant 6 personas sur des providers cloud distincts (Groq, Cerebras, Mistral, Cohere, OpenRouter/Nvidia, Gemini Flash & Pro).
- **FinAcumen (Mémoire d'Expérience) :** Exécution asynchrone d'un agent cognitif profond, injectant son analyse structurelle dans le Morning Brief pour guider les décisions temps réel.

## Building and Running

### Prerequisites

- Python 3.12+ (via `uv`)
- Clés API configurées dans `.env` (au moins un provider pour `NexusAI-Client`, ex: `GEMINI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `NVIDIA_API_KEY`, etc.)
- Clé API Alpha Vantage (pour la macroéconomie et le sentiment)

### Installation

1.  **Installer `uv`** : [astral.sh/uv](https://astral.sh/uv)
2.  **Initialiser l'environnement (TimesFM 3.0 vient de PyPI via `uv sync`)** :
    ```bash
    # Synchroniser l'environnement complet
    uv sync

    # Pré-télécharger le checkpoint TimesFM 3.0 (~1.3 Go, une fois par machine)
    uv run python tests/smoke_timesfm3.py

    # Installer les navigateurs pour la recherche Web
    uv run python -m playwright install chromium
    ```
3.  **Configurer l'API** : Créer un fichier `.env` avec vos clés API (voir `.env.example`).

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

The scheduler will run in the background, perform periodic analysis (every 30 minutes), and execute trades on Trading 212. All activities are logged in `scheduler.log`.

## Configuration

The behavior of the Intelligent Scheduler and all trading components (Decision Engine, Risk Manager, Weight Manager) is centralized in a `scheduler_config.json` file.

### Centralized Configuration (`scheduler_config.json`)

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
*   **Standardized Logging :** Utilisation du module `logging` avec **UTF-8 encoding** pour le support des caractères spéciaux et emojis sur Windows.
*   **Modularity :** Structure modulaire avec séparation nette des composants.
*   **Data Caching :** Données de marché et requêtes mises en cache localement au format Parquet / JSON dans `data_cache/`.
*   **Documentation-Driven Development :** Le répertoire `memory-bank/` contient le suivi déterministe de l'état du projet.

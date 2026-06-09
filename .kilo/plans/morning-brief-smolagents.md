# Plan: Morning Market Brief — Orchestrateur Multi-Agents smolagents

## Version API cible
**smolagents v1.26.0** — [Documentation officielle](https://huggingface.co/docs/smolagents/v1.26.0/en/index)

## Objectif
Créer un script Python autonome utilisant `smolagents` (`CodeAgent` + `LiteLLMModel`) dans un répertoire dédié `morning_brief/`. L'agent analyse le système Trading-AI, le marché WTI/Nasdaq, et génère un rapport Markdown structuré avec débat de 3 personas (Bull/Bear/Risk Manager). **Aucun fichier existant n'est modifié.**

---

## Structure du livrable

```
morning_brief/
├── morning_brief.py          # Script principal (entry point unique)
├── tools/
│   ├── __init__.py
│   ├── analyze_trading_logs.py
│   ├── audit_portfolio_performance.py
│   ├── analyze_wti_market.py
│   ├── analyze_nasdaq.py
│   └── analyze_market_sentiment.py
├── prompts/
│   └── debate_prompt.py      # Template du débat Bull/Bear/Risk Manager
├── requirements.txt          # Dépendances smolagents
└── output/                   # Répertoire de sortie (gitignored)
    └── morning_market_brief.md
```

---

## 1. Outils Techniques

### Approche v1.26.0 : `Tool` subclass (recommandé pour outils complexes)

Les outils sont créés par subclassing `Tool` (approche recommandée par la doc v1.26.0 pour les outils avec logique complexe). Chaque outil définit :
- `name: str` — nom descriptif
- `description: str` — description injectée dans le system prompt de l'agent
- `inputs: dict` — types Pydantic (`"string"`, `"boolean"`, `"integer"`, `"number"`, `"array"`, `"object"`)
- `output_type: str` — type de retour (Pydantic format)
- `forward(self, ...)` — logique métier

> Note : Tous les imports sont définis **à l'intérieur** des méthodes (requis par smolagents v1.26.0 pour `save()`/`push_to_hub()`).

### 1.1 `analyze_trading_logs`
- **Source de données** : `D:\temp\Trading-AI\trading.log` (format Python logging standard)
- **Logique** : 
  - Parse le fichier avec regex sur les patterns `ERROR`, `WARNING`, `CRITICAL`
  - Extrait les déconnexions API (pattern `circuit breaker`, `timeout`, `connection`, `FRED.*failed`)
  - Détecte le slippage (pattern `slippage` ou écart prix T212 vs Yahoo)
  - Extrait les erreurs modèle (pattern `Erreur`, `error`, `failed`)
  - Retourne un dict structuré : `{errors: [...], warnings: [...], api_disconnects: int, slippage_events: [...], health_score: 0-100}`
- **Fallback** : Si le fichier n'existe pas ou est vide, retourne un diagnostic "NO_DATA"

### 1.2 `audit_portfolio_performance`
- **Source de données** : `D:\temp\Trading-AI\performance_monitor.db` (SQLite)
- **Schéma cible** : Tables `realtime_metrics`, `daily_performance`, `performance_alerts` (définies dans `src/performance_monitor.py:130-201`)
- **Logique** :
  - Query `realtime_metrics` trié par `timestamp DESC LIMIT 1` pour chaque ticker
  - Calcule PnL veille = `daily_return` de la dernière entrée
  - Nombre de trades = `total_trades` de la dernière entrée
  - Drawdown max = `max_drawdown` de la dernière entrée
  - Compte les alerts non acquittées sur 24h
  - Retourne : `{tickers: [{ticker, pnl, trades, drawdown, alerts}], overall_health: str}`

### 1.3 `analyze_wti_market`
- **Sources** : yfinance (`CL=F`) + Hyperliquid (via `hyperliquid-python-sdk`) + EIA API
- **Logique native réécrite** (pas de subprocess, pas d'import du projet alerte-wti) :
  - **Prix WTI temps réel** : `yf.Ticker("CL=F").history(period="5d")` — extraction Close, High, Low, Volume
  - **Variations** : calcul variation 1j, 5j, 20j en %
  - **Moyennes mobiles** : SMA20, SMA50, SMA200 calculées sur `history(period="1y")`
  - **VWAP** : calcul sur données intraday ou daily (`sum(Price * Volume) / sum(Volume)`)
  - **RSI 14** : calcul standard Wilder
  - **Bandes de Bollinger** (20, 2σ)
  - **Hyperliquid** : prix Brent Spot et spread Brent/WTI (même logique que `src/data.py` via `hyperliquid.info.Info`)
  - **EIA** : si `EIA_API_KEY` disponible, query inventaires via `src/eia_client.py` logique (HTTP direct vers `api.eia.gov/v2`)
  - **Flux RSS critique** : intégration du système de keyword filtering de `alerte-wti` (`CRITICAL_KEYWORDS`, `CRITICAL_COMBO`, `OIL_CONTEXT_WORDS` du fichier `alerte-wti-main/trading_bot/main.py:208-331`) + RSS feeds de `sources.json` — fetch des 14 flux, pre-filtrage par mots-clés, retourne les 5 headlines les plus critiques
- **Retour** : `{price, change_1d, change_5d, sma20, sma50, sma200, vwap, rsi, bollinger, brent_spread, eia_inventories, critical_headlines: [...]}`

### 1.4 `analyze_nasdaq`
- **Sources** : yfinance (`^NDX`, `SXRV.DE`) + données existantes Trading-AI
- **Logique** (réutilise les patterns de `src/data.py` et `src/grebenkov_model.py`) :
  - **Technique** : RSI, MACD, Bandes Bollinger sur `^NDX` history
  - **Volumes** : analyse volume vs moyenne 20j
  - **Corrélation WTI-Nasdaq** : calcul corrélation rolling 20j entre CL=F et ^NDX sur 6 mois de données
  - **Divergence** : détection divergence price/RSI sur 5j
- **Retour** : `{price, change, rsi, macd, volume_ratio, wti_correlation_20d, divergence_signal}`

### 1.5 `analyze_market_sentiment`
- **Sources** : RSS feeds macro (EIA, Bloomberg, Fed) + Alpha Vantage NEWS_SENTIMENT
- **Logique** :
  - Fetch flux RSS macro : EIA Today in Energy, EIA Weekly Petroleum, Google News "macroeconomics M2"
  - Utilise l'API Alpha Vantage `NEWS_SENTIMENT` si clé disponible (même pattern que `src/news_fetcher.py:42-60`)
  - Scoring sentiment simple : comptage mots positifs/négatifs sur les titres (dictionnaire financier léger en mémoire)
  - Extraction indicateurs macro : références à M2, taux Fed, CPI, chômage
- **Retour** : `{headlines: [...], sentiment_score: -1 to +1, macro_signals: {fed, cpi, employment, m2}, key_themes: [...]}`

---

## 2. Mécanique des Personas

Le prompt de l'agent (`prompts/debate_prompt.py`) impose un débat structuré en 3 phases :

### Phase 1 — Collecte (implicite via les outils)
L'agent appelle séquentiellement les 5 outils.

### Phase 2 — Débat structuré (via `instructions` paramètre v1.26.0)

En v1.26.0, `MultiStepAgent` (parent de `CodeAgent`) accepte un paramètre `instructions: str` injecté dans le system prompt. C'est plus propre que d'embed le débat dans le `run()` task. Le `run()` reçoit seulement la tâche concise.

```
DEBATE_INSTRUCTIONS = """
Tu es un comité d'investissement à 3 voix. Analyse les données des outils puis produis un débat structuré.

ÉTAPE 1 - Voix 🟢 THE BULL :
Analyse TOUS les arguments haussiers : supports WTI tenus, MA price > MA200, RSI non suracheté, logs système OK, macro positive, sentiment favorable. Sois convaincant.

ÉTAPE 2 - Voix 🔴 THE BEAR :
Recherche ACTIVEMENT les failles et risques : WTI surachat (RSI>70), slippage détecté, erreurs système, macro incertaine, volumes faibles, divergence baissière. Sois critique.

ÉTAPE 3 - Voix 🛡️ RISK MANAGER (Décision Finale) :
Arbitre le débat en te basant sur le drawdown actuel du portefeuille. Si drawdown > 5% => bias Bear obligatoire. Si drawdown < 2% et Bull convaincant => bias Bull possible. Sinon => Neutral.
Produis la recommandation finale avec position sizing.

Format de sortie obligatoire : Markdown strict selon le template fourni dans la tâche.
"""
```

**Paramètres v1.26.0 utilisés :**
- `instructions=DEBATE_INSTRUCTIONS` sur `CodeAgent` — injecté dans le system prompt (v1.26.0 `MultiStepAgent`)
- `use_structured_outputs_internally=True` — améliore la qualité des appels d'outils pour de nombreux modèles
- `final_answer_checks=[validate_markdown]` — valide que le output respecte le template Markdown avant acceptation

L'agent est un `CodeAgent` unique — les personas sont gérés par le prompt, pas par 3 agents séparés (plus simple, plus fiable, un seul passage LLM).

---

## 3. Orchestrateur (`morning_brief.py`)

### API v1.26.0 — Signatures exactes

```python
from smolagents import CodeAgent, LiteLLMModel

# LiteLLMModel v1.26.0 — kwargs (dont num_ctx) forwardés à litellm.completion()
model = LiteLLMModel(
    model_id="ollama_chat/gemma4:12b",
    api_base="http://localhost:11434",
    num_ctx=32768,
)

# CodeAgent v1.26.0
agent = CodeAgent(
    tools=[
        analyze_trading_logs,
        audit_portfolio_performance,
        analyze_wti_market,
        analyze_nasdaq,
        analyze_market_sentiment,
    ],
    model=model,
    instructions=DEBATE_INSTRUCTIONS,  # v1.26.0: injecté dans system prompt
    additional_authorized_imports=["json", "datetime", "re", "sqlite3", "pathlib", "os"],
    max_steps=10,
    planning_interval=3,
    use_structured_outputs_internally=True,  # v1.26.0: améliore tool calling
    final_answer_checks=[validate_markdown_output],  # v1.26.0: validation output
)

# run() v1.26.0 — task concise, instructions déjà dans le system prompt
result = agent.run(
    "Génère le Morning Market Brief pour aujourd'hui. "
    "Appelle les 5 outils dans l'ordre, puis synthétise le débat des 3 personas "
    "et écris le résultat dans le fichier markdown selon le template."
)
```

### Fonction de validation (`final_answer_checks`)

```python
def validate_markdown_output(final_answer, memory, agent):
    """Vérifie que le output contient les sections obligatoires du template."""
    required_sections = [
        "Santé du Système",
        "Analyse WTI",
        "Le Débat des Agents",
        "Risk Manager",
    ]
    return all(section in str(final_answer) for section in required_sections)
```

- **Séquence** : L'agent appelle les outils séquentiellement (imposé par le prompt), débat, puis écrit le fichier
- **Sortie** : Écriture du Markdown dans `morning_brief/output/morning_market_brief.md`
- **Logging** : Logging Python standard avec rotation

---

## 4. Template Markdown de sortie

Le prompt inclut le template exact à respecter :

```markdown
# 📈 Morning Market Brief — [YYYY-MM-DD]

## 1. Santé du Système & Portefeuille (`Trading-AI`)
* **Logs :** [Résumé des erreurs/slippage, nombre d'alertes]
* **Portefeuille :** [PnL veille par ticker, Drawdown max global]

## 2. Analyse WTI & Fondamentale
* **Technique WTI :** [Prix, Variation, RSI, Bollinger, VWAP, MA20/50/200, Brent Spread]
* **EIA Fondamentaux :** [Inventaires si disponibles]
* **Actualités Critiques :** [Top 3 headlines filtrées par mots-clés]
* **Sentiment Macro :** [Score sentiment, signaux Fed/CPI/M2]

## 3. Corrélations & Nasdaq
* **Nasdaq Technique :** [RSI, MACD, Volumes]
* **Corrélation WTI-Nasdaq :** [Coefficient 20j, divergence]

## 4. Le Débat des Agents (Comité d'Investissement)
* **🟢 The Bull :** [Argumentaire haussier structuré — 3-5 points]
* **🔴 The Bear :** [Argumentaire baissier structuré — 3-5 points]
* **🛡️ Risk Manager (Décision Finale) :**
  * Drawdown actuel : [X%]
  * Arbitrage : [Résumé de la décision]
  * **Biais recommandé : Bull / Bear / Neutral**
  * Position sizing : [% d'exposition recommandé]
```

---

## 5. Dépendances (`requirements.txt`)

```
smolagents[litellm]>=1.26.0
yfinance>=0.2.40
feedparser>=6.0.0
beautifulsoup4>=4.12.0
requests>=2.31.0
hyperliquid-python-sdk>=0.23.0
python-dotenv>=1.0.0
```

> Note : Pas d'installation globale. Le script sera exécuté depuis le venv existant de Trading-AI ou son propre venv.

---

## 6. Décisions architecturales

| Décision | Justification |
|---|---|
| Répertoire dédié `morning_brief/` | Isolation complète, aucun fichier existant modifié |
| CodeAgent unique + prompt multi-personas via `instructions` | v1.26.0 `instructions` param, plus simple que MultiAgent, un seul passage LLM |
| `gemma4:12b` via LiteLLMModel | Déjà utilisé dans le projet, bon ratio qualité/VRAM |
| Outils via `Tool` subclass | Recommandé v1.26.0 pour outils complexes (méthodes multiples, state interne) |
| `use_structured_outputs_internally=True` | v1.26.0 — améliore la fiabilité des appels d'outils |
| `final_answer_checks` | v1.26.0 — validation du template Markdown avant acceptation |
| Pas de subprocess | Contrainte explicite de l'utilisateur |
| Logique alerte-wti réécrite nativement | Keyword filtering + RSS + yfinance intégré dans l'outil Python |
| Pas de LLM local pour le filtrage news | Trop lourd pour un morning brief — on garde le keyword filtering (suffisant) |
| `num_ctx=32768` via kwargs | Forwardé à litellm.completion() — le débat + données tools + template markdown nécessite un contexte large |

---

## 7. Ordre d'implémentation

1. **Créer `morning_brief/` + sous-répertoires** (structure complète)
2. **`tools/analyze_trading_logs.py`** — Parsing du fichier log
3. **`tools/audit_portfolio_performance.py`** — Query SQLite
4. **`tools/analyze_wti_market.py`** — Capteur WTI complet (le plus complexe)
5. **`tools/analyze_nasdaq.py`** — Capteur Nasdaq + corrélation
6. **`tools/analyze_market_sentiment.py`** — RSS + sentiment
7. **`prompts/debate_prompt.py`** — Template prompt + template Markdown
8. **`morning_brief.py`** — Orchestrateur principal
9. **`requirements.txt`** — Dépendances
10. **Test d'intégration** — Exécution complète avec `ollama serve` actif

---

## 8. Risques et mitigations

| Risque | Mitigation |
|---|---|
| Ollama pas démarré | Check au lancement avec message clair |
| `trading.log` vide/inexistant | Fallback NO_DATA dans l'outil |
| SQLite DB verrouillée | Retry avec timeout + WAL mode |
| yfinance timeout | Circuit breaker identique à `src/data.py` |
| Context window insuffisant | `num_ctx=32768` + prompt concis |
| Flux RSS indisponibles | Timeout 10s par flux + fallback |

---

## 9. Points validés avec l'utilisateur

- [x] Repo alerte-wti : code source local `alerte-wti-main/trading_bot/main.py` — logique réécrite nativement
- [x] Modèle LLM : `gemma4:12b` via Ollama
- [x] Nasdaq : réutiliser la logique existante de Trading-AI (data.py, grebenkov_model.py)
- [x] Aucune modification du projet principal
- [x] smolagents v1.26.0 — API vérifiée contre la documentation officielle HF

import json
import logging
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "output" / "morning_brief.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("morning_brief")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DEBATE_INSTRUCTIONS = """Tu es un comite d'investissement a 3 voix. Analyse les donnees des outils puis produis un debat structure.

ETAPE 1 - Voix THE BULL :
Analyse TOUS les arguments haussiers : supports WTI tenus, MA price > MA200, RSI non surachte, logs systeme OK, macro positive, sentiment favorable.

ETAPE 2 - Voix THE BEAR :
Recherche ACTIVEMENT les failles et risques : WTI surachat (RSI>70), slippage detecte, erreurs systeme, macro incertaine, volumes faibles, divergence baissiere.

ETAPE 3 - Voix RISK MANAGER (Decision Finale) :
Arbitre le debat en te basant sur le drawdown actuel du portefeuille. Si drawdown > 5% => bias Bear obligatoire. Si drawdown < 2% et Bull convaincant => bias Bull possible. Sinon => Neutral.
Produis la recommandation finale avec position sizing.

Format de sortie obligatoire : Markdown strict selon le template fourni dans la tache.
Utilise final_answer() pour retourner le markdown complet.
"""

MARKDOWN_TEMPLATE = """# Morning Market Brief — {date}

## 1. Sante du Systeme & Portefeuille (Trading-AI)
* **Logs :** [Resume des erreurs/slippage, nombre d'alertes]
* **Portefeuille :** [PnL veille par ticker, Drawdown max global]

## 2. Analyse WTI & Fondamentale
* **Technique WTI :** [Prix, Variation, RSI, Bollinger, VWAP, MA20/50/200, Brent Spread]
* **EIA Fondamentaux :** [Inventaires si disponibles]
* **Actualites Critiques :** [Top 3 headlines filtrees par mots-cles]
* **Sentiment Macro :** [Score sentiment, signaux Fed/CPI/M2]

## 3. Correlations & Nasdaq
* **Nasdaq Technique :** [RSI, MACD, Volumes]
* **Correlation WTI-Nasdaq :** [Coefficient 20j, divergence]

## 4. Le Debat des Agents (Comite d'Investissement)
* **THE BULL :** [Argumentaire haussier structure — 3-5 points]
* **THE BEAR :** [Argumentaire baissier structure — 3-5 points]
* **RISK MANAGER (Decision Finale) :**
  * Drawdown actuel : [X%]
  * Arbitrage : [Resume de la decision]
  * **Biais recommande : Bull / Bear / Neutral**
  * Position sizing : [% d'exposition recommande]
"""


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _clean_output(raw: str) -> str:
    import re
    text = raw.strip()
    text = re.sub(r"```(?:python|markdown)?\s*", "", text)
    text = re.sub(r"^\s*markdown_output\s*=\s*\"\"\"", "", text, flags=re.MULTILINE)
    text = re.sub(r"\"\"\"\s*", "", text)
    text = re.sub(r"^\s*final_answer\(.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*print\(.*$", "", text, flags=re.MULTILINE)
    if not text.startswith("#"):
        for i, line in enumerate(text.split("\n")):
            if line.strip().startswith("#"):
                lines = text.split("\n")
                text = "\n".join(lines[i:])
                break
    return text.strip()


def validate_markdown_output(final_answer, _memory, agent):
    cleaned = _clean_output(str(final_answer))
    text = _strip_accents(cleaned)
    required = [
        "Sante du Systeme",
        "Analyse WTI",
        "Le Debat des Agents",
        "Risk Manager",
    ]
    missing = [s for s in required if _strip_accents(s).lower() not in text.lower()]
    if missing:
        raise ValueError(f"Missing sections: {missing}")
    return True


def main():
    project_root = str(Path(__file__).resolve().parents[1])
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import requests

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        logger.info("Ollama server is reachable.")
    except Exception:
        logger.error(
            "Ollama server is not reachable at http://localhost:11434. "
            "Start it with 'ollama serve' and try again."
        )
        sys.exit(1)

    from smolagents import CodeAgent, LiteLLMModel

    from morning_brief.tools.analyze_trading_logs import AnalyzeTradingLogsTool
    from morning_brief.tools.audit_portfolio_performance import AuditPortfolioPerformanceTool
    from morning_brief.tools.analyze_wti_market import AnalyzeWtiMarketTool
    from morning_brief.tools.analyze_nasdaq import AnalyzeNasdaqTool
    from morning_brief.tools.analyze_market_sentiment import AnalyzeMarketSentimentTool

    today = datetime.now().strftime("%Y-%m-%d")
    template = MARKDOWN_TEMPLATE.format(date=today)

    model = LiteLLMModel(
        model_id="ollama_chat/hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K",
        api_base="http://localhost:11434",
        num_ctx=16384,
        timeout=1200,
    )

    agent = CodeAgent(
        tools=[
            AnalyzeTradingLogsTool(),
            AuditPortfolioPerformanceTool(),
            AnalyzeWtiMarketTool(),
            AnalyzeNasdaqTool(),
            AnalyzeMarketSentimentTool(),
        ],
        model=model,
        instructions=DEBATE_INSTRUCTIONS,
        additional_authorized_imports=[
            "json", "datetime", "re", "pathlib",
            "math", "statistics", "collections", "itertools",
            "unicodedata", "logging",
        ],
        max_steps=6,
        planning_interval=None,
        use_structured_outputs_internally=True,
        final_answer_checks=[validate_markdown_output],
        executor_kwargs={"timeout_seconds": 120},
    )

    task = (
        f"Genere le Morning Market Brief du {today}.\n"
        f"Step 1: Appelle les 5 outils dans un seul bloc de code.\n"
        f"Step 2: Synthetise le debat des 3 personas (Bull/Bear/Risk Manager).\n"
        f"Step 3: Retourne le markdown complet via final_answer().\n\n"
        f"Template obligatoire:\n{template}"
    )

    logger.info("Starting Morning Market Brief generation...")
    result = agent.run(task)
    logger.info("Morning Market Brief generation complete.")

    output_path = OUTPUT_DIR / "morning_market_brief.md"
    md_content = _clean_output(str(result))
    output_path.write_text(md_content, encoding="utf-8")

    tools_dir = OUTPUT_DIR / "tools"
    summary_path = tools_dir / "full_summary.json"
    summary_data = {"date": today, "validation": "PASS"}
    try:
        validate_markdown_output(md_content, None, None)
    except ValueError as e:
        summary_data["validation"] = f"PARTIAL: {e}"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    print(f"\nOutput: {output_path}")
    print(f"Tool data: {tools_dir}")
    print(f"Validation: {summary_data['validation']}")


if __name__ == "__main__":
    main()

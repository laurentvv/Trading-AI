import json
import logging
import sys
import unicodedata
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tools").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            OUTPUT_DIR / "morning_brief.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("morning_brief")

DEBATE_INSTRUCTIONS = """Tu es un comité d'investissement à 3 voix pour Trading-AI. Analyse les données collectées par les outils et produis un Morning Market Brief rigoureux et structuré en Markdown.

ÉTAPE 1 - Voix THE BULL :
Analyse TOUS les arguments haussiers : supports WTI tenus, moyennes mobiles, RSI sain, logs système nominaux, macro/sentiment favorables.

ÉTAPE 2 - Voix THE BEAR :
Recherche ACTIVEMENT les failles et risques : surachat WTI, volatilité, slippage éventuel, signaux d'alertes, macro incertaine, divergences.

ÉTAPE 3 - Voix RISK MANAGER (Décision Finale) :
Arbitre le débat en te basant sur le drawdown actuel du portefeuille et l'état des risques.
Produis la recommandation finale, le biais recommandé (Bull / Bear / Neutral) et le position sizing optimal.

Format de sortie obligatoire : Markdown strict respectant exactement le template demandé."""

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


def validate_markdown_output(final_answer):
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


async def generate_brief_async() -> str:
    from nexusai_client import AIGateway
    from morning_brief.tools.analyze_trading_logs import AnalyzeTradingLogsTool
    from morning_brief.tools.audit_portfolio_performance import AuditPortfolioPerformanceTool
    from morning_brief.tools.analyze_wti_market import AnalyzeWtiMarketTool
    from morning_brief.tools.analyze_nasdaq import AnalyzeNasdaqTool
    from morning_brief.tools.analyze_market_sentiment import AnalyzeMarketSentimentTool

    logger.info("Executing Morning Brief tools...")
    logs_res = AnalyzeTradingLogsTool().forward()
    port_res = AuditPortfolioPerformanceTool().forward()
    wti_res = AnalyzeWtiMarketTool().forward()
    nasdaq_res = AnalyzeNasdaqTool().forward()
    sent_res = AnalyzeMarketSentimentTool().forward()

    tools_summary = f"""### DONNÉES SYSTÈME ET MARCHÉ COLLECTÉES :

1. LOGS & SYSTÈME :
{logs_res}

2. AUDIT PORTEFEUILLE :
{port_res}

3. ANALYSE WTI (PÉTROLE) :
{wti_res}

4. ANALYSE NASDAQ :
{nasdaq_res}

5. SENTIMENT & MACRO :
{sent_res}
"""

    today = datetime.now().strftime("%Y-%m-%d")
    template = MARKDOWN_TEMPLATE.format(date=today)

    prompt = f"""Tu es le rédacteur en chef et comité d'investissement de Trading-AI.
Aujourd'hui nous sommes le {today}.

{tools_summary}

CONSIGNE :
Rédige le Morning Market Brief complet en suivant rigoureusement le template Markdown ci-dessous. Remplis chaque section avec les données réelles fournies ci-dessus.

Template Markdown obligatoire :
{template}
"""

    logger.info("Querying NexusAI Gateway for Morning Brief synthesis...")
    async with AIGateway.auto_fallback() as client:
        resp = await client.generate_text(
            prompt,
            system_prompt=DEBATE_INSTRUCTIONS,
            temperature=0.3,
            max_tokens=4096,
            json_mode=False,
        )
        logger.info(f"Morning Brief received from [{resp.provider} / {resp.model}].")
        return _clean_output(resp.text)


def main():
    project_root = str(Path(__file__).resolve().parents[1])
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("Starting Morning Market Brief generation via NexusAI-Client...")

    try:
        md_content = asyncio.run(generate_brief_async())
    except Exception as e:
        logger.error(f"Error during Morning Brief generation: {e}")
        md_content = f"# Morning Market Brief — {today}\n\n*Erreur lors de la génération automatique: {e}*\n"

    output_path = OUTPUT_DIR / "morning_market_brief.md"
    output_path.write_text(md_content, encoding="utf-8")

    tools_dir = OUTPUT_DIR / "tools"
    summary_path = tools_dir / "full_summary.json"
    summary_data = {"date": today, "validation": "PASS"}
    try:
        validate_markdown_output(md_content)
    except ValueError as e:
        summary_data["validation"] = f"PARTIAL: {e}"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    print(f"\nOutput: {output_path}")
    print(f"Tool data: {tools_dir}")
    print(f"Validation: {summary_data['validation']}")


if __name__ == "__main__":
    main()

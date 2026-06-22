import asyncio
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

import sys

# Ensure the project root is in sys.path when running as a subprocess
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importer les composants
from src.core.memory import FinancialMemory
from src.agents.annotator import AnnotatorAgent
from src.agents.solver import SolverAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("finacumen")


async def run_finacumen_pipeline(symbol: str, context_details: dict):
    """
    Exécute le pipeline complet de FinAcumen.
    """
    logger.info(f"Démarrage de l'analyse FinAcumen pour {symbol}")

    # 1. Préparation de la mémoire
    memory = FinancialMemory()
    # Dans un environnement de test/init, on s'assure d'avoir les données de départ
    memory.seed_initial_memories()

    # 2. Construction du contexte
    current_context = f"Analyze {symbol}. Details: {json.dumps(context_details, default=str)}"

    # 3. Récupération de la mémoire
    logger.info("Interrogation de la Financial Memory...")
    memory_block = memory.retrieve(current_context)

    # 4. Agent Annotateur
    logger.info("Génération des directives par l'Annotator Agent...")
    annotator = AnnotatorAgent()
    directives = annotator.run_annotator(memory_block, current_context)

    # 5. Agent Solveur (Boucle ReAct)
    logger.info("Démarrage du raisonnement par le Solver Agent...")
    solver = SolverAgent()
    result = solver.run_react_loop(current_context, memory_block, directives)

    logger.info("Analyse FinAcumen terminée.")

    return result


async def main():
    parser = argparse.ArgumentParser(description="Run FinAcumen Experience Memory Analysis")
    parser.add_argument(
        "--ticker",
        type=str,
        default="WTI",
        help="Ticker to analyze (default: WTI)",
    )
    args = parser.parse_args()

    # TODO: Intégrer de vraies données de marché via src/data.py
    context = {"price_trend": "Unknown", "volatility": "Normal", "date": datetime.now().strftime("%Y-%m-%d")}

    result = await run_finacumen_pipeline(args.ticker, context)

    # Sauvegarder la trace complète pour le journal
    output_dir = Path("data_cache/finacumen")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fichier d'état lu par main.py
    import re

    safe_ticker = re.sub(r"[^a-zA-Z0-9_=^.-]", "", args.ticker)
    state_file = output_dir / f"finacumen_{safe_ticker}.json"

    if result.get("status") == "success" and "decision" in result:
        decision = result["decision"]
        state_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": args.ticker,
            "signal": decision.get("action", "HOLD"),
            "confidence": decision.get("confidence", 0.0),
            "analysis": decision.get("reasoning", "Aucune analyse fournie."),
            "status": "success",
        }
    else:
        state_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "ticker": args.ticker,
            "signal": "HOLD",
            "confidence": 0.0,
            "analysis": f"FinAcumen n'a pas pu converger: {result.get('reason', 'Erreur inconnue')}",
            "status": result.get("status", "error"),
        }

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)

    print(f"\n--- RÉSULTAT FINACUMEN ({args.ticker}) ---")
    print(json.dumps(state_data, indent=2))
    print(f"Enregistré dans: {state_file}")

    if "trajectory" in result:
        with open(output_dir / f"trajectory_{safe_ticker}.txt", "w", encoding="utf-8") as f:
            for step in result["trajectory"]:
                f.write(step + "\n")
        print(f"Trajectory sauvegardée dans: {output_dir / f'trajectory_{safe_ticker}.txt'}")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

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
    # Dans un environnement de test, on s'assure d'avoir les données de départ
    memory.seed_initial_memories()

    # 2. Construction du contexte
    current_context = f"Analyze {symbol}. Details: {json.dumps(context_details)}"

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

    # Enregistrer le résultat dans la mémoire pour un apprentissage futur (Optionnel, ici simplifié)
    if result["status"] == "success":
        # action_taken = result["decision"]
        # En mode réel, on évaluerait le trade après T jours avant de l'ajouter en tant que Findings ou Cautions
        pass

    return result


async def main():
    # Exemple d'exécution
    context = {
        "price_trend": "Down",
        "recent_news": "EIA reports unexpected massive inventory build",
        "volatility": "High",
    }

    result = await run_finacumen_pipeline("WTI", context)

    # Sauvegarder la trace pour le journal
    output_dir = Path("logs_prod")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"finacumen_trace_{timestamp}.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n--- RÉSULTAT FINACUMEN ---")
    print(json.dumps(result.get("decision", result), indent=2))


if __name__ == "__main__":
    asyncio.run(main())

import json
import logging
from typing import Dict, Any

from src.core.tools import NumericalReasoningEngine, AnswerConsolidationGate
from src.llm_client import TEXT_LLM_MODEL, SCHEMA_FINACUMEN_SOLVER, OLLAMA_API_URL, _strip_thinking_prefix

logger = logging.getLogger(__name__)


class SolverAgent:
    """
    Main ReAct loop orchestrating the Numerical Reasoning Engine and generating final decisions.
    """

    def __init__(self, model_name: str = TEXT_LLM_MODEL, max_iterations: int = 6):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.engine = NumericalReasoningEngine()

    def run_react_loop(self, context: str, memory_block: str, annotator_directives: str) -> Dict[str, Any]:
        """
        Executes a ReAct loop (Reasoning + Acting) to solve the trading problem.
        """
        trajectory = []

        system_prompt = f"""
<Memory_Handling>
{memory_block}
</Memory_Handling>

<Available_Tools>
You have a python execution environment. ``pd`` (pandas) and ``np`` (numpy) are
ALREADY IMPORTED for you — DO NOT write any ``import`` statement (imports are
disabled and will raise ImportError).

The single data tool is:
    lookup_ohlc(symbol, date, indicators)
where:
  - symbol: a yfinance ticker (e.g. "CRUDP.PA", "SXRV.DE", "CL=F") OR an alias
            ("WTI", "NASDAQ", "NDX", "BRENT", "SP500").
  - date:   "latest" or "YYYY-MM-DD".
  - indicators: a LIST of strings. The function returns a dict
            {{indicator: value}} for the requested date.
Supported indicators:
  - raw price : "open", "high", "low", "close", "volume"
  - derived   : "vwap", "rsi" (14-period Wilder), "sma_50", "sma_200",
                "ema_12", "ema_26", "macd"

Example (correct usage):
    data = lookup_ohlc("CRUDP.PA", "latest", ["close", "rsi", "sma_50", "sma_200"])
    price = data["close"]; rsi = data["rsi"]
DO NOT pass a single string. DO NOT redefine lookup_ohlc yourself. DO NOT use
``import``. Unknown indicators return None in the dict.
</Available_Tools>

<Think_Steps>
### 1. Input Inventory (MANDATORY --- DO NOT proceed to calculation)
Call ``lookup_ohlc`` ONCE with a LIST of the indicators you need and ALWAYS
``print()`` the returned dict so you can see the values. Example:
    data = lookup_ohlc("CRUDP.PA", "latest", ["close", "rsi", "sma_50", "sma_200"])
    print(data)
### 2. Experience Applicability
{annotator_directives}
### 3. Problem Understanding
Define the expected output type (BUY, SELL, HOLD).
### 4. Decide (MANDATORY --- DO NOT loop)
After you have the data, MOVE ON to the final answer. Do not call
``lookup_ohlc`` more than once. Decide BUY/SELL/HOLD with a concrete
confidence and a reasoning that cites the fetched numbers.
### 5. Next Action
Finish with the final decision (do NOT fetch data repeatedly).
</Think_Steps>

<Invariants>
- "data not available" is never true. Request data via the lookup_ohlc tool in python, then print it.
- NEVER write an ``import`` statement. ``pd`` and ``np`` are pre-imported.
- NEVER redefine lookup_ohlc. Always call it with a LIST of indicators, and call it AT MOST ONCE.
- NEVER submit placeholder text or vague explanations as final_answer. Only a clear and traceable decision is accepted.
- To execute python, return a JSON EXACTLY like this: {{"python_code": "your code here", "action": "NONE", "confidence": 0.0, "reasoning": ""}}
- To return the final answer, return a JSON EXACTLY like this: {{"python_code": "", "action": "BUY|SELL|HOLD", "confidence": 0.8, "reasoning": "your reasoning"}}
- You MUST provide ALL 4 keys ("python_code", "action", "confidence", "reasoning") in every response. Use empty strings or "NONE"/0.0 for fields you are not using.
</Invariants>
...never add a 'thought' key.
"""
        import requests

        current_prompt = f"Context:\n{context}\n\nStart your reasoning and next action following the invariants."

        for iteration in range(self.max_iterations):
            logger.info(f"Solver Iteration {iteration + 1}/{self.max_iterations}")

            payload = {
                "model": self.model_name,
                "prompt": current_prompt,
                "stream": False,
                "format": SCHEMA_FINACUMEN_SOLVER,
                "options": {"temperature": 0.1, "num_predict": 2048},
                "system": system_prompt,
            }

            try:
                response = requests.post(OLLAMA_API_URL, json=payload, timeout=1800)
                response.raise_for_status()

                raw_output = response.json().get("response", "").strip()
                clean_output = _strip_thinking_prefix(raw_output)
                trajectory.append(f"Model: {clean_output}")

                # Parse JSON
                try:
                    action_data = json.loads(clean_output)
                except json.JSONDecodeError:
                    current_prompt += "\n\nModel Output Error: Invalid JSON. Please output strictly valid JSON according to invariants."
                    continue

                # Le schema impose toujours les 4 cles. On distingue :
                #  - demande d'execution : python_code NON vide
                #  - reponse finale     : python_code vide ET action != NONE
                wants_execute = bool(str(action_data.get("python_code", "")).strip())
                action_val = str(action_data.get("action", "")).upper()
                is_final_answer = action_val in ("BUY", "SELL", "HOLD")

                if wants_execute:
                    code = action_data["python_code"]
                    trajectory.append(f"Action: execute_python({code})")

                    result = self.engine.execute(code)
                    output_str = f"Execution result:\nStdout:\n{result['output']}\n"
                    if result["error"]:
                        output_str += f"Error:\n{result['error']}\n"

                    # Si le code a simplement assigné des donnees sans les afficher
                    # (cas typique: ``data = lookup_ohlc(...)``), on les renvoie
                    # explicitement au modele, sinon il boucle sans jamais voir
                    # ce qu'il a recupere (cause racine des timeouts prod).
                    ns = self.engine.namespace
                    if not result["output"]:
                        fetched = ns.get("data")
                        if isinstance(fetched, dict):
                            output_str += f"Fetched data:\n{fetched}\n"
                        elif fetched is not None:
                            output_str += f"Fetched data:\n{repr(fetched)}\n"

                    trajectory.append(f"Observation: {output_str}")
                    current_prompt += f"\n\nModel:\n{clean_output}\n\nObservation:\n{output_str}"

                elif is_final_answer and "confidence" in action_data and "reasoning" in action_data:
                    # Final Answer
                    trajectory.append(f"Action: final_answer({action_data})")

                    # Validate via the gate
                    validation = AnswerConsolidationGate.verify(trajectory, action_data)
                    if validation["valid"]:
                        return {"status": "success", "decision": action_data, "trajectory": trajectory}
                    else:
                        error_msg = f"Answer consolidation failed: {validation['reason']}. Adjust your reasoning."
                        trajectory.append(f"Observation: {error_msg}")
                        current_prompt += f"\n\nModel:\n{clean_output}\n\nObservation:\n{error_msg}"
                else:
                    error_msg = (
                        "Invalid JSON structure. Must contain 'python_code' OR 'action', 'confidence', 'reasoning'."
                    )
                    trajectory.append(f"Observation: {error_msg}")
                    current_prompt += f"\n\nModel:\n{clean_output}\n\nObservation:\n{error_msg}"

            except Exception as e:
                logger.error(f"Solver loop error: {e}")
                return {"status": "error", "reason": str(e), "trajectory": trajectory}

        return {
            "status": "timeout",
            "reason": f"Max iterations ({self.max_iterations}) reached.",
            "trajectory": trajectory,
        }

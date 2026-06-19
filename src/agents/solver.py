import json
import logging
from typing import Dict, Any

from src.core.tools import NumericalReasoningEngine, AnswerConsolidationGate
from src.llm_client import TEXT_LLM_MODEL, OLLAMA_API_URL, _strip_thinking_prefix

logger = logging.getLogger(__name__)


class SolverAgent:
    """
    Main ReAct loop orchestrating the Numerical Reasoning Engine and generating final decisions.
    """

    def __init__(self, model_name: str = TEXT_LLM_MODEL, max_iterations: int = 16):
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

<Think_Steps>
### 1. Input Inventory (MANDATORY --- DO NOT proceed to calculation)
Take a complete inventory of market data before any tool call. List each value (e.g., opening price, VWAP, RSI) with its unit.
### 2. Experience Applicability
{annotator_directives}
### 3. Problem Understanding
Define the expected output type (BUY, SELL, HOLD).
### 4. Strategy
Write ALL values as python variables into the Numerical Reasoning Engine before evaluating the trade condition.
You have access to a python execution environment. You can return python code to be executed by returning a JSON object with "python_code" key.
The python code can call `lookup_ohlc(symbol, date, indicator)`.
### 5. Next Action
Call a tool or finish with the final decision.
</Think_Steps>

<Invariants>
- "data not available" is never true. Request data via lookup_ohlc tool in python.
- NEVER submit placeholder text or vague explanations as final_answer. Only a clear and traceable decision is accepted.
- To execute python, return a JSON: {{"python_code": "your code here"}}
- To return the final answer, return a JSON: {{"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reasoning": "your reasoning"}}
</Invariants>
"""
        import requests

        current_prompt = f"Context:\n{context}\n\nStart your reasoning and next action following the invariants."

        for iteration in range(self.max_iterations):
            logger.info(f"Solver Iteration {iteration + 1}/{self.max_iterations}")

            payload = {
                "model": self.model_name,
                "prompt": current_prompt,
                "stream": False,
                "format": "json",  # Relaxed json structure since it can be python_code OR final answer
                "options": {"temperature": 0.1, "num_predict": 1024},
                "system": system_prompt,
            }

            try:
                response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
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

                if "python_code" in action_data:
                    code = action_data["python_code"]
                    trajectory.append(f"Action: execute_python({code})")

                    result = self.engine.execute(code)
                    output_str = f"Execution result:\nStdout:\n{result['output']}\n"
                    if result["error"]:
                        output_str += f"Error:\n{result['error']}\n"

                    trajectory.append(f"Observation: {output_str}")
                    current_prompt += f"\n\nModel:\n{clean_output}\n\nObservation:\n{output_str}"

                elif "action" in action_data and "confidence" in action_data and "reasoning" in action_data:
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

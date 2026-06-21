import logging

# Adjusting import to use existing LLM integration
from src.llm_client import TEXT_LLM_MODEL, _query_ollama, SCHEMA_FINACUMEN_ANNOTATOR

logger = logging.getLogger(__name__)


class AnnotatorAgent:
    """
    Translates raw experiences (memories) into actionable directives (MUST/DO NOT)
    adapted to the current market context.
    """

    def __init__(self, model_name: str = TEXT_LLM_MODEL):
        self.model_name = model_name

    def run_annotator(self, memory_block: str, current_market_context: str) -> str:
        """
        Takes the XML <Memory_Handling> block and returns strict directives.
        """
        # If no memories were retrieved, return an empty string
        if "No relevant memories" in memory_block or "No memories available" in memory_block:
            return "<!-- No historical directives applicable for the current context -->"

        prompt = f"""
        Act as a strict Financial Methods Annotator.

        You will be provided with a set of historical experiences (<Memory_Handling>) and the current market context.

        CURRENT MARKET CONTEXT:
        {current_market_context}

        HISTORICAL EXPERIENCES:
        {memory_block}

        YOUR TASK:
        For each Entry retrieved, write 1 to 3 directive sentences using MUST or DO NOT.
        Extract the METHODOLOGY (extraction model, calculation logic, verification) and adapt it to the current market context.
        RULE: Map the METHODOLOGY of the memory, not its literal entity names. Adapt the old rules to today's data.

        Output ONLY the directives. Do not include any introductory or concluding text.
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt.strip(),
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
            "system": "<|think|> You are a strict rules annotator. Never add a 'thought' key.",
        }

        # We don't use schema for this one as we want free text, but we use the _query_ollama helper
        # by bypassing the json strictness, or we can just make a direct request.
        import requests
        from src.llm_client import OLLAMA_API_URL, _strip_thinking_prefix

        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=1800)
            response.raise_for_status()

            raw_output = response.json().get("response", "").strip()
            # Remove thinking tokens
            clean_output = _strip_thinking_prefix(raw_output)

            return clean_output.strip()

        except Exception as e:
            logger.error(f"AnnotatorAgent failed: {e}")
            return f"<!-- Error generating directives: {e} -->"

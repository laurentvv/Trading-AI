import logging
from nexusai_client import AIGateway
from src.llm_client import _run_sync, strip_thinking_debris

logger = logging.getLogger(__name__)


class AnnotatorAgent:
    """
    Translates raw experiences (memories) into actionable directives (MUST/DO NOT)
    adapted to the current market context.
    """

    def __init__(self, model_name: str = "nexusai"):
        self.model_name = model_name

    def run_annotator(self, memory_block: str, current_market_context: str) -> str:
        """
        Takes the XML <Memory_Handling> block and returns strict directives.
        """
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

        async def _generate():
            async with AIGateway.auto_fallback() as client:
                resp = await client.generate_text(
                    prompt.strip(),
                    system_prompt="You are a strict rules annotator.",
                    temperature=0.2,
                    max_tokens=512,
                    json_mode=False,
                )
                return strip_thinking_debris(resp.text.strip())

        try:
            return _run_sync(_generate())
        except Exception as e:
            logger.error(f"AnnotatorAgent failed: {e}")
            return f"<!-- Error generating directives: {e} -->"

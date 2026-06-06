from typing import Dict

# Configuration centralisée des poids de base
# Utilisée par AdaptiveWeightManager et EnhancedDecisionEngine
# pour éviter la dérive de configuration (Duplication code issue)
DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    "classic": 0.13,
    "llm_text": 0.21,
    "llm_visual": 0.19,
    "sentiment": 0.16,
    "timesfm": 0.20,
    "vincent_ganne": 0.05,
    "oil_bench": 0.05,
    "tensortrade": 0.05,
    "grebenkov": 0.05,  # From enhanced_decision_engine.py
    "hmm_model": 0.05,
}


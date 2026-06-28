from typing import Dict

# Configuration centralisée des poids de base
# Utilisée par AdaptiveWeightManager et EnhancedDecisionEngine
# pour éviter la dérive de configuration (Duplication code issue)
#
# Repondération ADR-002 (juin 2026) fondée sur l'edge_buy observé en prod
# (rendement moyen des jours BUY vs rendement marché), PAS sur le win_rate
# qui mesurait le marché. Période 29/05-25/06, marché baissier (78% jours ↓).
#
# edge_buy observé par modèle (négatif = ses BUY détruisent de la valeur):
#   sentiment      +0.0166   llm_visual  -0.0016
#   oil_bench      +0.0093   tensortrade -0.0047
#   classic        +0.0038   hmm_model   -0.0023
#   timesfm        -0.0005   vincent_ganne -0.0071
#   (llm_text      -0.0141)  (grebenkov  -0.0089)
DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    "classic": 0.13,        # inchangé — quantitatif neutre
    "llm_text": 0.12,       # 0.21 -> 0.12 (edge_buy -0.014, le pire)
    "llm_visual": 0.16,     # 0.19 -> 0.16
    "sentiment": 0.16,      # inchangé (edge + mais biais de données amont)
    "timesfm": 0.15,        # 0.20 -> 0.15 (0 SELL avant correctif Groupe 2)
    "vincent_ganne": 0.02,  # 0.05 -> 0.02 (edge -0.007)
    "oil_bench": 0.08,      # 0.05 -> 0.08 (edge +0.009, seul SELL rentable)
    "tensortrade": 0.04,    # 0.05 -> 0.04 (confiance non calibrée, cap ajouté)
    "grebenkov": 0.05,      # inchangé (correctif de logique appliqué au Groupe 2)
    "hmm_model": 0.04,      # 0.05 -> 0.04
}
# Somme = 0.95 (volontairement < 1.0 ; les poids sont renormalisés à 1.0
# à l'usage dans EnhancedDecisionEngine.make_enhanced_decision et
# AdaptiveWeightManager.calculate_adaptive_weights).

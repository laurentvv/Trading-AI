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
# Repondération simplifiée (octobre 2026, accord utilisateur) :
# Quarantaine formelle des modèles zombies / morts (poids = 0.0) :
#   - tensortrade : 0.0 (RL 2000 pas sans fine-tune, politique non convergée)
#   - hmm_model : 0.0 (HMM discrétisé 2 états, 50/50 pile ou face)
#   - vincent_ganne : 0.0 (désactivé sur indices/actions, N/A permanent)
#   - sentiment : 0.0 (quota AV épuisé + filtre ticker Yahoo impossible)
# Réallocation des poids vers les modèles réels et viables :
#   - timesfm : 0.25 (fondation model, quantiles)
#   - classic : 0.20 (quantitatif)
#   - grebenkov : 0.20 (suivi de tendance EMA + parité de risque agnostique)
#   - llm_text : 0.15 (raisonnement qualitatif)
#   - llm_visual : 0.10 (structure technique charts)
#   - council : 0.10 (délibération hebdomadaire week-end)
#   - oil_bench : 0.00 (sur-pondéré dynamiquement sur les tickers énergie)
DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    "classic": 0.20,
    "timesfm": 0.25,
    "llm_text": 0.15,
    "llm_visual": 0.10,
    "grebenkov": 0.20,
    "council": 0.10,
    "oil_bench": 0.00,
    "tensortrade": 0.00,
    "hmm_model": 0.00,
    "vincent_ganne": 0.00,
    "sentiment": 0.00,
}
# Somme des modèles actifs = 1.00.


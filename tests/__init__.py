"""Suite de tests Trading-AI.

Le régulateur d'appels T212 (src/t212_rate_limit.py) espace réellement les requêtes ; pendant les
tests, aucune attente réelle n'est souhaitée (les requêtes HTTP sont mockées).
"""

import os

os.environ["T212_RATE_GATE_DISABLED"] = "1"  # forcé : une valeur héritée ≠ "1" rallongerait la suite de plusieurs minutes

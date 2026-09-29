"""Suite de tests Trading-AI.

Le régulateur d'appels T212 (src/t212_rate_limit.py) espace réellement les requêtes ; pendant les
tests, aucune attente réelle n'est souhaitée (les requêtes HTTP sont mockées).
"""

import os

os.environ.setdefault("T212_RATE_GATE_DISABLED", "1")

"""Banc de vérité (phase 1 du plan de passage en réel) : backtest quotidien sans donnée future.

Le moteur (`engine.run_backtest`) consomme une SÉRIE D'EXPOSITION CIBLE (0 à 1) décidée à la clôture du
jour t avec les seules données <= t, exécutée à l'ouverture de t+1, coûts et impôt inclus. Toute stratégie
(règle simple, modèle seul, ensemble) se branche donc de la même façon ; les références obligatoires
(buy & hold, MA200, 50 % fixe) sont dans `strategies`. Rapport : ``python -m src.backtest.report``.
"""

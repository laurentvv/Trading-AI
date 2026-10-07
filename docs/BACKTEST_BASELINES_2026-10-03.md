# Références de performance — généré le 2026-10-03

Sortie de `python -m src.backtest.report` (données : `data_cache/`). Chiffres bruts sans interprétation.

## Qualité des séries

| Série | Lignes | Période | Lignes gelées | Lignes vivantes finales | Vivante depuis |
|---|---|---|---|---|---|
| SXRV.DE | 1274 | 2021-10-04 → 2026-10-02 | 0 % | 1274 | 2021-10-04 |
| CRUDP.PA | 1278 | 2021-10-04 → 2026-10-02 | 82 % | 187 | 2026-01-08 |
| ^NDX | 1255 | 2021-10-04 → 2026-10-02 | 0 % | 1255 | 2021-10-04 |
| CL=F | 1257 | 2021-10-04 → 2026-10-02 | 0 % | 1257 | 2021-10-04 |

Une ligne « gelée » = volume nul et clôture recopiée de la veille (flux Yahoo factice). Une série majoritairement gelée ne peut pas être backtestée.

## SXRV.DE — ETF Nasdaq-100 en EUR (instrument tradé)

Fenêtre commune : 2022-07-15 → 2026-10-02 (1074 séances).

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 21.9 % | 1.12 | 1.64 | -26.7 % | 0.82 | 100 % | 0.2× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 11.0 % | 1.14 | 1.66 | -14.1 % | 0.78 | 100 % | 0.2× | 35 | 48.1 % | -2.1 % |
| Tendance MA100 | 14.9 % | 1.00 | 1.43 | -21.8 % | 0.68 | 77 % | 7.6× | 16 | 9.7 % | -2.3 % |
| Tendance MA200 | 15.2 % | 1.01 | 1.46 | -16.3 % | 0.93 | 81 % | 5.9× | 13 | 21.4 % | -2.0 % |
| Tendance MA200 (hystérésis 2 %) | 16.5 % | 1.09 | 1.57 | -15.0 % | 1.10 | 81 % | 1.9× | 4 | 31.9 % | -5.6 % |
| Momentum 12 mois | 18.0 % | 1.09 | 1.55 | -33.0 % | 0.55 | 80 % | 1.5× | 3 | 26.4 % | -2.1 % |
| Sorties actuelles, entrée toujours haussière | 8.8 % | 0.54 | 0.77 | -31.5 % | 0.28 | 92 % | 39.7× | 83 | 2.9 % | -2.9 % |
| Tendance MA200 + sorties actuelles | 3.7 % | 0.32 | 0.45 | -21.7 % | 0.17 | 75 % | 35.7× | 74 | 3.1 % | -2.4 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 57 338 € | 16.6 % | 11 716 € | -26.7 % |
| 50 % fixe (rééquilibré / 21 séances) | 41 462 € | 8.0 % | 4 912 € | -14.1 % |
| Tendance MA100 | 44 689 € | 9.9 % | 7 129 € | -23.1 % |
| Tendance MA200 | 44 671 € | 9.9 % | 7 825 € | -19.9 % |
| Tendance MA200 (hystérésis 2 %) | 46 583 € | 11.0 % | 8 510 € | -24.3 % |
| Momentum 12 mois | 49 684 € | 12.7 % | 8 436 € | -37.2 % |
| Sorties actuelles, entrée toujours haussière | 35 959 € | 4.4 % | 5 331 € | -30.5 % |
| Tendance MA200 + sorties actuelles | 31 976 € | 1.5 % | 2 804 € | -21.6 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | -15.0 % | 51.2 % | 33.5 % | 7.0 % | 25.7 % |
| 50 % fixe (rééquilibré / 21 séances) | -7.4 % | 23.4 % | 16.3 % | 3.9 % | 12.7 % |
| Tendance MA100 | -6.5 % | 28.4 % | 11.8 % | 9.3 % | 22.3 % |
| Tendance MA200 | -6.7 % | 24.6 % | 32.5 % | -0.4 % | 18.2 % |
| Tendance MA200 (hystérésis 2 %) | -6.5 % | 23.4 % | 33.5 % | 4.6 % | 18.1 % |
| Momentum 12 mois | 0.0 % | 28.4 % | 33.5 % | -4.8 % | 23.1 % |
| Sorties actuelles, entrée toujours haussière | -25.2 % | 41.9 % | 11.2 % | 3.6 % | 16.4 % |
| Tendance MA200 + sorties actuelles | -6.6 % | 11.2 % | 15.5 % | -6.7 % | 4.0 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.11 [-0.58 ; +0.84] | 0.92 |
| 50 % fixe (rééquilibré / 21 séances) | +0.01 [-0.01 ; +0.03] | +0.12 [-0.57 ; +0.85] | 0.92 |
| Tendance MA100 | -0.12 [-0.86 ; +0.66] | -0.01 [-0.57 ; +0.55] | 0.87 |
| Tendance MA200 | -0.11 [-0.84 ; +0.58] | — | 0.87 |
| Tendance MA200 (hystérésis 2 %) | -0.04 [-0.69 ; +0.62] | +0.07 [-0.11 ; +0.30] | 0.90 |
| Momentum 12 mois | -0.03 [-0.61 ; +0.65] | +0.08 [-0.73 ; +0.88] | 0.90 |
| Sorties actuelles, entrée toujours haussière | -0.58 [-0.89 ; -0.28] | -0.47 [-1.18 ; +0.25] | 0.57 |
| Tendance MA200 + sorties actuelles | -0.81 [-1.57 ; -0.06] | -0.70 [-0.95 ; -0.45] | 0.39 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 8 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 22.0 % / 1.13 | 22.0 % / 1.13 | 21.9 % / 1.12 | 21.9 % / 1.12 |
| Tendance MA200 (hystérésis 2 %) | 17.1 % / 1.12 | 16.9 % / 1.11 | 16.5 % / 1.09 | 16.2 % / 1.07 |
| Sorties actuelles, entrée toujours haussière | 20.1 % / 1.07 | 15.4 % / 0.86 | 8.8 % / 0.54 | 4.5 % / 0.33 |

## CL=F — contrat WTI continu (PROXY du pétrole, non tradable tel quel)

Fenêtre commune : 2022-07-21 → 2026-10-02 (1057 séances). Contrat à terme continu non ajusté du roll : les sauts de roll faussent les rendements. À lire comme un ordre de grandeur, jamais comme une performance atteignable.

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | -1.4 % | 0.16 | 0.23 | -44.0 % | -0.03 | 100 % | 0.3× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 0.4 % | 0.12 | 0.17 | -23.8 % | 0.02 | 100 % | 0.3× | 22 | 10.9 % | -15.5 % |
| Tendance MA100 | -9.8 % | -0.23 | -0.32 | -51.4 % | -0.19 | 36 % | 12.7× | 28 | 10.3 % | -3.3 % |
| Tendance MA200 | -9.7 % | -0.22 | -0.30 | -51.8 % | -0.19 | 32 % | 9.8× | 21 | 11.8 % | -3.4 % |
| Tendance MA200 (hystérésis 2 %) | -8.1 % | -0.15 | -0.21 | -51.8 % | -0.16 | 32 % | 4.3× | 9 | 6.2 % | -6.8 % |
| Momentum 12 mois | -3.6 % | 0.02 | 0.02 | -41.5 % | -0.09 | 34 % | 8.2× | 17 | 2.1 % | -3.4 % |
| Sorties actuelles, entrée toujours haussière | -12.5 % | -0.17 | -0.23 | -60.2 % | -0.21 | 90 % | 48.6× | 101 | 5.0 % | -6.8 % |
| Tendance MA200 + sorties actuelles | -12.6 % | -0.38 | -0.50 | -53.5 % | -0.24 | 29 % | 23.4× | 51 | 7.4 % | -5.7 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 28 211 € | -1.5 % | 0 € | -44.0 % |
| 50 % fixe (rééquilibré / 21 séances) | 30 070 € | 0.1 % | 393 € | -23.8 % |
| Tendance MA100 | 18 529 € | -10.8 % | 951 € | -51.4 % |
| Tendance MA200 | 18 564 € | -10.8 % | 961 € | -51.8 % |
| Tendance MA200 (hystérésis 2 %) | 19 750 € | -9.5 % | 1 184 € | -51.8 % |
| Momentum 12 mois | 23 893 € | -5.3 % | 1 722 € | -41.5 % |
| Sorties actuelles, entrée toujours haussière | 16 325 € | -13.5 % | 1 395 € | -59.9 % |
| Tendance MA200 + sorties actuelles | 17 129 € | -12.5 % | 491 € | -53.3 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | -17.0 % | -10.7 % | 0.1 % | -19.9 % | 58.8 % |
| 50 % fixe (rééquilibré / 21 séances) | -8.7 % | -4.6 % | 0.3 % | -10.3 % | 29.7 % |
| Tendance MA100 | 0.5 % | -9.7 % | -14.5 % | -29.4 % | 18.2 % |
| Tendance MA200 | -8.4 % | -5.5 % | -20.5 % | -21.0 % | 20.0 % |
| Tendance MA200 (hystérésis 2 %) | -6.2 % | -4.5 % | -13.7 % | -26.7 % | 23.8 % |
| Momentum 12 mois | -6.3 % | -21.7 % | -9.7 % | 0.5 % | 28.8 % |
| Sorties actuelles, entrée toujours haussière | -15.6 % | -25.3 % | -16.3 % | -18.7 % | 33.1 % |
| Tendance MA200 + sorties actuelles | -8.4 % | -7.5 % | -20.8 % | -21.0 % | 7.1 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.38 [-0.26 ; +1.13] | 0.41 |
| 50 % fixe (rééquilibré / 21 séances) | -0.04 [-0.08 ; -0.00] | +0.34 [-0.27 ; +1.07] | 0.38 |
| Tendance MA100 | -0.40 [-1.09 ; +0.18] | -0.01 [-0.57 ; +0.58] | 0.15 |
| Tendance MA200 | -0.38 [-1.13 ; +0.26] | — | 0.15 |
| Tendance MA200 (hystérésis 2 %) | -0.32 [-1.07 ; +0.33] | +0.06 [-0.18 ; +0.33] | 0.19 |
| Momentum 12 mois | -0.15 [-0.74 ; +0.42] | +0.23 [-0.24 ; +0.80] | 0.30 |
| Sorties actuelles, entrée toujours haussière | -0.33 [-0.73 ; +0.04] | +0.05 [-0.76 ; +0.92] | 0.18 |
| Tendance MA200 + sorties actuelles | -0.54 [-1.26 ; +0.13] | -0.16 [-0.65 ; +0.19] | 0.09 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 8 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | -1.3 % / 0.17 | -1.4 % / 0.16 | -1.4 % / 0.16 | -1.4 % / 0.16 |
| Tendance MA200 (hystérésis 2 %) | -7.1 % / -0.11 | -7.5 % / -0.13 | -8.1 % / -0.15 | -8.5 % / -0.17 |
| Sorties actuelles, entrée toujours haussière | -1.2 % / 0.16 | -5.9 % / 0.03 | -12.5 % / -0.17 | -16.6 % / -0.30 |

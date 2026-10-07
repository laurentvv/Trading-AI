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

## QDVR.DE — iShares S&P 500 Energy Sector UCITS ETF en EUR (candidat actions énergie T212)

Fenêtre commune : 2022-07-15 → 2026-10-01 (1072 séances). Actions des producteurs d'énergie US (Exxon, Chevron, ConocoPhillips). Zéro coût de roll, dividende réinvesti, 0.2 % de lignes gelées.

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 12.5 % | 0.87 | 1.22 | -23.9 % | 0.52 | 100 % | 0.2× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 6.3 % | 0.87 | 1.22 | -12.6 % | 0.50 | 100 % | 0.2× | 29 | 24.8 % | -0.3 % |
| Tendance MA100 | 2.6 % | 0.29 | 0.40 | -26.7 % | 0.10 | 75 % | 13.1× | 28 | 5.6 % | -1.7 % |
| Tendance MA200 | 3.4 % | 0.35 | 0.49 | -24.5 % | 0.14 | 74 % | 9.3× | 20 | 4.1 % | -1.9 % |
| Tendance MA200 (hystérésis 2 %) | 4.7 % | 0.46 | 0.64 | -21.4 % | 0.22 | 72 % | 2.4× | 5 | 18.3 % | -4.4 % |
| Momentum 12 mois | 8.3 % | 0.74 | 1.04 | -18.7 % | 0.44 | 73 % | 5.8× | 13 | 6.7 % | -1.7 % |
| Sorties actuelles, entrée toujours haussière | 1.8 % | 0.19 | 0.28 | -25.3 % | 0.07 | 92 % | 38.3× | 80 | 2.5 % | -2.3 % |
| Tendance MA200 + sorties actuelles | -4.5 % | -0.38 | -0.50 | -33.6 % | -0.13 | 69 % | 35.3× | 74 | 2.2 % | -2.0 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 43 449 € | 9.2 % | 5 764 € | -23.9 % |
| 50 % fixe (rééquilibré / 21 séances) | 36 141 € | 4.5 % | 2 632 € | -12.7 % |
| Tendance MA100 | 30 442 € | 0.3 % | 2 655 € | -26.7 % |
| Tendance MA200 | 31 411 € | 1.1 % | 2 773 € | -24.5 % |
| Tendance MA200 (hystérésis 2 %) | 32 674 € | 2.0 % | 3 280 € | -21.4 % |
| Momentum 12 mois | 37 634 € | 5.5 % | 3 777 € | -22.7 % |
| Sorties actuelles, entrée toujours haussière | 30 726 € | 0.6 % | 1 499 € | -25.9 % |
| Tendance MA200 + sorties actuelles | 23 803 € | -5.3 % | 899 € | -33.1 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | -4.4 % | 20.3 % | 20.3 % | -0.8 % | 19.9 % |
| 50 % fixe (rééquilibré / 21 séances) | -1.9 % | 9.9 % | 9.9 % | -0.2 % | 9.7 % |
| Tendance MA100 | -8.7 % | -6.4 % | 9.7 % | 2.1 % | 16.3 % |
| Tendance MA200 | -10.2 % | 0.4 % | 19.1 % | -9.7 % | 18.7 % |
| Tendance MA200 (hystérésis 2 %) | -10.0 % | -1.3 % | 20.3 % | -5.4 % | 19.9 % |
| Momentum 12 mois | -2.1 % | 10.9 % | 20.3 % | -7.4 % | 15.6 % |
| Sorties actuelles, entrée toujours haussière | -6.0 % | 1.1 % | 2.4 % | 1.7 % | 8.7 % |
| Tendance MA200 + sorties actuelles | -10.9 % | -4.5 % | -3.1 % | -12.4 % | 14.1 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.51 [-0.25 ; +1.29] | 0.70 |
| 50 % fixe (rééquilibré / 21 séances) | -0.00 [-0.01 ; +0.01] | +0.51 [-0.26 ; +1.29] | 0.70 |
| Tendance MA100 | -0.58 [-1.41 ; +0.09] | -0.07 [-0.68 ; +0.50] | 0.25 |
| Tendance MA200 | -0.51 [-1.29 ; +0.25] | — | 0.30 |
| Tendance MA200 (hystérésis 2 %) | -0.40 [-1.07 ; +0.22] | +0.11 [-0.29 ; +0.60] | 0.38 |
| Momentum 12 mois | -0.13 [-0.71 ; +0.45] | +0.39 [-0.28 ; +1.04] | 0.61 |
| Sorties actuelles, entrée toujours haussière | -0.68 [-1.17 ; -0.10] | -0.16 [-0.90 ; +0.57] | 0.19 |
| Tendance MA200 + sorties actuelles | -1.25 [-2.04 ; -0.54] | -0.73 [-1.09 ; -0.40] | 0.02 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 8 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 12.6 % / 0.87 | 12.6 % / 0.87 | 12.5 % / 0.87 | 12.5 % / 0.87 |
| Tendance MA200 (hystérésis 2 %) | 5.3 % / 0.52 | 5.1 % / 0.50 | 4.7 % / 0.46 | 4.4 % / 0.44 |
| Sorties actuelles, entrée toujours haussière | 12.0 % / 0.82 | 7.8 % / 0.57 | 1.8 % / 0.19 | -2.1 % / -0.06 |

## XDW0.DE — Xtrackers MSCI World Energy UCITS ETF en EUR (candidat actions énergie mondial T212)

Fenêtre commune : 2022-07-15 → 2026-10-01 (1072 séances). Majors énergétiques mondiales (Shell, TotalEnergies, BP, Exxon, Chevron). Zéro coût de roll, 0.2 % de lignes gelées.

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 15.5 % | 0.76 | 1.04 | -23.7 % | 0.65 | 100 % | 0.2× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 7.9 % | 0.74 | 1.00 | -12.3 % | 0.64 | 100 % | 0.2× | 31 | 26.3 % | 0.0 % |
| Tendance MA100 | -4.0 % | -0.15 | -0.20 | -42.0 % | -0.10 | 61 % | 14.7× | 31 | 5.9 % | -2.6 % |
| Tendance MA200 | 4.2 % | 0.31 | 0.43 | -38.4 % | 0.11 | 74 % | 8.4× | 18 | 6.5 % | -2.5 % |
| Tendance MA200 (hystérésis 2 %) | 4.4 % | 0.32 | 0.44 | -39.1 % | 0.11 | 72 % | 3.4× | 7 | 6.1 % | -5.9 % |
| Momentum 12 mois | 9.9 % | 0.60 | 0.84 | -23.0 % | 0.43 | 70 % | 10.6× | 23 | 2.7 % | -1.5 % |
| Sorties actuelles, entrée toujours haussière | 5.4 % | 0.36 | 0.48 | -23.4 % | 0.23 | 92 % | 39.7× | 83 | 3.2 % | -3.4 % |
| Tendance MA200 + sorties actuelles | 1.8 % | 0.19 | 0.26 | -38.5 % | 0.05 | 68 % | 35.1× | 72 | 3.3 % | -2.8 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 47 466 € | 11.5 % | 7 485 € | -23.7 % |
| 50 % fixe (rééquilibré / 21 séances) | 37 822 € | 5.7 % | 3 352 € | -12.5 % |
| Tendance MA100 | 23 108 € | -6.0 % | 1 967 € | -42.0 % |
| Tendance MA200 | 32 484 € | 1.9 % | 3 303 € | -38.2 % |
| Tendance MA200 (hystérésis 2 %) | 31 715 € | 1.3 % | 4 119 € | -40.8 % |
| Momentum 12 mois | 39 447 € | 6.7 % | 4 631 € | -23.2 % |
| Sorties actuelles, entrée toujours haussière | 34 769 € | 3.6 % | 2 894 € | -21.6 % |
| Tendance MA200 + sorties actuelles | 28 754 € | -1.0 % | 3 399 € | -36.8 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | 16.0 % | 0.2 % | 7.5 % | 2.2 % | 43.8 % |
| 50 % fixe (rééquilibré / 21 séances) | 8.1 % | 0.3 % | 4.1 % | 1.3 % | 20.5 % |
| Tendance MA100 | -11.8 % | -9.3 % | -6.4 % | -16.5 % | 34.5 % |
| Tendance MA200 | 16.0 % | -17.8 % | -1.9 % | -11.5 % | 43.8 % |
| Tendance MA200 (hystérésis 2 %) | 16.0 % | -7.2 % | -10.0 % | -13.8 % | 43.8 % |
| Momentum 12 mois | 10.8 % | -3.8 % | 4.1 % | -0.6 % | 35.1 % |
| Sorties actuelles, entrée toujours haussière | 5.9 % | -1.2 % | -7.1 % | 1.5 % | 26.7 % |
| Tendance MA200 + sorties actuelles | 6.7 % | -20.8 % | -4.8 % | -4.9 % | 40.7 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.45 [+0.05 ; +0.93] | 0.74 |
| 50 % fixe (rééquilibré / 21 séances) | -0.02 [-0.04 ; -0.01] | +0.42 [+0.03 ; +0.90] | 0.72 |
| Tendance MA100 | -0.91 [-1.61 ; -0.32] | -0.47 [-1.11 ; +0.15] | 0.11 |
| Tendance MA200 | -0.45 [-0.93 ; -0.05] | — | 0.40 |
| Tendance MA200 (hystérésis 2 %) | -0.44 [-0.90 ; -0.03] | +0.01 [-0.27 ; +0.30] | 0.40 |
| Momentum 12 mois | -0.16 [-0.69 ; +0.38] | +0.29 [-0.32 ; +0.91] | 0.63 |
| Sorties actuelles, entrée toujours haussière | -0.41 [-0.76 ; -0.03] | +0.04 [-0.43 ; +0.55] | 0.43 |
| Tendance MA200 + sorties actuelles | -0.57 [-1.19 ; -0.02] | -0.13 [-0.53 ; +0.24] | 0.30 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 8 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 15.6 % / 0.76 | 15.6 % / 0.76 | 15.5 % / 0.76 | 15.5 % / 0.76 |
| Tendance MA200 (hystérésis 2 %) | 5.4 % / 0.37 | 5.0 % / 0.35 | 4.4 % / 0.32 | 4.1 % / 0.30 |
| Sorties actuelles, entrée toujours haussière | 16.4 % / 0.82 | 11.9 % / 0.64 | 5.4 % / 0.36 | 1.4 % / 0.17 |

## CRUD.L — WisdomTree WTI Crude Oil ETC en USD (candidat matières premières T212)

Fenêtre commune : 2022-07-21 → 2026-10-01 (1061 séances). ETC pur sur contrats à terme WTI coté sur LSE. 0.0 % de lignes gelées. Subit le coût du roll en contango.

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 13.5 % | 0.55 | 0.75 | -27.3 % | 0.49 | 100 % | 0.2× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 7.9 % | 0.54 | 0.74 | -14.4 % | 0.55 | 100 % | 0.3× | 24 | 28.2 % | -5.2 % |
| Tendance MA100 | -1.5 % | 0.05 | 0.07 | -40.7 % | -0.04 | 48 % | 15.5× | 33 | 7.6 % | -2.9 % |
| Tendance MA200 | -9.5 % | -0.25 | -0.34 | -65.7 % | -0.14 | 53 % | 15.0× | 33 | 1.2 % | -3.3 % |
| Tendance MA200 (hystérésis 2 %) | -10.8 % | -0.31 | -0.42 | -68.7 % | -0.16 | 52 % | 7.7× | 16 | 0.0 % | -6.6 % |
| Momentum 12 mois | 2.9 % | 0.24 | 0.34 | -38.5 % | 0.08 | 56 % | 12.3× | 27 | 0.9 % | -2.4 % |
| Sorties actuelles, entrée toujours haussière | 4.2 % | 0.29 | 0.40 | -41.3 % | 0.10 | 91 % | 45.1× | 93 | 4.9 % | -4.4 % |
| Tendance MA200 + sorties actuelles | -13.6 % | -0.44 | -0.60 | -70.3 % | -0.19 | 50 % | 33.2× | 71 | 5.7 % | -4.0 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 44 603 € | 9.9 % | 6 259 € | -27.3 % |
| 50 % fixe (rééquilibré / 21 séances) | 37 738 € | 5.6 % | 3 375 € | -14.4 % |
| Tendance MA100 | 25 039 € | -4.2 % | 2 934 € | -40.7 % |
| Tendance MA200 | 17 070 € | -12.6 % | 2 591 € | -65.7 % |
| Tendance MA200 (hystérésis 2 %) | 15 927 € | -14.0 % | 2 558 € | -68.7 % |
| Momentum 12 mois | 29 749 € | -0.2 % | 4 067 € | -38.5 % |
| Sorties actuelles, entrée toujours haussière | 29 942 € | -0.0 % | 4 779 € | -41.1 % |
| Tendance MA200 + sorties actuelles | 14 039 € | -16.5 % | 1 917 € | -70.3 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | -11.5 % | 0.4 % | 9.3 % | -9.2 % | 92.6 % |
| 50 % fixe (rééquilibré / 21 séances) | -5.8 % | 1.1 % | 4.7 % | -3.8 % | 43.4 % |
| Tendance MA100 | -12.8 % | -9.8 % | -7.0 % | -17.0 % | 54.7 % |
| Tendance MA200 | -23.6 % | -6.4 % | -25.4 % | -31.1 % | 78.8 % |
| Tendance MA200 (hystérésis 2 %) | -22.2 % | -13.2 % | -26.9 % | -32.8 % | 86.1 % |
| Momentum 12 mois | -5.5 % | -16.6 % | -5.8 % | -9.0 % | 67.3 % |
| Sorties actuelles, entrée toujours haussière | -15.6 % | -16.1 % | 0.8 % | -9.4 % | 83.7 % |
| Tendance MA200 + sorties actuelles | -23.5 % | -12.6 % | -31.0 % | -30.9 % | 70.0 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.80 [+0.22 ; +1.43] | 0.49 |
| 50 % fixe (rééquilibré / 21 séances) | -0.01 [-0.05 ; +0.03] | +0.79 [+0.21 ; +1.40] | 0.48 |
| Tendance MA100 | -0.49 [-1.12 ; +0.04] | +0.31 [-0.25 ; +0.90] | 0.15 |
| Tendance MA200 | -0.80 [-1.43 ; -0.22] | — | 0.05 |
| Tendance MA200 (hystérésis 2 %) | -0.86 [-1.53 ; -0.27] | -0.06 [-0.29 ; +0.17] | 0.04 |
| Momentum 12 mois | -0.30 [-0.85 ; +0.23] | +0.49 [-0.06 ; +1.14] | 0.26 |
| Sorties actuelles, entrée toujours haussière | -0.26 [-0.54 ; +0.02] | +0.54 [-0.07 ; +1.19] | 0.29 |
| Tendance MA200 + sorties actuelles | -0.99 [-1.68 ; -0.35] | -0.19 [-0.46 ; +0.10] | 0.02 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 8 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 13.5 % / 0.55 | 13.5 % / 0.55 | 13.5 % / 0.55 | 13.4 % / 0.55 |
| Tendance MA200 (hystérésis 2 %) | -9.1 % / -0.24 | -9.8 % / -0.27 | -10.8 % / -0.31 | -11.5 % / -0.34 |
| Sorties actuelles, entrée toujours haussière | 16.5 % / 0.64 | 11.4 % / 0.50 | 4.2 % / 0.29 | -0.4 % / 0.15 |

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

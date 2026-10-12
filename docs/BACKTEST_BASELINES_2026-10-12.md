# Références de performance — généré le 2026-10-12

Sortie de `python -m src.backtest.report` (données : `data_cache/`). Chiffres bruts sans interprétation.

## Qualité des séries

| Série | Lignes | Période | Lignes gelées | Lignes vivantes finales | Vivante depuis |
|---|---|---|---|---|---|
| SXRV.DE | 1275 | 2021-10-12 → 2026-10-12 | 0 % | 1275 | 2021-10-12 |
| QDVF.DE | 1273 | 2021-10-12 → 2026-10-12 | 0 % | 1273 | 2021-10-12 |
| CRUDP.PA | 1279 | 2021-10-12 → 2026-10-12 | 82 % | 194 | 2026-01-08 |
| ^NDX | 1255 | 2021-10-11 → 2026-10-09 | 0 % | 1255 | 2021-10-11 |
| CL=F | 1257 | 2021-10-12 → 2026-10-12 | 0 % | 1257 | 2021-10-12 |

Une ligne « gelée » = volume nul et clôture recopiée de la veille (flux Yahoo factice). Une série majoritairement gelée ne peut pas être backtestée.

## SXRV.DE — ETF Nasdaq-100 en EUR (instrument tradé)

Fenêtre commune : 2022-07-25 → 2026-10-12 (1075 séances).

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 22.1 % | 1.14 | 1.66 | -26.7 % | 0.83 | 100 % | 0.2× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 11.1 % | 1.14 | 1.67 | -13.9 % | 0.80 | 100 % | 0.2× | 36 | 52.5 % | -4.7 % |
| Tendance MA100 | 15.6 % | 1.04 | 1.49 | -21.8 % | 0.71 | 78 % | 7.6× | 16 | 9.7 % | -2.1 % |
| Tendance MA200 | 15.5 % | 1.03 | 1.49 | -16.3 % | 0.95 | 81 % | 5.9× | 13 | 21.4 % | -2.0 % |
| Tendance MA200 (hystérésis 2 %) | 16.9 % | 1.11 | 1.60 | -15.0 % | 1.12 | 82 % | 1.9× | 4 | 31.9 % | -5.6 % |
| Momentum 12 mois | 18.4 % | 1.11 | 1.57 | -33.0 % | 0.56 | 80 % | 1.5× | 3 | 26.4 % | -2.1 % |
| Sorties actuelles, entrée toujours haussière | 8.8 % | 0.54 | 0.77 | -31.5 % | 0.28 | 92 % | 40.2× | 84 | 3.0 % | -2.9 % |
| Tendance MA200 + sorties actuelles | 3.8 % | 0.33 | 0.46 | -21.7 % | 0.18 | 75 % | 36.2× | 75 | 3.2 % | -2.4 % |
| Sorties swing, entrée toujours haussière | 19.6 % | 1.01 | 1.44 | -30.9 % | 0.64 | 99 % | 2.7× | 6 | 31.5 % | -10.8 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 16.2 % | 1.07 | 1.52 | -15.8 % | 1.03 | 82 % | 2.8× | 6 | 18.9 % | -4.0 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 57 691 € | 16.8 % | 11 868 € | -26.7 % |
| 50 % fixe (rééquilibré / 21 séances) | 41 515 € | 8.0 % | 4 935 € | -13.9 % |
| Tendance MA100 | 45 647 € | 10.5 % | 7 405 € | -23.1 % |
| Tendance MA200 | 45 149 € | 10.2 % | 8 030 € | -19.9 % |
| Tendance MA200 (hystérésis 2 %) | 47 082 € | 11.3 % | 8 723 € | -24.3 % |
| Momentum 12 mois | 50 217 € | 13.0 % | 8 664 € | -37.2 % |
| Sorties actuelles, entrée toujours haussière | 35 889 € | 4.3 % | 5 372 € | -30.7 % |
| Tendance MA200 + sorties actuelles | 32 138 € | 1.6 % | 2 873 € | -21.6 % |
| Sorties swing, entrée toujours haussière | 49 639 € | 12.7 % | 11 229 € | -32.7 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 45 929 € | 10.6 % | 8 230 € | -22.5 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | -15.6 % | 51.2 % | 33.5 % | 7.0 % | 27.5 % |
| 50 % fixe (rééquilibré / 21 séances) | -7.9 % | 23.4 % | 16.0 % | 4.2 % | 13.4 % |
| Tendance MA100 | -5.4 % | 28.4 % | 11.8 % | 9.3 % | 24.1 % |
| Tendance MA200 | -6.7 % | 24.6 % | 32.5 % | -0.4 % | 19.9 % |
| Tendance MA200 (hystérésis 2 %) | -6.5 % | 23.4 % | 33.5 % | 4.6 % | 19.8 % |
| Momentum 12 mois | 0.0 % | 28.4 % | 33.5 % | -4.8 % | 24.9 % |
| Sorties actuelles, entrée toujours haussière | -25.7 % | 41.9 % | 11.2 % | 3.6 % | 17.2 % |
| Tendance MA200 + sorties actuelles | -6.6 % | 11.2 % | 15.5 % | -6.7 % | 4.7 % |
| Sorties swing, entrée toujours haussière | -14.4 % | 51.2 % | 28.5 % | 0.9 % | 27.0 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -6.5 % | 23.4 % | 28.5 % | 6.3 % | 19.8 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.10 [-0.56 ; +0.86] | 0.92 |
| 50 % fixe (rééquilibré / 21 séances) | +0.01 [-0.01 ; +0.03] | +0.11 [-0.56 ; +0.87] | 0.92 |
| Tendance MA100 | -0.10 [-0.81 ; +0.63] | +0.01 [-0.57 ; +0.57] | 0.89 |
| Tendance MA200 | -0.10 [-0.86 ; +0.56] | — | 0.89 |
| Tendance MA200 (hystérésis 2 %) | -0.03 [-0.71 ; +0.59] | +0.07 [-0.12 ; +0.32] | 0.91 |
| Momentum 12 mois | -0.02 [-0.56 ; +0.62] | +0.08 [-0.72 ; +0.88] | 0.91 |
| Sorties actuelles, entrée toujours haussière | -0.59 [-0.90 ; -0.29] | -0.49 [-1.24 ; +0.30] | 0.58 |
| Tendance MA200 + sorties actuelles | -0.81 [-1.60 ; -0.13] | -0.71 [-0.98 ; -0.44] | 0.41 |
| Sorties swing, entrée toujours haussière | -0.12 [-0.31 ; +0.07] | -0.02 [-0.72 ; +0.79] | 0.88 |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -0.07 [-0.77 ; +0.59] | +0.03 [-0.18 ; +0.27] | 0.90 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 10 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 22.2 % / 1.14 | 22.2 % / 1.14 | 22.1 % / 1.14 | 22.1 % / 1.13 |
| Tendance MA200 (hystérésis 2 %) | 17.5 % / 1.14 | 17.2 % / 1.13 | 16.9 % / 1.11 | 16.6 % / 1.09 |
| Sorties actuelles, entrée toujours haussière | 20.2 % / 1.08 | 15.5 % / 0.87 | 8.8 % / 0.54 | 4.5 % / 0.33 |
| Sorties swing, entrée toujours haussière | 20.6 % / 1.06 | 20.2 % / 1.04 | 19.6 % / 1.01 | 19.3 % / 1.00 |

## QDVF.DE — iShares S&P 500 Energy Sector UCITS ETF en EUR (instrument énergie tradé, T212 QDVFd_EQ)

Fenêtre commune : 2022-07-25 → 2026-10-12 (1073 séances). Actions des producteurs d'énergie US (Exxon, Chevron, ConocoPhillips). Zéro coût de roll, dividende réinvesti.

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 15.1 % | 0.69 | 0.96 | -27.1 % | 0.56 | 100 % | 0.2× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 8.0 % | 0.69 | 0.94 | -14.4 % | 0.56 | 100 % | 0.3× | 31 | 23.3 % | 0.0 % |
| Tendance MA100 | -3.0 % | -0.07 | -0.09 | -44.1 % | -0.07 | 59 % | 15.7× | 33 | 5.8 % | -2.5 % |
| Tendance MA200 | 1.8 % | 0.19 | 0.26 | -47.3 % | 0.04 | 69 % | 12.3× | 27 | 4.0 % | -2.2 % |
| Tendance MA200 (hystérésis 2 %) | 3.0 % | 0.25 | 0.34 | -45.4 % | 0.07 | 66 % | 3.8× | 8 | 3.7 % | -6.0 % |
| Momentum 12 mois | 3.6 % | 0.28 | 0.38 | -27.6 % | 0.13 | 64 % | 8.4× | 18 | 2.7 % | -3.2 % |
| Sorties actuelles, entrée toujours haussière | 0.3 % | 0.13 | 0.18 | -43.1 % | 0.01 | 92 % | 40.7× | 85 | 3.7 % | -3.6 % |
| Tendance MA200 + sorties actuelles | -2.9 % | -0.05 | -0.07 | -57.0 % | -0.05 | 64 % | 38.6× | 79 | 3.8 % | -3.2 % |
| Sorties swing, entrée toujours haussière | 15.6 % | 0.72 | 0.99 | -24.7 % | 0.63 | 99 % | 3.8× | 8 | 12.7 % | -10.8 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 3.0 % | 0.25 | 0.34 | -45.5 % | 0.07 | 66 % | 5.8× | 12 | 11.6 % | -5.7 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 46 852 € | 11.2 % | 7 222 € | -27.1 % |
| 50 % fixe (rééquilibré / 21 séances) | 37 936 € | 5.7 % | 3 401 € | -14.4 % |
| Tendance MA100 | 23 865 € | -5.3 % | 2 293 € | -44.1 % |
| Tendance MA200 | 28 915 € | -0.9 % | 3 480 € | -47.2 % |
| Tendance MA200 (hystérésis 2 %) | 30 445 € | 0.3 % | 3 530 € | -45.3 % |
| Momentum 12 mois | 31 776 € | 1.4 % | 2 925 € | -27.6 % |
| Sorties actuelles, entrée toujours haussière | 28 006 € | -1.6 % | 2 457 € | -40.1 % |
| Tendance MA200 + sorties actuelles | 22 858 € | -6.2 % | 3 220 € | -54.8 % |
| Sorties swing, entrée toujours haussière | 44 764 € | 10.0 % | 8 449 € | -27.1 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 29 130 € | -0.7 % | 4 462 € | -47.2 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | 13.9 % | -3.7 % | 9.2 % | -2.7 % | 55.1 % |
| 50 % fixe (rééquilibré / 21 séances) | 7.4 % | -1.1 % | 4.9 % | -0.9 % | 25.4 % |
| Tendance MA100 | -7.7 % | -13.1 % | -3.8 % | -19.2 % | 40.8 % |
| Tendance MA200 | 13.9 % | -24.5 % | -5.3 % | -14.5 % | 55.1 % |
| Tendance MA200 (hystérésis 2 %) | 13.9 % | -12.0 % | -5.8 % | -22.6 % | 55.1 % |
| Momentum 12 mois | -0.5 % | -9.8 % | -5.1 % | -2.4 % | 39.4 % |
| Sorties actuelles, entrée toujours haussière | -0.6 % | -13.5 % | -5.9 % | -11.0 % | 40.4 % |
| Tendance MA200 + sorties actuelles | 7.6 % | -31.4 % | -10.6 % | -15.4 % | 58.0 % |
| Sorties swing, entrée toujours haussière | 13.5 % | -6.1 % | 8.3 % | 2.2 % | 56.5 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 13.5 % | -12.0 % | -6.4 % | -22.6 % | 56.5 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.50 [+0.09 ; +1.00] | 0.68 |
| 50 % fixe (rééquilibré / 21 séances) | -0.01 [-0.02 ; +0.01] | +0.50 [+0.09 ; +1.00] | 0.68 |
| Tendance MA100 | -0.76 [-1.41 ; -0.16] | -0.26 [-0.92 ; +0.38] | 0.14 |
| Tendance MA200 | -0.50 [-1.00 ; -0.09] | — | 0.29 |
| Tendance MA200 (hystérésis 2 %) | -0.45 [-0.93 ; -0.00] | +0.06 [-0.19 ; +0.32] | 0.33 |
| Momentum 12 mois | -0.42 [-1.02 ; +0.18] | +0.09 [-0.56 ; +0.75] | 0.35 |
| Sorties actuelles, entrée toujours haussière | -0.56 [-0.90 ; -0.23] | -0.06 [-0.62 ; +0.54] | 0.25 |
| Tendance MA200 + sorties actuelles | -0.74 [-1.30 ; -0.28] | -0.24 [-0.48 ; +0.01] | 0.15 |
| Sorties swing, entrée toujours haussière | +0.02 [-0.09 ; +0.14] | +0.53 [+0.10 ; +1.04] | 0.70 |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -0.45 [-0.93 ; -0.02] | +0.06 [-0.18 ; +0.32] | 0.33 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 10 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 15.1 % / 0.70 | 15.1 % / 0.70 | 15.1 % / 0.69 | 15.0 % / 0.69 |
| Tendance MA200 (hystérésis 2 %) | 4.1 % / 0.30 | 3.7 % / 0.28 | 3.0 % / 0.25 | 2.6 % / 0.23 |
| Sorties actuelles, entrée toujours haussière | 11.0 % / 0.55 | 6.6 % / 0.38 | 0.3 % / 0.13 | -3.7 % / -0.04 |
| Sorties swing, entrée toujours haussière | 16.8 % / 0.76 | 16.3 % / 0.74 | 15.6 % / 0.72 | 15.2 % / 0.70 |

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
| Sorties swing, entrée toujours haussière | 17.5 % | 0.83 | 1.15 | -23.0 % | 0.76 | 99 % | 3.3× | 7 | 11.9 % | -6.2 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 4.5 % | 0.33 | 0.45 | -38.1 % | 0.12 | 72 % | 4.8× | 10 | 11.0 % | -5.6 % |

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
| Sorties swing, entrée toujours haussière | 48 207 € | 11.9 % | 8 729 € | -23.9 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 31 333 € | 1.0 % | 4 371 € | -39.8 % |

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
| Sorties swing, entrée toujours haussière | 15.3 % | -0.1 % | 8.8 % | 9.8 % | 43.6 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 15.3 % | -7.2 % | -8.9 % | -13.8 % | 43.6 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.45 [+0.05 ; +0.93] | 0.72 |
| 50 % fixe (rééquilibré / 21 séances) | -0.02 [-0.04 ; -0.01] | +0.42 [+0.03 ; +0.90] | 0.70 |
| Tendance MA100 | -0.91 [-1.61 ; -0.32] | -0.47 [-1.11 ; +0.15] | 0.10 |
| Tendance MA200 | -0.45 [-0.93 ; -0.05] | — | 0.37 |
| Tendance MA200 (hystérésis 2 %) | -0.44 [-0.90 ; -0.03] | +0.01 [-0.27 ; +0.30] | 0.38 |
| Momentum 12 mois | -0.16 [-0.69 ; +0.38] | +0.29 [-0.32 ; +0.91] | 0.60 |
| Sorties actuelles, entrée toujours haussière | -0.41 [-0.76 ; -0.03] | +0.04 [-0.43 ; +0.55] | 0.40 |
| Tendance MA200 + sorties actuelles | -0.57 [-1.19 ; -0.02] | -0.13 [-0.53 ; +0.24] | 0.28 |
| Sorties swing, entrée toujours haussière | +0.07 [-0.02 ; +0.22] | +0.52 [+0.13 ; +1.00] | 0.76 |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -0.43 [-0.89 ; -0.03] | +0.01 [-0.25 ; +0.29] | 0.38 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 10 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 15.6 % / 0.76 | 15.6 % / 0.76 | 15.5 % / 0.76 | 15.5 % / 0.76 |
| Tendance MA200 (hystérésis 2 %) | 5.4 % / 0.37 | 5.0 % / 0.35 | 4.4 % / 0.32 | 4.1 % / 0.30 |
| Sorties actuelles, entrée toujours haussière | 16.4 % / 0.82 | 11.9 % / 0.64 | 5.4 % / 0.36 | 1.4 % / 0.17 |
| Sorties swing, entrée toujours haussière | 18.6 % / 0.87 | 18.2 % / 0.85 | 17.5 % / 0.83 | 17.1 % / 0.82 |

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
| Sorties swing, entrée toujours haussière | 11.5 % | 0.50 | 0.68 | -33.2 % | 0.35 | 98 % | 8.4× | 17 | 14.1 % | -8.4 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -11.5 % | -0.34 | -0.46 | -70.0 % | -0.16 | 52 % | 9.8× | 21 | 20.3 % | -6.8 % |

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
| Sorties swing, entrée toujours haussière | 39 473 € | 6.8 % | 6 841 € | -33.1 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 15 166 € | -15.0 % | 2 413 € | -70.0 % |

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
| Sorties swing, entrée toujours haussière | -13.8 % | -8.6 % | 6.9 % | -3.7 % | 94.7 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -22.2 % | -16.9 % | -26.9 % | -32.8 % | 88.1 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.80 [+0.22 ; +1.43] | 0.44 |
| 50 % fixe (rééquilibré / 21 séances) | -0.01 [-0.05 ; +0.03] | +0.79 [+0.21 ; +1.40] | 0.43 |
| Tendance MA100 | -0.49 [-1.12 ; +0.04] | +0.31 [-0.25 ; +0.90] | 0.12 |
| Tendance MA200 | -0.80 [-1.43 ; -0.22] | — | 0.04 |
| Tendance MA200 (hystérésis 2 %) | -0.86 [-1.53 ; -0.27] | -0.06 [-0.29 ; +0.17] | 0.03 |
| Momentum 12 mois | -0.30 [-0.85 ; +0.23] | +0.49 [-0.06 ; +1.14] | 0.22 |
| Sorties actuelles, entrée toujours haussière | -0.26 [-0.54 ; +0.02] | +0.54 [-0.07 ; +1.19] | 0.25 |
| Tendance MA200 + sorties actuelles | -0.99 [-1.68 ; -0.35] | -0.19 [-0.46 ; +0.10] | 0.01 |
| Sorties swing, entrée toujours haussière | -0.05 [-0.18 ; +0.09] | +0.75 [+0.14 ; +1.42] | 0.40 |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -0.89 [-1.58 ; -0.29] | -0.09 [-0.34 ; +0.14] | 0.02 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 10 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 13.5 % / 0.55 | 13.5 % / 0.55 | 13.5 % / 0.55 | 13.4 % / 0.55 |
| Tendance MA200 (hystérésis 2 %) | -9.1 % / -0.24 | -9.8 % / -0.27 | -10.8 % / -0.31 | -11.5 % / -0.34 |
| Sorties actuelles, entrée toujours haussière | 16.5 % / 0.64 | 11.4 % / 0.50 | 4.2 % / 0.29 | -0.4 % / 0.15 |
| Sorties swing, entrée toujours haussière | 13.9 % / 0.56 | 12.9 % / 0.54 | 11.5 % / 0.50 | 10.6 % / 0.47 |

## CL=F — contrat WTI continu (PROXY du pétrole, non tradable tel quel)

Fenêtre commune : 2022-07-29 → 2026-10-12 (1057 séances). Contrat à terme continu non ajusté du roll : les sauts de roll faussent les rendements. À lire comme un ordre de grandeur, jamais comme une performance atteignable.

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | -1.9 % | 0.15 | 0.21 | -44.0 % | -0.04 | 100 % | 0.3× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 0.8 % | 0.14 | 0.20 | -23.0 % | 0.04 | 100 % | 0.4× | 21 | 19.0 % | -15.7 % |
| Tendance MA100 | -9.8 % | -0.23 | -0.31 | -51.4 % | -0.19 | 37 % | 12.7× | 28 | 10.3 % | -3.3 % |
| Tendance MA200 | -10.1 % | -0.24 | -0.32 | -51.8 % | -0.19 | 32 % | 9.8× | 21 | 11.8 % | -3.5 % |
| Tendance MA200 (hystérésis 2 %) | -8.5 % | -0.17 | -0.23 | -51.8 % | -0.16 | 32 % | 4.3× | 9 | 6.2 % | -7.0 % |
| Momentum 12 mois | -3.8 % | 0.01 | 0.01 | -41.4 % | -0.09 | 34 % | 8.2× | 17 | 2.1 % | -3.5 % |
| Sorties actuelles, entrée toujours haussière | -13.0 % | -0.19 | -0.25 | -60.2 % | -0.22 | 90 % | 47.9× | 100 | 5.1 % | -6.9 % |
| Tendance MA200 + sorties actuelles | -13.0 % | -0.40 | -0.52 | -53.5 % | -0.24 | 29 % | 23.4× | 51 | 7.4 % | -5.8 % |
| Sorties swing, entrée toujours haussière | -5.2 % | 0.06 | 0.08 | -47.1 % | -0.11 | 98 % | 12.0× | 24 | 13.4 % | -11.1 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -9.0 % | -0.20 | -0.27 | -48.8 % | -0.19 | 31 % | 8.4× | 18 | 11.9 % | -8.2 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 27 634 € | -1.9 % | 0 € | -44.0 % |
| 50 % fixe (rééquilibré / 21 séances) | 30 367 € | 0.3 % | 647 € | -23.0 % |
| Tendance MA100 | 18 521 € | -10.8 % | 947 € | -51.4 % |
| Tendance MA200 | 18 187 € | -11.2 % | 938 € | -51.8 % |
| Tendance MA200 (hystérésis 2 %) | 19 350 € | -9.9 % | 1 157 € | -51.8 % |
| Momentum 12 mois | 23 686 € | -5.5 % | 1 703 € | -41.4 % |
| Sorties actuelles, entrée toujours haussière | 15 925 € | -14.0 % | 1 359 € | -60.1 % |
| Tendance MA200 + sorties actuelles | 16 782 € | -12.9 % | 478 € | -53.3 % |
| Sorties swing, entrée toujours haussière | 22 154 € | -7.0 % | 2 310 € | -47.0 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | 19 687 € | -9.5 % | 843 € | -48.7 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | -18.7 % | -10.7 % | 0.1 % | -19.9 % | 58.7 % |
| 50 % fixe (rééquilibré / 21 séances) | -9.2 % | -4.7 % | 0.5 % | -9.7 % | 31.8 % |
| Tendance MA100 | 0.5 % | -9.7 % | -14.5 % | -29.4 % | 18.1 % |
| Tendance MA200 | -10.2 % | -5.5 % | -20.5 % | -21.0 % | 19.9 % |
| Tendance MA200 (hystérésis 2 %) | -8.1 % | -4.5 % | -13.7 % | -26.7 % | 23.8 % |
| Momentum 12 mois | -7.1 % | -21.7 % | -9.7 % | 0.5 % | 28.8 % |
| Sorties actuelles, entrée toujours haussière | -17.6 % | -25.3 % | -16.3 % | -18.7 % | 33.0 % |
| Tendance MA200 + sorties actuelles | -10.2 % | -7.5 % | -20.8 % | -21.0 % | 7.0 % |
| Sorties swing, entrée toujours haussière | -17.6 % | -17.0 % | -0.2 % | -18.8 % | 44.0 % |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -8.1 % | -2.5 % | -13.7 % | -23.8 % | 13.9 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.39 [-0.19 ; +1.06] | 0.39 |
| 50 % fixe (rééquilibré / 21 séances) | -0.01 [-0.05 ; +0.03] | +0.38 [-0.20 ; +1.06] | 0.38 |
| Tendance MA100 | -0.38 [-1.11 ; +0.19] | +0.00 [-0.60 ; +0.59] | 0.14 |
| Tendance MA200 | -0.39 [-1.06 ; +0.19] | — | 0.14 |
| Tendance MA200 (hystérésis 2 %) | -0.32 [-1.07 ; +0.29] | +0.06 [-0.17 ; +0.34] | 0.17 |
| Momentum 12 mois | -0.14 [-0.76 ; +0.42] | +0.24 [-0.30 ; +0.80] | 0.28 |
| Sorties actuelles, entrée toujours haussière | -0.34 [-0.73 ; +0.02] | +0.05 [-0.74 ; +0.86] | 0.16 |
| Tendance MA200 + sorties actuelles | -0.55 [-1.26 ; +0.09] | -0.16 [-0.63 ; +0.18] | 0.08 |
| Sorties swing, entrée toujours haussière | -0.09 [-0.22 ; +0.03] | +0.30 [-0.31 ; +1.02] | 0.32 |
| Tendance MA200 (hystérésis 2 %) + sorties swing | -0.35 [-1.06 ; +0.27] | +0.03 [-0.24 ; +0.32] | 0.16 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 10 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | -1.8 % / 0.15 | -1.8 % / 0.15 | -1.9 % / 0.15 | -1.9 % / 0.15 |
| Tendance MA200 (hystérésis 2 %) | -7.5 % / -0.13 | -7.9 % / -0.15 | -8.5 % / -0.17 | -9.0 % / -0.19 |
| Sorties actuelles, entrée toujours haussière | -1.9 % / 0.14 | -6.5 % / 0.01 | -13.0 % / -0.19 | -17.0 % / -0.31 |
| Sorties swing, entrée toujours haussière | -2.4 % / 0.13 | -3.5 % / 0.10 | -5.2 % / 0.06 | -6.3 % / 0.03 |

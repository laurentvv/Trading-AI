# Références de performance — généré le 2026-09-29

Tableaux générés par `python -m src.backtest.report` (données : `data_cache/`, coûts 25 pb par côté, exécution à l'ouverture suivante). La section « Lecture » ci-dessous est rédigée à la main ; tout le reste est régénérable.

## Lecture (phase 1, tranche A : les références à battre)

1. **CRUDP.PA n'est pas backtestable.** Le flux est gelé à 82 % (volume nul, clôture recopiée) et vivant seulement depuis le 2026-01-08 (184 séances), trop court pour une MA200. Aucune conclusion sur le pétrole tradé n'est possible avec cette source. Le contrat WTI continu (CL=F) n'est qu'un proxy, non ajusté du roll. **Décision à prendre : une autre source de prix pour l'instrument pétrole, ou un autre instrument.**
2. **SXRV.DE, 2022-07 → 2026-09 : aucune règle simple ne bat le buy & hold sur le Sharpe.** Buy & hold : CAGR 21,8 %, Sharpe 1,12, drawdown max −26,7 %. Tous les écarts de Sharpe contre lui ont un intervalle qui contient 0 : rien n'est statistiquement distinguable. Le profil le plus intéressant est la **MA200 avec hystérésis 2 %** : CAGR 15,9 %, Sharpe 1,06, drawdown −15,0 %, Calmar 1,06 contre 0,82. C'est un profil « moins de risque, moins de rendement » : 6 points de CAGR en moins pour 12 points de drawdown en moins.
3. **Les règles de sortie actuelles détruisent de la valeur, surtout par le churn.** Sur le même actif et avec une entrée toujours haussière (le cas le plus favorable), elles donnent 8,6 % de CAGR (Sharpe 0,54) contre 21,8 %, soit ΔSharpe −0,58 avec un IC à 95 % de [−0,89 ; −0,28], l'un des deux seuls écarts significatifs du tableau (l'autre est la MA200 + sorties actuelles, −0,84). La sensibilité aux coûts sépare les causes : à 0 pb, 20,0 % (la troncature des gains ne coûte qu'environ 2 points) ; à 10 pb, 15,3 % ; à 25 pb, 8,6 %. Le mal vient des **83 trades et d'une rotation de 40× le capital par an** multipliés par les frictions, pas seulement de la vente « trop tôt ». Ajouter ces sorties à la MA200 aggrave encore (CAGR 3,1 %).
4. **L'hypothèse de coût pèse lourd.** 25 pb par côté vient de la plage 0,2 à 0,35 % observée en démo ; s'agit-il d'un côté ou de l'aller-retour, et le spread réel d'un ETF liquide comme SXRV.DE en compte réel est probablement plus faible ? À vérifier sur le compte réel avant de conclure. Lire toute la ligne de sensibilité, pas seulement la colonne 25 pb.
5. **Après impôt (30 %, position soldée), l'écart se creuse** : buy & hold 16,5 % net contre 10,6 % pour la MA200 hystérésis, parce que le trading actif réalise des gains chaque année. Le taux de 30 % est une hypothèse à confirmer.
6. **Conséquence pour la suite.** Les barres à battre sont le buy & hold (Sharpe 1,12) et la MA200 hystérésis (Calmar 1,06). Le critère d'edge du plan (drawdown nettement inférieur avec un CAGR proche) n'est **pas atteint** par les règles simples sur cette période. Un modèle ou une couche de décision n'a de raison d'exister que s'il fait mieux que ces deux lignes, hors échantillon et net de coûts.

**Limites.** Quatre ans et un seul cycle (baisse de 2022 puis marché très haussier) ; aucun taux sans risque ; le Sharpe déflaté ne compte que les 8 variantes de ce rapport, donc surestime la confiance ; aucun modèle de l'ensemble n'est encore testé (tranche B : rejeu des modèles en walk-forward et ablation).

## Qualité des séries

| Série | Lignes | Période | Lignes gelées | Lignes vivantes finales | Vivante depuis |
|---|---|---|---|---|---|
| SXRV.DE | 1274 | 2021-09-29 → 2026-09-29 | 0 % | 1274 | 2021-09-29 |
| CRUDP.PA | 1278 | 2021-09-29 → 2026-09-29 | 82 % | 184 | 2026-01-08 |
| ^NDX | 1255 | 2021-09-29 → 2026-09-29 | 0 % | 1255 | 2021-09-29 |
| CL=F | 1257 | 2021-09-29 → 2026-09-29 | 0 % | 1257 | 2021-09-29 |

Une ligne « gelée » = volume nul et clôture recopiée de la veille (flux Yahoo factice). Une série majoritairement gelée ne peut pas être backtestée.

## SXRV.DE — ETF Nasdaq-100 en EUR (instrument tradé)

Fenêtre commune : 2022-07-12 → 2026-09-29 (1074 séances).

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | 21.8 % | 1.12 | 1.64 | -26.7 % | 0.82 | 100 % | 0.2× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 10.9 % | 1.12 | 1.64 | -14.1 % | 0.77 | 100 % | 0.2× | 33 | 51.9 % | -3.4 % |
| Tendance MA100 | 14.3 % | 0.97 | 1.38 | -21.8 % | 0.65 | 77 % | 7.6× | 16 | 9.7 % | -2.3 % |
| Tendance MA200 | 14.6 % | 0.98 | 1.41 | -16.3 % | 0.89 | 80 % | 5.9× | 13 | 21.4 % | -2.0 % |
| Tendance MA200 (hystérésis 2 %) | 15.9 % | 1.06 | 1.52 | -15.0 % | 1.06 | 81 % | 1.9× | 4 | 31.9 % | -5.6 % |
| Momentum 12 mois | 17.4 % | 1.06 | 1.50 | -33.0 % | 0.53 | 80 % | 1.5× | 3 | 26.4 % | -2.1 % |
| Sorties actuelles, entrée toujours haussière | 8.6 % | 0.54 | 0.76 | -31.5 % | 0.27 | 92 % | 39.7× | 83 | 2.9 % | -2.9 % |
| Tendance MA200 + sorties actuelles | 3.1 % | 0.28 | 0.40 | -21.7 % | 0.14 | 75 % | 35.8× | 74 | 3.1 % | -2.4 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 57 144 € | 16.5 % | 11 633 € | -26.7 % |
| 50 % fixe (rééquilibré / 21 séances) | 41 295 € | 7.9 % | 4 841 € | -14.1 % |
| Tendance MA100 | 44 020 € | 9.5 % | 6 842 € | -23.1 % |
| Tendance MA200 | 43 966 € | 9.5 % | 7 523 € | -19.9 % |
| Tendance MA200 (hystérésis 2 %) | 45 849 € | 10.6 % | 8 195 € | -24.3 % |
| Momentum 12 mois | 48 898 € | 12.3 % | 8 099 € | -37.2 % |
| Sorties actuelles, entrée toujours haussière | 36 050 € | 4.5 % | 5 191 € | -30.2 % |
| Tendance MA200 + sorties actuelles | 31 500 € | 1.2 % | 2 600 € | -21.6 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | -13.6 % | 51.2 % | 33.5 % | 7.0 % | 23.1 % |
| 50 % fixe (rééquilibré / 21 séances) | -6.7 % | 23.6 % | 15.8 % | 3.8 % | 11.6 % |
| Tendance MA100 | -6.5 % | 28.4 % | 11.8 % | 9.3 % | 19.7 % |
| Tendance MA200 | -6.7 % | 24.6 % | 32.5 % | -0.4 % | 15.7 % |
| Tendance MA200 (hystérésis 2 %) | -6.5 % | 23.4 % | 33.5 % | 4.6 % | 15.6 % |
| Momentum 12 mois | 0.0 % | 28.4 % | 33.5 % | -4.8 % | 20.5 % |
| Sorties actuelles, entrée toujours haussière | -23.9 % | 41.9 % | 11.2 % | 3.6 % | 14.0 % |
| Tendance MA200 + sorties actuelles | -6.6 % | 11.2 % | 15.5 % | -6.7 % | 1.8 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.14 [-0.52 ; +0.86] | 0.91 |
| 50 % fixe (rééquilibré / 21 séances) | +0.00 [-0.02 ; +0.02] | +0.14 [-0.51 ; +0.86] | 0.91 |
| Tendance MA100 | -0.15 [-0.88 ; +0.61] | -0.01 [-0.61 ; +0.52] | 0.85 |
| Tendance MA200 | -0.14 [-0.86 ; +0.52] | — | 0.86 |
| Tendance MA200 (hystérésis 2 %) | -0.06 [-0.75 ; +0.56] | +0.07 [-0.11 ; +0.32] | 0.89 |
| Momentum 12 mois | -0.05 [-0.67 ; +0.62] | +0.08 [-0.73 ; +0.87] | 0.89 |
| Sorties actuelles, entrée toujours haussière | -0.58 [-0.89 ; -0.28] | -0.45 [-1.19 ; +0.32] | 0.57 |
| Tendance MA200 + sorties actuelles | -0.84 [-1.58 ; -0.16] | -0.70 [-0.98 ; -0.45] | 0.36 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 8 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | 21.9 % / 1.12 | 21.9 % / 1.12 | 21.8 % / 1.12 | 21.8 % / 1.12 |
| Tendance MA200 (hystérésis 2 %) | 16.5 % / 1.09 | 16.3 % / 1.08 | 15.9 % / 1.06 | 15.6 % / 1.04 |
| Sorties actuelles, entrée toujours haussière | 20.0 % / 1.07 | 15.3 % / 0.85 | 8.6 % / 0.54 | 4.4 % / 0.32 |

## CL=F — contrat WTI continu (PROXY du pétrole, non tradable tel quel)

Fenêtre commune : 2022-07-18 → 2026-09-29 (1057 séances). Contrat à terme continu non ajusté du roll : les sauts de roll faussent les rendements. À lire comme un ordre de grandeur, jamais comme une performance atteignable.

### Performance brute (coûts inclus, avant impôt)

| Stratégie | CAGR | Sharpe | Sortino | Drawdown max | Calmar | Temps investi | Turnover/an | Ventes | Gain moy. | Perte moy. |
|---|---|---|---|---|---|---|---|---|---|---|
| Buy & hold | -2.1 % | 0.15 | 0.20 | -47.0 % | -0.04 | 100 % | 0.3× | 0 | — | — |
| 50 % fixe (rééquilibré / 21 séances) | 0.8 % | 0.14 | 0.19 | -24.4 % | 0.03 | 100 % | 0.4× | 22 | 14.4 % | -18.2 % |
| Tendance MA100 | -9.3 % | -0.21 | -0.29 | -51.4 % | -0.18 | 36 % | 12.7× | 28 | 10.3 % | -3.3 % |
| Tendance MA200 | -10.3 % | -0.24 | -0.32 | -51.8 % | -0.20 | 32 % | 9.8× | 21 | 11.8 % | -3.6 % |
| Tendance MA200 (hystérésis 2 %) | -8.7 % | -0.18 | -0.24 | -51.8 % | -0.17 | 32 % | 4.3× | 9 | 6.2 % | -7.4 % |
| Momentum 12 mois | -1.7 % | 0.09 | 0.12 | -41.5 % | -0.04 | 34 % | 8.2× | 17 | 2.1 % | -3.0 % |
| Sorties actuelles, entrée toujours haussière | -12.6 % | -0.17 | -0.24 | -62.2 % | -0.20 | 90 % | 48.0× | 100 | 5.1 % | -6.6 % |
| Tendance MA200 + sorties actuelles | -11.7 % | -0.34 | -0.45 | -53.5 % | -0.22 | 29 % | 23.0× | 50 | 7.4 % | -5.5 % |

### Après impôt (30% sur les plus-values réalisées) et position soldée à la fin

| Stratégie | Capital final | CAGR net | Impôt payé | Drawdown max |
|---|---|---|---|---|
| Buy & hold | 27 421 € | -2.1 % | 0 € | -47.0 % |
| 50 % fixe (rééquilibré / 21 séances) | 30 288 € | 0.2 % | 675 € | -24.4 % |
| Tendance MA100 | 18 883 € | -10.4 % | 1 103 € | -51.4 % |
| Tendance MA200 | 17 915 € | -11.6 % | 1 059 € | -51.8 % |
| Tendance MA200 (hystérésis 2 %) | 19 060 € | -10.2 % | 1 280 € | -51.8 % |
| Momentum 12 mois | 25 759 € | -3.6 % | 2 039 € | -41.5 % |
| Sorties actuelles, entrée toujours haussière | 15 941 € | -14.0 % | 1 535 € | -62.0 % |
| Tendance MA200 + sorties actuelles | 17 574 € | -12.0 % | 725 € | -53.3 % |

### Rendement par année civile (brut)

| Stratégie | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Buy & hold | -21.5 % | -10.7 % | 0.1 % | -19.9 % | 63.2 % |
| 50 % fixe (rééquilibré / 21 séances) | -11.0 % | -4.0 % | 0.3 % | -9.3 % | 32.9 % |
| Tendance MA100 | 0.5 % | -9.7 % | -14.5 % | -29.4 % | 21.4 % |
| Tendance MA200 | -13.3 % | -5.5 % | -20.5 % | -21.0 % | 23.2 % |
| Tendance MA200 (hystérésis 2 %) | -11.3 % | -4.5 % | -13.7 % | -26.7 % | 27.2 % |
| Momentum 12 mois | -1.0 % | -21.7 % | -9.7 % | 0.5 % | 32.4 % |
| Sorties actuelles, entrée toujours haussière | -20.1 % | -25.3 % | -16.3 % | -18.7 % | 39.6 % |
| Tendance MA200 + sorties actuelles | -9.0 % | -7.5 % | -20.8 % | -21.0 % | 12.3 % |

### Robustesse : Sharpe contre les références (bootstrap par blocs de 21 séances, IC 95 %)

| Stratégie | ΔSharpe vs B&H | ΔSharpe vs MA200 | Sharpe déflaté (P>0) |
|---|---|---|---|
| Buy & hold | — | +0.39 [-0.23 ; +1.18] | 0.39 |
| 50 % fixe (rééquilibré / 21 séances) | -0.01 [-0.05 ; +0.03] | +0.38 [-0.24 ; +1.18] | 0.39 |
| Tendance MA100 | -0.36 [-1.12 ; +0.23] | +0.03 [-0.58 ; +0.60] | 0.16 |
| Tendance MA200 | -0.39 [-1.18 ; +0.23] | — | 0.14 |
| Tendance MA200 (hystérésis 2 %) | -0.32 [-1.08 ; +0.33] | +0.06 [-0.16 ; +0.33] | 0.17 |
| Momentum 12 mois | -0.06 [-0.64 ; +0.49] | +0.33 [-0.17 ; +0.96] | 0.35 |
| Sorties actuelles, entrée toujours haussière | -0.32 [-0.74 ; +0.04] | +0.07 [-0.76 ; +0.93] | 0.18 |
| Tendance MA200 + sorties actuelles | -0.49 [-1.25 ; +0.15] | -0.10 [-0.58 ; +0.27] | 0.10 |

Coûts : 25 points de base par côté ; exécution à l'ouverture de la séance suivante. Le Sharpe déflaté ne compte que les 8 variantes de ce tableau : il SURESTIME la confiance puisque le projet en a essayé bien davantage.

### Sensibilité aux coûts (par côté, brut avant impôt)

| CAGR / Sharpe | 0 pb | 10 pb | 25 pb | 35 pb |
|---|---|---|---|---|
| Buy & hold | -2.0 % / 0.15 | -2.0 % / 0.15 | -2.1 % / 0.15 | -2.1 % / 0.15 |
| Tendance MA200 (hystérésis 2 %) | -7.7 % / -0.14 | -8.1 % / -0.15 | -8.7 % / -0.18 | -9.1 % / -0.19 |
| Sorties actuelles, entrée toujours haussière | -1.5 % / 0.15 | -6.1 % / 0.02 | -12.6 % / -0.17 | -16.7 % / -0.30 |

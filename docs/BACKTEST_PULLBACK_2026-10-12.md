# Banc de la méthode pullback (généré par `src.pullback.bench`)

**Données jusqu'au 2026-10-12.** 6595 symboles US de Trading 212 avec des barres valides, 9266 setups candidats. Capital 30,000 €, frais 25 pb par sens (change 0,15 % + spread/glissement), impôt 30 % sur les plus-values nettes de l'année, équité en EUR, positions soldées à la fin pour comparer honnêtement au buy & hold.

## Verdict de la porte du plan (§2.1)

### FAIL

| Critère | Valeur | Seuil | OK |
|---|---|---|---|
| CAGR > buy & hold Nasdaq-100 EUR | -0.226 | 0.146 | **non** |
| Sharpe > buy & hold Nasdaq-100 EUR | -1.931 | 0.742 | **non** |
| Drawdown maximal >= -35 % | -0.789 | -0.350 | **non** |
| Espérance >= +0,25 R | -0.546 | 0.250 | **non** |
| Au moins 150 trades | 680 | 150 | oui |
| Contrôle « entrées au hasard » battu (Sharpe, p <= 0,05) | 0.976 | 0.050 | **non** |
| Contrôle « entrées au hasard » battu (espérance R, p <= 0,05) | 0.951 | 0.050 | **non** |
| Positif sur le sous-univers très liquide (proxy du biais du survivant) | -0.089 | 0.000 | **non** |

Un PASS reste **provisoire** : l'univers ne contient que des titres encore cotés aujourd'hui (les faillites et radiations ont disparu), donc les résultats absolus sont optimistes. Le contrôle « entrées au hasard » partage ce biais : l'écart stratégie - hasard est la mesure la plus fiable. Le sous-univers très liquide (>= 200 M$/jour) n'est qu'un proxy, pas l'univers à la date.

Variante retenue sur le calibrage 2015-2020 : **ema9_apres_3R**. Test 2021-01-04 -> 2026-10-12, sans réglage sur cette fenêtre.

## Test hors échantillon

| Stratégie | CAGR | Sharpe | Drawdown max | Trades | Réussite | Espérance R |
|---|---|---|---|---|---|---|
| Pullback (ema9_apres_3R) | -22.6 % | -1.93 | -78.9 % | 680 | 19.3 % | -0.55 |
| B&H Nasdaq-100 EUR (SXRV.DE) | 14.6 % | 0.74 | -31.3 % |  |  |  |
| MA200 Nasdaq-100 EUR | 9.7 % | 0.65 | -31.1 % |  |  |  |
| B&H S&P 500 EUR (SXR8.DE) | 13.1 % | 0.82 | -23.3 % |  |  |  |
| Momentum seul (mêmes sorties, sans setup) | -17.5 % | -1.11 | -71.8 % | 637 | 19.6 % | -0.40 |
| Entrées au hasard (médiane de 40) | -15.0 % | -1.25 | -64.6 % | 646.0 | 20.4 % | -0.45 |

- Trades par an : 118 ; durée moyenne 5.3 séances ; exposition moyenne 41.7 % ; gain moyen +2.74 R, perte moyenne -1.33 R.
- Frais payés : 9,382 € ; impôt payé : 0 €.
- Contrôle au hasard (40 tirages) : Sharpe médian -1.25 (min -2.11, max -0.56) ; p(Sharpe) = 0.976, p(espérance) = 0.951.
- Bootstrap par blocs, Sharpe(stratégie) - Sharpe(B&H Nasdaq-100) : -2.57 [-3.70 ; -1.54], p(<= 0) = 1.000.
- Sharpe déflaté (6 variantes) : 0.000.

### Années (test)

| Année | Pullback | B&H Nasdaq-100 |
|---|---|---|
| 2021 | -28.3 % | 41.2 % |
| 2022 | -30.9 % | -30.1 % |
| 2023 | -6.9 % | 51.2 % |
| 2024 | -10.0 % | 33.5 % |
| 2025 | -24.7 % | 7.0 % |
| 2026 | -26.9 % | 2.8 % |

### Sensibilité à l'ordre intrajournalier supposé (test)

Quand une même barre touche plusieurs niveaux, l'ordre est inconnu. `worst` (règle du plan, utilisée pour le verdict) : entrée puis stop, stop avant cible. `ohlc` : chemin O-L-H-C si la barre clôture au-dessus de l'ouverture, sinon O-H-L-C. `best` : entrée annulée si le stop cède le jour du déclenchement, cibles avant stop.

| Hypothèse | Pullback : CAGR | Sharpe | Drawdown max | Trades | Réussite | Espérance R | Hasard (médiane de 20) : Sharpe | Espérance R |
|---|---|---|---|---|---|---|---|---|
| worst | -22.6 % | -1.93 | -78.9 % | 680 | 19.3 % | -0.55 | -1.29 | -0.45 |
| ohlc | -17.1 % | -1.45 | -69.0 % | 580 | 22.6 % | -0.42 | -0.81 | -0.33 |
| best | -7.8 % | -0.63 | -43.3 % | 451 | 29.5 % | -0.16 | -0.26 | -0.13 |

### Sensibilité aux frais (test)

| Frais par sens | CAGR | Sharpe | Drawdown max | Trades | Réussite | Espérance R |
|---|---|---|---|---|---|---|
| 0 pb | -14.1 % | -1.17 | -62.4 % | 673 | 20.4 % | -0.26 |
| 10 pb | -17.5 % | -1.47 | -69.8 % | 678 | 20.4 % | -0.36 |
| 25 pb | -22.6 % | -1.93 | -78.9 % | 680 | 19.3 % | -0.55 |
| 40 pb | -27.7 % | -2.41 | -85.6 % | 679 | 18.1 % | -0.73 |

## Variantes pré-enregistrées

| Variante | Calibrage 2015-2020 : CAGR | Sharpe | DD | Trades | Réussite | Esp. R | Test : CAGR | Sharpe | DD | Trades |
|---|---|---|---|---|---|---|---|---|---|---|
| base | -12.1 % | -1.12 | -55.7 % | 733 | 23.9 % | -0.33 | -25.1 % | -2.33 | -82.2 % | 848 |
| ema9_apres_3R | -9.4 % | -0.83 | -45.8 % | 590 | 25.1 % | -0.26 | -22.6 % | -1.93 | -78.9 % | 680 |
| rang_80 | -12.1 % | -1.11 | -55.0 % | 720 | 23.9 % | -0.35 | -27.5 % | -2.58 | -84.9 % | 817 |
| atr_0.7 | -12.3 % | -1.44 | -58.5 % | 479 | 21.5 % | -0.39 | -17.1 % | -2.02 | -66.2 % | 481 |
| stop_4pct | -14.8 % | -1.32 | -66.8 % | 838 | 23.7 % | -0.34 | -28.2 % | -2.41 | -87.1 % | 956 |
| filtre_marche_QQQ | -13.4 % | -1.27 | -60.1 % | 702 | 23.4 % | -0.37 | -22.1 % | -2.31 | -77.6 % | 691 |

Sélection sur le Sharpe du calibrage uniquement ; les colonnes « Test » sont informatives.

## Références sur le calibrage

| Stratégie | CAGR | Sharpe | Drawdown max |
|---|---|---|---|
| B&H Nasdaq-100 EUR | 16.2 % | 0.78 | -28.4 % |
| MA200 Nasdaq-100 EUR | 7.6 % | 0.50 | -27.5 % |
| B&H S&P 500 EUR | 9.0 % | 0.54 | -33.8 % |

## Limites connues

- **Biais du survivant** non éliminé (univers = titres encore cotés ; pas de composition historique d'indice).
- Dates de résultats inconnues dans l'historique : les gaps de résultats sont payés en entier (pas de filtre).
- Déclencheur journalier (le plus haut du jour de setup) au lieu de l'intraday de la source.
- Ordre intrajournalier inconnu : stop avant cible, entrée avant stop (hypothèse défavorable).
- Change appliqué au cours de clôture du jour ; frais de change inclus dans les 25 pb.
- Barres incohérentes écartées (High/Low > 5, O/C hors [L, H], lignes gelées) ; erreurs résiduelles possibles.

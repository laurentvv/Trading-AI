# 🏛️ Weekend Council — Guide de Test PROD

Ce guide permet de valider le council sur le serveur PROD : installation des
modèles, déclenchement forcé via le scheduler, et récupération des logs pour
analyse.

## Prérequis

- Avoir fait `git pull` de la branche `feat/weekend-council-10527370498881876130`
- Ollama doit tourner (`ollama serve`)

---

## Étape 1 — Installer les 6 modèles du council

Le council utilise 5 familles de modèles distinctes (Gemma / GLM / Qwen / LFM /
Mistral) + 1 Juge (Qwen3.5-MTP). **Sans eux, les membres retombent silencieusement
sur Gemma seul — la diversité de raisonnement est perdue.**

```bash
uv run python setup_council_models.py
```

- **Idempotent** : saute les modèles déjà installés.
- **Taille totale** : ~40 GB (première fois). Prévoir de la place disque.
- **Vérification** : le script affiche `✅ Tous les modèles du council sont déjà installés.` quand c'est bon.

Liste attendue (6 modèles) :
1. `hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K` — Le Stratège
2. `hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q6_K` — Le Gestionnaire de Risque
3. `qwen3.5:9b` — Le Quant
4. `lfm2.5-thinking:1.2b-bf16` — Le Sceptique + Le Comportementaliste
5. `mistral-nemo:12b-instruct-2407-q6_K` — Le Tacticien
6. `hf.co/unsloth/Qwen3.5-9B-MTP-GGUF:Q6_K` — Le Juge

---

## Étape 2 — Lancer le council via le scheduler (Option 2)

Ce script force le déclenchement dans ~2 minutes (au lieu d'attendre sam/dim 09:00),
simule un jour de week-end, et s'arrête tout seul après le 1er run.

```bash
.venv\Scripts\python.exe schedule_test.py
```

### Ce qui va se passer :
1. Le dashboard affiche un compte à rebours (`déclenchement dans Xs`)
2. À H+2min, le council démarre via subprocess (~15-20 min sur CPU)
3. Le rapport est sauvegardé dans `docs/council_reports/council_report_AAA-MM-JJ.md`
4. Le script s'arrête tout seul (exit 0)

### Pendant l'attente (~2 min avant déclenchement)
Le dashboard rafraîchit toutes les 15s. Ne pas fermer la fenêtre.

### Pendant le council (~15-20 min)
- **Ne pas fermer la fenêtre** — le subprocess tourne dedans
- Les logs vont dans `weekend_council_test.log`
- Si tu vois `Falling back to hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`, c'est
  normal : un modèle a retourné une réponse vide (thinking tokens épuisés) et
  le fallback gracieux a pris le relais.

---

## Étape 3 — Vérifier le rapport généré

```bash
# Voir le rapport (le verdict + le décompte des positions + les débats)
cat docs/council_reports/council_report_AAA-MM-JJ.md
```

Points à vérifier :
- ✅ Section `## Décompte des positions` (tableau avec STANCE par membre)
- ✅ Section `## Reformulation du problème (Round 0)`
- ✅ Section `## Verdict du Juge` commence par `Ce que le conseil NE PEUT PAS déterminer`
- ✅ Section `## Modèles utilisés` montre les 6 familles distinctes
- ✅ Bloc `VERDICT_TICKER:` à la toute fin du verdict du Juge (format :
  `SXRV.DE: BUY (0.65)`) — **critique**, c'est ce que `main.py` parse pour
  voter dans le consensus

### Vérifier que le verdict est bien injecté dans main.py

```bash
.venv\Scripts\python.exe -c "from src.llm_client import get_council_verdict_context, get_council_ticker_stance; print('Contexte injecté:', len(get_council_verdict_context()), 'chars'); print('SXRV.DE:', get_council_ticker_stance('SXRV.DE')); print('CRUDP.PA:', get_council_ticker_stance('CRUDP.PA'))"
```

Doit retourner le stance par ticker (ex: `('SELL', 0.771)`) si le bloc
`VERDICT_TICKER` est bien présent. Si retourne `('None', 0.0)`, le Juge n'a pas
respecté le format — le vote council sera silencieusement skipé.

---

## Étape 4 — Récupérer les logs pour analyse

À me partager après le test pour que je puisse analyser :

### 4a. Le rapport complet
```bash
# Copier le rapport généré (remplacer AAA-MM-JJ par la date du jour)
cat docs/council_reports/council_report_AAA-MM-JJ.md
```

### 4b. Les logs du scheduler
```bash
cat weekend_council_test.log
```
(Pour voir quels modèles ont échoué / fait du fallback, les warnings, etc.)

### 4c. Vérification santé des modèles
```bash
.venv\Scripts\python.exe -c "import requests; r=requests.get('http://localhost:11434/api/tags', timeout=5); [print(m['name']) for m in r.json().get('models',[])]"
```

---

## Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Lancer avec `python -m src.council.weekend_council` (pas le chemin direct) |
| `OLLAMA INDISPONIBLE` | Démarrer Ollama : `ollama serve` |
| Un membre ne répond pas (réponse vide) | Normal sur CPU, le fallback Gemma prend le relais automatiquement |
| Le verdict du Juge n'a pas de bloc `VERDICT_TICKER` | Le Juge (Qwen3.5-MTP) n'a pas respecté le format. Le council fonctionne mais le vote Niveau 3 est skipé. Me le signaler pour ajuster le prompt. |
| Le council prend >1h | Sur CPU c'est possible avec 6 modèles. Le timeout est à 3600s (1h). |
| `uv: Failed to canonicalize script path` | Utiliser `.venv\Scripts\python.exe` au lieu de `uv run` |

---

## Après le test

Une fois le rapport généré, le verdict est **automatiquement actif** pour la
semaine : `main.py` l'injectera dans chaque décision de trading (poids 9.5%
dans le consensus, avec décroissance linéaire jusqu'à J+7).

Pour un lancement automatique par la suite, le scheduler normal (`schedule.py`)
déclenchera le council **samedi ET dimanche à 09:00** sans intervention.

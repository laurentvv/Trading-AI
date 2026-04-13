 Code Review Complet — Trading-AI
📋 Vue d'ensemble du projet
Système de trading algorithmique hybride combinant 5 sources de signaux (modèle classique ML, LLM texte, LLM visuel, analyse de sentiment, TimesFM 2.5) avec gestion du risque, position sizing, et exécution via Trading 212. ~4 500 lignes de Python réparties dans 15+ modules.

🔴 CRITIQUE — Sécurité
1. Secrets en clair dans le code
Fichiers : t212_executor.py:32-37, news_fetcher.py:23, llm_client.py:11-13

# t212_executor.py
api_key = os.getenv("T212_API_KEY")
api_secret = os.getenv("T212_API_SECRET")
auth_str = f"{api_key}:{api_secret}"
L'API key et secret Trading 212 sont chargés depuis .env.t212 et concaténés en Base64. Le fichier .env.t212 est dans .gitignore mais le pattern load_dotenv(".env.t212") avec un chemin hardcodé est fragile. De plus, les modèles LLM gemma4:e4b sont hardcodés — si une clé API était nécessaire, elle serait exposée.

Recommandation : Centraliser la config dans un module config.py unique, valider la présence de toutes les variables d'environnement au démarrage, et ne jamais logger de secrets.

2. Appels subprocess avec des données utilisateur
Fichier : setup_timesfm.py:56

run_command(f"git clone {repo_url} {timesfm_dir}", "Cloning...")
shell=True est utilisé dans run_command si l'argument est une string. Bien que repo_url soit hardcodé, le pattern est dangereux si quelqu'un modifie la variable.

Recommandation : Toujours utiliser des listes pour subprocess.run, jamais shell=True.

3. Injections SQL impossibles mais pas de paramétrage dans _calculate_win_rate
Fichier : performance_monitor.py:743

df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date ASC", conn)
Ce n'est pas une injection directe, mais la connexion est ouverte sans context manager. Un crash laisserait la connexion ouverte.

4. .env.example incomplet
Le fichier .env.example ne contient que ALPHA_VANTAGE_API_KEY=. Il manque T212_API_KEY, T212_API_SECRET, T212_ENV.

🔴 CRITIQUE — Bugs et Comportement Incorrect
5. schedule.py : Calcul de next_run_dt cassé
Fichier : schedule.py:106-108

next_run_dt = now.replace(second=0, microsecond=0) + \
              (dt.now() - now) + \
              (dt.now() - dt.now()) # Reset
Ce calcul est absurde : (dt.now() - dt.now()) fait ~0, et (dt.now() - now) ajoute quelques microsecondes. La ligne suivante (111) réassigne correctement next_run_dt, mais ces 3 lignes sont du code mort incorrect qui prête à confusion.

6. main.py : Imports conditionnels dans le flux d'exécution
Fichier : main.py:96-98, 104, 153

import csv
from datetime import datetime
# ...
from t212_executor import load_portfolio_state as load_t212_state
Ces imports sont faits à l'intérieur d'un if is_t212: puis à nouveau dans un elif is_t212:. Cela fonctionne mais c'est du code dupliqué. Le module csv est importé dans le corps de la fonction alors qu'il devrait l'être en tête de fichier.

7. backtest_engine.py : Recalcul redondant des indicateurs
Fichier : backtest_engine.py:140, 165

Les indicateurs techniques sont calculés deux fois par itération : une fois pour le LLM (ligne 140) et une fois pour le decision engine (ligne 165). C'est un gaspillage de CPU significatif sur un backtest de plusieurs années.

8. database.py : portfolio_history — UNIQUE constraint sur date seule
Fichier : database.py:39

date TEXT NOT NULL UNIQUE,
La contrainte UNIQUE est sur date seul, pas sur (date, ticker). Cela signifie qu'on ne peut pas avoir deux tickers différents avec le même timestamp — un bug pour le trading multi-tickers.

9. enhanced_decision_engine.py : Biais haussier hardcodé
Fichier : enhanced_decision_engine.py:178, 316-318

score += 0.05  # Bullish Bias
# ...
if classic_decision.signal == 'BUY' and classic_decision.confidence > 0.4:
    adjusted_score += 0.1
if timesfm_decision.signal == 'BUY' and timesfm_decision.confidence > 0.2:
    adjusted_score += 0.1
Un biais haussier constant de +0.05 est ajouté à chaque décision, plus des bonus de +0.1 pour les modèles quantitatifs acheteurs. Ce n'est pas documenté et rend le système structurellement acheteur, ce qui est dangereux pour un système de trading réel.

10. test_full_cycle.py : Suppression non sécurisée de fichiers DB
Fichier : test_full_cycle.py:43-44

for db in db_files:
    if os.path.exists(db):
        os.remove(db)
Pas de vérification, pas de confirmation. Si le script est lancé en production, il détruit l'historique de trading.

11. t212_executor.py : Bare except clause
Fichier : t212_executor.py:88

except:
    full_state = {"tickers": {}}
Un except: nu avale toutes les exceptions, y compris KeyboardInterrupt et SystemExit. C'est une mauvaise pratique.

12. t212_executor.py : Fallback prices hardcodés
Fichier : t212_executor.py:121-123

if "SXRV" in target: return 1240.0
if "CRUD" in target: return 12.50
return 100.0
Des prix en dur comme fallback. Si le prix réel a changé significativement, cela conduirait à des ordres incorrects.

🟠 MAJEUR — Architecture & Conception
13. sys.path.append partout
Fichiers : main.py:17, backtest_engine.py:12, run_short_backtest.py:6, setup_timesfm.py (non), t212_executor.py:12

sys.path.append(str(Path(__file__).parent / 'src'))
Ce pattern est fragile et répété. Le projet devrait être un package Python installable (ce que pyproject.toml suggère partiellement) avec uv pip install -e ..

Recommandation : Ajouter un [tool.setuptools.packages.find] dans pyproject.toml et supprimer tous les sys.path.append.

14. enhanced_trading_example.py n'est pas un "exemple"
Le nom du fichier suggère un exemple/démo, mais c'est le cœur du système (EnhancedTradingSystem). Le nommage est trompeur.

Recommandation : Renommer en trading_system.py ou core.py.

15. Singleton mal implémenté dans TimesFMModel
Fichier : timesfm_model.py:22-28

class TimesFMModel:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
Ce singleton n'est pas thread-safe et l'instance est partagée globalement. De plus, dans backtest_engine.py, Backtester.__init__ crée une nouvelle instance directement (self.timesfm = TimesFMModel()) au lieu d'utiliser le singleton, ce qui charge potentiellement plusieurs fois le modèle en mémoire.

16. Deux systèmes de poids incohérents
EnhancedDecisionEngine.__init__ : {classic: 0.15, llm_text: 0.25, llm_visual: 0.20, sentiment: 0.15, timesfm: 0.25}
AdaptiveWeightManager.__init__ : {classic: 0.25, llm_text: 0.25, llm_visual: 0.20, sentiment: 0.15, timesfm: 0.15}
Les poids de base différent entre les deux modules. Quand AdaptiveWeightManager fournit des poids au DecisionEngine, les valeurs de base sont différentes.

17. Pas d'abstraction pour les sources de données
data.py mélange yfinance, Alpha Vantage, FRED, et des fallbacks hardcodés dans une seule fonction monolithique de 675 lignes. Il devrait y avoir une interface/classe abstraite DataProvider avec des implémentations concrètes.

18. Gestion d'erreur incohérente
Le projet mélange trois stratégies :

Retourner des valeurs par défaut ({}, None, pd.DataFrame())
Logger l'erreur et continuer
Lever des exceptions
Il n'y a pas de stratégie unifiée. Par exemple, enhanced_trading_example.py:43-46 fait un sys.exit(1) si ALPHA_VANTAGE_API_KEY est manquant, mais data.py:23-24 se contente d'un warning.

🟠 MAJEUR — Performance
19. Ré-entraînement du modèle à chaque cycle
Fichier : enhanced_trading_example.py:186 et backtest_engine.py:52

Le modèle classique est ré-entraîné à chaque exécution. Pour un backtest, cela signifie un fit() par jour de trading. Avec RandomForestClassifier(n_estimators=200), c'est extrêmement lent.

20. Appels LLM séquentiels
Fichier : enhanced_trading_example.py:222-265

Les appels au LLM texte, LLM visuel, web research, et news sont tous séquentiels. Avec le wrapper async de web_researcher.py, il serait possible de paralléliser au moins la recherche web et le fetch de news.

21. TimesFM chargé en mémoire potentiellement plusieurs fois
Si le backtest crée un Backtester qui instancie TimesFMModel() directement (pas via le singleton), et que le système principal utilise aussi le modèle, deux copies de TimesFM 2.5 (200M paramètres) sont en RAM.

22. pandas-datareader importé deux fois
Fichier : data.py:8 et data.py:445

import pandas_datareader.data as web  # Ligne 8
# ...
import pandas_datareader.data as web  # Ligne 445 (dans la fonction)
🟡 MODÉRÉ — Qualité du Code
23. Langue mixte (FR/EN) dans le code
Les commentaires, docstrings, logs, et messages d'erreur mélangent français et anglais sans logique claire. Exemples :

main.py : commentaires en français, variables en anglais
backtest_engine.py : logger en français, code en anglais
database.py : docstrings en anglais, commentaires en français
Recommandation : Standardiser sur une seule langue (anglais recommandé pour un projet open-source).

24. Logging configuré plusieurs fois
Fichiers : main.py:28-35, schedule.py:18-26, enhanced_trading_example.py:646-649, backtest_engine.py:24

logging.basicConfig() est appelé dans plusieurs points d'entrée. Le premier appel gagne, les autres sont ignorés silencieusement. Cela rend le logging fragile.

25. Magic numbers
Fichier : enhanced_decision_engine.py

score += 0.05     # Pourquoi 0.05 ?
if volatility > 0.04:  # Pourquoi 0.04 ?
if rsi > 85:      # Pourquoi 85 et pas 80 ?
Ces seuils devraient être des constantes nommées ou dans un fichier de configuration.

26. Imports non utilisés
Fichier : enhanced_trading_example.py:11 — import subprocess est utilisé indirectement, mais l'approche de lancer un script Python via subprocess pour les news (news_fetcher.py) est un anti-pattern. Il devrait être importé directement.

27. read_simul.py référencé mais non trouvé
Le fichier src/read_simul.py existe dans le glob mais n'a pas été examiné — il pourrait contenir du code mort.

28. Types de retour non documentés
Plusieurs fonctions retournent des tuples non nommés :

get_latest_portfolio_state() retourne (position, cash, total_value, benchmark_value) — devrait être un NamedTuple ou dataclass.
get_latest_transaction() retourne (date, type, quantity, price, cost) — idem.
🟡 MODÉRÉ — Tests
29. Couverture de test insuffisante
Les tests couvrent :

✅ database.py, features.py, llm_client.py, chart_generator.py, sentiment_analysis.py, timesfm_model.py, enhanced_decision_engine.py
Mais ne couvrent PAS :

❌ advanced_risk_manager.py — critique pour un système de trading
❌ adaptive_weight_manager.py
❌ data.py — source de données
❌ t212_executor.py — exécution réelle d'ordres
❌ performance_monitor.py
❌ web_researcher.py
❌ enhanced_trading_example.py — le cœur du système
30. Tests manuels vs automatisés
test_full_cycle.py et test_t212.py sont des scripts manuels, pas des tests unitaires. Ils attendent une interaction humaine et suppriment des données.

🟡 MODÉRÉ — Robustesse
31. Pas de validation des réponses API Trading 212
Fichier : t212_executor.py:170

current_pos = next((p for p in portfolio['positions'] 
                   if p['instrument']['ticker'] == t212_ticker), None)
Si la structure JSON de l'API change, cela lève une KeyError silencieusement avalée par le except nu (point 11).

32. Race condition potentielle sur t212_portfolio_state.json
Fichier : t212_executor.py

Le fichier JSON est lu, modifié, puis réécrit sans aucun verrou. Si deux instances du scheduler tournent en parallèle, il y a un risque de corruption de données.

33. news_fetcher.py : Chemin hardcodé
Fichier : news_fetcher.py:8

ALPHA_EAR_PATH = Path("D:/GIT/fork/Trading-AI/.agents/skills/alphaear-news/scripts")
Ce chemin absolu Windows ne fonctionnera sur aucune autre machine.

🟢 MINEUR — Style & Maintainabilité
34. Nommage du fichier principal
enhanced_trading_example.py — voir point 14.

35. __init__.py vide dans src/
Fichier : src/__init__.py

Le fichier est vide, ce qui signifie que le package n'exporte rien. Il devrait au minimum importer les classes principales.

36. Dépendances potentiellement inutiles
pyproject.toml liste hyperliquid-python-sdk, crawl4ai, playwright, torch, jax, jaxlib, shap — certaines ne semblent pas utilisées directement dans le code source principal (shap n'est importé nulle part).

37. start_scheduler.bat — Script Windows only
Le scheduler utilise un fichier .bat pour démarrer, ce qui limite la portabilité.

📊 Résumé des Priorités
Priorité	Issue	Impact
🔴 P0	Biais haussier hardcodé (#9)	Pertes financières
🔴 P0	UNIQUE constraint DB (#8)	Crash multi-ticker
🔴 P0	Singleton ignoré dans backtest (#15)	OOM
🔴 P0	except: nu (#11)	Bugs silencieux
🔴 P1	Poids incohérents (#16)	Décisions biaisées
🔴 P1	Race condition JSON (#32)	Corruption données
🟠 P1	sys.path.append (#13)	Imports fragiles
🟠 P1	Chemin hardcodé news (#33)	Cassé sur autre machine
🟠 P1	Ré-entraînement chaque cycle (#19)	Performance
🟠 P2	Tests insuffisants (#29)	Régressions
🟡 P2	Langue mixte (#23)	Maintainabilité
🟡 P2	Magic numbers (#25)	Maintainabilité
🟢 P3	Dépendances inutilisées (#36)	Taille du projet
En résumé : L'architecture hybride à 5 modèles est intéressante et bien structurée en dataclasses. Cependant, le système comporte des biais de décision hardcodés dangereux (#9), des incohérences de configuration (#16), et des problèmes de concurrence (#32) qui doivent être résolus avant une utilisation en trading réel. La priorité immédiate devrait être d'éliminer le biais haussier, de corriger la contrainte DB, et d'ajouter des tests pour le risk manager et l'exécuteur T212.
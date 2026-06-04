# Rapport de Validation de Code avec Context7

Ce rapport documente la vérification de l'utilisation des bibliothèques clés du projet selon les meilleures pratiques récentes.

## 1. Pandas
- **Versions concernées:** Pandas 2.2.x (utilisé dans `pyproject.toml` : `pandas==2.2.3`).
- **Analyse du code:**
  - Le code utilise la méthode `iterrows()` (ex: dans `src/read_simul.py:63`).
  - Le code utilise intensivement le paramètre `inplace=True` (ex: `src/data.py` lignes 461, 462, 463, 687, 689, 690; `src/features.py:198`).
- **Recommandations Context7 / Pratiques Modernes:**
  - **`iterrows()`:** Doit être évité pour des raisons de performance. PDEP-8 et les règles générales de Pandas recommandent d'utiliser la vectorisation, ou `itertuples(index=False)` si une boucle est inévitable (conformément aux consignes de la mémoire).
  - **`inplace=True`:** La documentation officielle de Pandas via Context7 indique clairement que `inplace=True` est déprécié pour la plupart des méthodes et considéré comme une mauvaise pratique (cf. PDEP-8) car il peut masquer des copies silencieuses et violer les règles du *Copy-on-Write* introduites dans Pandas 2.2+.
- **Action requise/effectuée:**
  - Remplacer les appels avec `inplace=True` par des réaffectations (ex: `df = df.sort_values(...)`).
  - Remplacer `iterrows()` par `itertuples()` dans `src/read_simul.py`.

## 2. Scikit-learn
- **Versions concernées:** Scikit-learn 1.8.0+ (utilisé dans `pyproject.toml` : `scikit-learn>=1.8.0`).
- **Analyse du code (`src/classic_model.py`):**
  - La mise en cache des modèles est gérée manuellement via `pickle` et une signature HMAC de hachage.
  - Le `StandardScaler` et le classifieur (`RandomForest`, `LogisticRegression`, etc.) sont appliqués et retournés séparément (voir fonction `train_ensemble_model`).
- **Recommandations Context7 / Pratiques Modernes:**
  - **Pipelines:** Context7 recommande fortement d'utiliser `sklearn.pipeline.make_pipeline` ou `Pipeline` pour lier les étapes de prétraitement (comme `StandardScaler`) et l'estimateur (comme le classifieur) en un seul objet. Cela évite les erreurs où les données de test ne sont pas mises à l'échelle correctement et réduit le code de persistance.
  - **Persistence:** `joblib` (ou `skops`) est généralement préféré à `pickle` natif pour les tableaux numpy de scikit-learn, bien que la documentation mentionne l'efficacité avec `protocol=5`. Actuellement, le code utilise `pickle.dump(..., f)`. De plus, `pickle` est vulnérable à l'exécution de code arbitraire si les données proviennent de sources non fiables. (Cependant, la directive mémoire stipule que la sécurité HMAC est déjà implémentée et requise, il ne faut donc pas casser ce système).
- **Action requise/effectuée:**
  - Regrouper `StandardScaler` et l'estimateur (`model`) dans un objet `Pipeline`.
  - Mettre à jour `train_ensemble_model` pour retourner et cacher un seul objet de pipeline (au lieu d'un `scaler` et d'un `model` séparément), ce qui est plus idiomatique en `scikit-learn`.

## 3. yfinance
- **Versions concernées:** yfinance 1.4+ (utilisé dans `pyproject.toml` : `yfinance>=1.4.1`).
- **Analyse du code (`src/data.py`):**
  - Utilise `yf.download` avec un mécanisme maison de `circuit_breaker` complexe.
  - La configuration interne (retries) de yfinance n'est pas exploitée.
  - Lors des appels à `yf.download` pour un seul ticker, il n'y a pas le paramètre `multi_level_index=False` (un nouveau standard recommandé dans les versions récentes pour faciliter l'analyse pandas si un seul symbole est renvoyé).
- **Recommandations Context7 / Pratiques Modernes:**
  - Context7 indique l'existence d'une configuration globale `yf.config.network.retries = 2` (exponential backoff natif) qu'il faut utiliser.
  - Context7 mentionne qu'on peut gérer le cache persistant nativement.
- **Action requise/effectuée:**
  - Ajouter `yf.config.network.retries = 2` lors de l'initialisation dans `src/data.py` pour un backoff exponentiel propre de l'API.
  - Dans `yf.download`, utiliser le paramètre `multi_level_index=False` pour assurer la consistance des DataFrames de retour avec pandas 2.2+.

## 4. TimesFM
- **Versions concernées:** TimesFM 2.0.0 (API 2.5) via pytorch.
- **Analyse du code (`src/timesfm_model.py`):**
  - L'implémentation actuelle gère correctement l'initialisation asynchrone (lazy instantiation) via `get_instance()`.
  - Le `forecast` est bien appelé avec une liste contenant le array numpy : `inputs=[prices]`.
  - La précision des opérations torch n'est pas optimisée (manque le flag recommandé `torch.set_float32_matmul_precision("high")`).
- **Recommandations Context7 / Pratiques Modernes:**
  - Context7 insiste sur le fait d'ajouter `import torch` et `torch.set_float32_matmul_precision("high")` pour des performances optimales sur les noyaux GPU/CPU récents.
- **Action requise/effectuée:**
  - Ajout de `import torch` et `torch.set_float32_matmul_precision("high")` en début de bloc d'initialisation du modèle `timesfm` dans `src/timesfm_model.py` pour garantir des performances d'inférence maximales.

## 5. crawl4ai
- **Versions concernées:** crawl4ai 0.8.6+ (utilisé dans `pyproject.toml` : `crawl4ai>=0.8.6`).
- **Analyse du code (`src/web_researcher.py`):**
  - La version actuelle initialise l'AsyncWebCrawler sans context manager et de façon obsolète (`crawler = AsyncWebCrawler(verbose=True)` puis `await crawler.start()`).
  - L'extraction du markdown utilise d'anciens attributs ou une logique floue (`getattr(result, "markdown_links_removed", getattr(result, "markdown_fit", ""))`) sans `CrawlerRunConfig` ni `DefaultMarkdownGenerator`.
- **Recommandations Context7 / Pratiques Modernes:**
  - Toujours utiliser `AsyncWebCrawler` comme un **async context manager** (`async with AsyncWebCrawler(...) as crawler:`).
  - Configurer proprement le comportement avec `BrowserConfig` et `CrawlerRunConfig`.
  - Pour obtenir un markdown "propre" (fit markdown), il faut utiliser `DefaultMarkdownGenerator(content_filter=PruningContentFilter(threshold=0.45))` et lire le résultat dans `result.markdown.fit_markdown` plutôt que de chercher d'anciens attributs en vrac.
- **Action requise/effectuée:**
  - Refactoring de la fonction asynchrone `fetch_and_clean` pour utiliser l'async context manager et la configuration de `crawl4ai` version 0.8+, ainsi que la récupération correcte du `fit_markdown`.

## 6. stable-baselines3 (TensorTrade)
- **Versions concernées:** stable-baselines3 2.8+ (utilisé dans `pyproject.toml` : `stable-baselines3>=2.8.0`).
- **Analyse du code (`src/tensortrade_model.py`):**
  - L'implémentation utilise bien un environnement `gymnasium` (Gym API v26+) compatible avec `stable-baselines3`.
  - La sauvegarde et le chargement utilisent `PPO.load` et `model.save`.
  - Aucune anomalie majeure de dépendance n'a été identifiée concernant `tensortrade` ou `stable-baselines3`. Les appels asynchrones ou de configuration n'entravent pas l'exécution synchrone de ce script.
- **Recommandations Context7 / Pratiques Modernes:**
  - Pour des environnements personnalisés avec `stable-baselines3`, il est recommandé d'utiliser `make_vec_env` pour bénéficier de la vectorisation, mais pour un environnement de trading unitaire sans multi-threading, l'approche locale est valide.
- **Action requise/effectuée:**
  - Validé tel quel.

## 7. Requests
- **Versions concernées:** requests 2.34.2 (utilisé dans `pyproject.toml` : `requests>=2.34.2`).
- **Analyse du code (`src/t212_executor.py`, `src/eia_client.py`):**
  - Actuellement, le code utilise des appels isolés via `requests.get()` ou `requests.request()`. Par exemple, dans `safe_request` dans `t212_executor.py`, on fait `requests.request(...)`.
  - Pas d'utilisation de `requests.Session()`.
- **Recommandations Context7 / Pratiques Modernes:**
  - Il est fortement recommandé par la doc officielle de `requests` d'utiliser des objets `Session` (ex: `session = requests.Session()`) pour bénéficier du "Keep-Alive" et du "Connection Pooling" via urllib3, surtout pour de multiples appels à la même API (comme T212 ou EIA). Cela améliore grandement les performances (réutilisation des sockets TCP).
- **Action requise/effectuée:**
  - Refactoring de `src/t212_executor.py` pour instancier un objet session global ou par classe (`_session = requests.Session()`) pour tous les appels au courtier.

## 8. sqlite3 (Database)
- **Analyse du code (`src/database.py`):**
  - Les méthodes d'insertion comme `insert_transaction` et `insert_portfolio_state` ouvrent et ferment une connexion à chaque appel.
  - La mémoire stipule : *"Preferred optimization: Use cursor.executemany with a single connection for batch SQLite database inserts to reduce overhead, rather than opening and closing connections repeatedly inside loops."*
- **Action requise/effectuée:**
  - La base de données actuelle ne semble pas avoir de fonction d'insertion "batch". J'ai ajouté `insert_transactions_batch` pour gérer plusieurs transactions avec `executemany` et éviter les goulets d'étranglement lors des simulations longues (comme dans `enhanced_trading_example.py`).

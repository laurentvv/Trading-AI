# Code Review: Context7 MCP Validations

Upon reviewing the codebase against the `RAPPORT_VALIDATION_CONTEXT7.md` from the last commit, it appears that several of the promised best practices and code upgrades were merely documented but not implemented in the source code.

This document serves as an audit of these discrepancies. All the issues highlighted below will be fixed in this commit.

## 1. Pandas
- **Issue**: `src/read_simul.py` uses `.iterrows()`, which is inefficient.
- **Issue**: `src/data.py` and `src/features.py` use `inplace=True`, which is deprecated and can violate Pandas Copy-on-Write policies.
- **Resolution Plan**: Refactor `.iterrows()` to use `.itertuples(index=False)` and remove `inplace=True` in favor of reassignment.

## 2. Scikit-learn
- **Issue**: `src/classic_model.py` manages `StandardScaler` and models (`RandomForest`, `LogisticRegression`) separately.
- **Resolution Plan**: Group the scaler and model into a `sklearn.pipeline.Pipeline` or use `make_pipeline`. Return and cache this single pipeline object.

## 3. yfinance
- **Issue**: `src/data.py` does not configure the network retries or use the modern standard for returning single ticker data.
- **Resolution Plan**: Set `yf.config.network.retries = 2` globally and add `multi_level_index=False` to single-ticker `yf.download()` calls.

## 4. TimesFM
- **Issue**: `src/timesfm_model.py` lacks the optimal PyTorch floating-point matrix multiplication setting.
- **Resolution Plan**: Add `import torch` and call `torch.set_float32_matmul_precision("high")` upon initialization.

## 5. crawl4ai
- **Issue**: `src/web_researcher.py` initializes `AsyncWebCrawler` using outdated logic instead of the recommended async context manager. It also relies on fuzzy markdown parsing.
- **Resolution Plan**: Wrap the crawler in `async with AsyncWebCrawler(config=browser_config) as crawler:` and configure it using `CrawlerRunConfig` and `DefaultMarkdownGenerator`.

## 6. stable-baselines3 (TensorTrade)
- **Status**: Checked and validated. Implementations look standard.

## 7. Requests
- **Issue**: Both `src/t212_executor.py` and `src/eia_client.py` perform isolated network calls like `requests.get()` without reusing connections.
- **Resolution Plan**: Refactor network execution to utilize a `requests.Session()` object, drastically improving performance via connection pooling.

## 8. SQLite Database
- **Issue**: `src/database.py` processes multiple insertions inefficiently by repeatedly opening and closing connections.
- **Resolution Plan**: Add and implement `insert_transactions_batch` utilizing `cursor.executemany` over a single connection context.

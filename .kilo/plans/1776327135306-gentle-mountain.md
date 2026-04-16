# Architecture Plan: EIA Data Integration Module for WTI

## Executive Summary

This plan details the architecture for a **completely cloisoned** EIA data module dedicated to WTI (Crude Oil) fundamental analysis. The module introduces two new files (`src/eia_client.py` and `src/oil_bench_model.py`) with minimal, controlled modifications to three existing files. The module is **technically and logically excluded** from all NASDAQ-related processing.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      SCHEDULED RUN                              │
│  schedule.py → main.py → EnhancedTradingSystem.run_analysis()   │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────┐
│  EnhancedTradingSystem.perform_enhanced_analysis()               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  TICKER GATE: is_oil_ticker(self.analysis_ticker)          │ │
│  │  ┌──────────────┐    ┌───────────────────────────────────┐ │ │
│  │  │  "CL=F"      │───▶│  OIL BENCH PATH (exclusive)       │ │ │
│  │  │  "CRUDP.PA"  │    │  1. eia_client → fundamentals     │ │ │
│  │  │  "BZ=F"      │    │  2. oil_bench_model → LLM decision│ │ │
│  │  └──────────────┘    │  3. Inject into consensus         │ │ │
│  │                      └───────────────────────────────────┘ │ │
│  │  ┌──────────────┐    ┌───────────────────────────────────┐ │ │
│  │  │  "^NDX"      │───▶│  STANDARD PATH (unchanged)        │ │ │
│  │  │  "SXRV.DE"   │    │  Vincent Ganne + existing models  │ │ │
│  │  │  "QQQ"       │    │  EIA: FORBIDDEN / SKIPPED         │ │ │
│  │  └──────────────┘    └───────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│             EnhancedDecisionEngine.make_enhanced_decision()      │
│             (consensus avec ou sans vote oil_bench)              │
└───────────────────────────────────────────────────────────────────┘
```

### 1.1 Isolation Contract

| Principle | Implementation |
|-----------|---------------|
| **Execution Gate** | `is_oil_ticker()` returns `True` only for `CL=F`, `CRUDP.PA`, `BZ=F`. All NASDAQ tickers (`^NDX`, `SXRV.DE`, `QQQ`) return `False` → EIA module is never invoked |
| **Data Isolation** | EIA data lives in `data_cache/eia/` (separate from `data_cache/macro/`). Cache files are prefixed `eia_` |
| **No Cross-Imports** | `eia_client.py` imports only stdlib + `requests` + `pandas` + `python-dotenv`. `oil_bench_model.py` imports `eia_client`, `llm_client._query_ollama`, `data._yf_download`. Neither imports from `classic_model`, `sentiment_analysis`, `features`, or `timesfm_model` |
| **Interface Contract** | `oil_bench_model.analyze()` returns the **exact same dict shape** `{"signal": str, "confidence": float, "analysis": str}` used by all other models. No schema extension required |
| **Conditional Injection** | The `oil_bench` vote is injected into `ModelDecision` list **only** when `is_oil_ticker()` is True. The weight is `0.0` otherwise. Zero impact on NASDAQ consensus math |

---

## 2. New File: `src/eia_client.py`

### 2.1 Responsibilities
- HTTP client for EIA API v2
- Fetches crude oil inventories, supply/demand data, and Short-Term Energy Outlook
- Handles caching, rate limiting, and error recovery
- Formats data into structured text for LLM consumption

### 2.2 EIA API Endpoints Used

Based on the swagger (`eia-api-swagger.yaml`) and EIA v2 documentation:

| Endpoint | Route | Data | Frequency |
|----------|-------|------|-----------|
| Crude Oil Inventories | `/v2/petroleum/stoc/wstk/data` | US commercial crude stocks (EIA-812) | Weekly |
| Crude Oil Production | `/v2/petroleum/sum/wtyp/data` | Field production of crude oil | Monthly |
| Crude Oil Imports | `/v2/crude-oil-imports/data` | Net imports by country | Monthly |
| ST Energy Outlook | `/v2/steo/data` | WTI price forecast, global demand | Monthly |

### 2.3 Class Design

```python
# src/eia_client.py

@dataclass
class EIACacheEntry:
    data: pd.DataFrame
    fetched_at: datetime
    ttl_hours: int

class EIAClient:
    BASE_URL = "https://api.eia.gov/v2"
    
    OIL_TICKERS = frozenset({"CL=F", "CRUDP.PA", "BZ=F", "CL", "BZ"})
    
    def __init__(self):
        self.api_key: str  # from EIA_API_KEY env var
        self._cache: dict[str, EIACacheEntry]
        self._cache_dir: Path  # data_cache/eia/
    
    @staticmethod
    def is_oil_ticker(ticker: str) -> bool:
        """Gate function - returns True ONLY for oil-related tickers."""
        return any(t in ticker for t in ("CL=F", "CRUDP", "BZ=F", "CL="))
    
    def get_fundamental_context(self) -> dict:
        """Main entry point. Returns all EIA fundamental data as a dict."""
        # Calls internal methods, catches all errors, always returns valid dict
    
    def get_crude_inventories(self, weeks: int = 8) -> pd.DataFrame:
        """Fetches last N weeks of US crude oil inventory data."""
        # GET /v2/petroleum/stoc/wstk/data?facets[duoarea]=R5X2
        #   &facets[product]=EPC0&frequency=weekly&sort[0][column]=period
        #   &sort[0][direction]=desc&length={weeks}
    
    def get_steo_outlook(self) -> pd.DataFrame:
        """Short-Term Energy Outlook - WTI price & global demand forecasts."""
        # GET /v2/steo/data?facets[seriesId]=...
    
    def get_crude_production(self) -> pd.DataFrame:
        """US crude oil field production."""
        # GET /v2/petroleum/sum/wtyp/data?facets[product]=EPC0...
    
    def get_imports(self) -> pd.DataFrame:
        """US crude oil imports."""
        # GET /v2/crude-oil-imports/data?...
    
    def format_for_llm(self, data: dict) -> str:
        """Converts raw EIA data into human-readable text for LLM prompt."""
        # Returns structured text block like:
        # "EIA Fundamental Data (as of 2026-04-16):
        #  - US Crude Inventories: 435.2M barrels (+2.1M WoW, +5.3% above 5yr avg)
        #  - Weekly Change: +2.1M barrels (Bearish signal: builds > expected)
        #  - US Production: 13.2M bbl/day (+0.1M MoM)
        #  - Net Imports: 6.1M bbl/day
        #  - STEO WTI Forecast: $72/bbl (next quarter avg)
        #  - Supply/Demand Balance: +0.8M bbl/day surplus"
    
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Core HTTP request with retry, timeout, and error handling."""
    
    def _get_from_cache(self, key: str) -> pd.DataFrame | None:
        """Check in-memory cache, then disk cache."""
    
    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """Save to both in-memory and disk cache."""
```

### 2.4 Error Handling Strategy

```
Network Error (ConnectionError/Timeout)
  ├─ Retry x3 with exponential backoff (2s, 4s, 8s)
  ├─ On final failure: fall back to disk cache (data_cache/eia/*.parquet)
  └─ If no cache: log warning, return empty DataFrame → empty context string

API Error (401/403/429/5xx)
  ├─ 401/403: log critical "EIA_API_KEY invalid", return empty → module degrades gracefully
  ├─ 429: log warning "rate limited", fall back to cache
  └─ 5xx: retry x2, fall back to cache

Malformed Response (JSON decode error, missing keys)
  ├─ Log error with response snippet
  ├─ Fall back to cache
  └─ If no cache: return empty DataFrame

Data Validation Error (negative inventories, implausible values)
  ├─ Log warning with specific values
  ├─ Filter out invalid rows
  └─ Return validated subset
```

### 2.5 Cache Strategy

| Layer | Location | TTL | Purpose |
|-------|----------|-----|---------|
| In-memory | `self._cache` dict | Per-run | Avoid duplicate API calls in same session |
| Disk | `data_cache/eia/{endpoint}_{frequency}.parquet` | 6h (inventories), 24h (STEO), 12h (production) | Survive restarts, offline fallback |

---

## 3. New File: `src/oil_bench_model.py`

### 3.1 Responsibilities
- Acts as "Commodity Quantitative Analyst" via LLM
- Aggregates: price data (yfinance) + fundamentals (EIA) + news (existing) + DXY
- Generates a single `{"signal", "confidence", "analysis"}` decision via local Ollama
- Translates LLM allocation (0-100%) into BUY/SELL/HOLD signal

### 3.2 Class Design

```python
# src/oil_bench_model.py

from dataclasses import dataclass
from eia_client import EIAClient
from llm_client import _query_ollama, TEXT_LLM_MODEL, OLLAMA_API_URL

@dataclass
class OilBenchConfig:
    wti_ticker: str = "CL=F"
    dxy_ticker: str = "DX-Y.NYB"
    brent_ticker: str = "BZ=F"
    allocation_buy_threshold: float = 55.0   # > 55% → BUY
    allocation_sell_threshold: float = 45.0  # < 45% → SELL
    # 45-55% → HOLD
    lookback_days: int = 5

class OilBenchModel:
    """
    Commodity Quant Analyst Model for WTI.
    
    INVARIANT: This model MUST only be instantiated and called when
    EIAClient.is_oil_ticker(ticker) == True. The caller is responsible
    for this check. The model will refuse to analyze non-oil tickers.
    """
    
    def __init__(self, config: OilBenchConfig = None):
        self.config = config or OilBenchConfig()
        self.eia_client = EIAClient()
    
    def analyze(self, ticker: str, headlines: list = None) -> dict:
        """
        Main analysis entry point.
        
        Args:
            ticker: Must be an oil ticker (validated internally)
            headlines: Optional news headlines from existing news_fetcher
        
        Returns:
            {"signal": "BUY"|"SELL"|"HOLD", "confidence": float, "analysis": str}
        """
        # 1. Validate ticker
        if not EIAClient.is_oil_ticker(ticker):
            return {"signal": "HOLD", "confidence": 0.0, "analysis": "Non-oil ticker, OilBench skipped."}
        
        # 2. Collect price data (WTI, DXY, Brent)
        price_data = self._fetch_price_data()
        
        # 3. Collect EIA fundamentals
        eia_context = self.eia_client.get_fundamental_context()
        eia_text = self.eia_client.format_for_llm(eia_context)
        
        # 4. Build LLM prompt
        prompt = self._construct_prompt(price_data, eia_text, headlines)
        
        # 5. Query LLM
        llm_response = self._query_llm(prompt)
        
        # 6. Translate allocation to signal
        return self._translate_signal(llm_response)
    
    def _fetch_price_data(self) -> dict:
        """Fetch WTI, Brent, DXY price data via yfinance."""
        # Uses data._yf_download or direct yf.download
        # Returns dict with current prices, % changes, trend info
    
    def _construct_prompt(self, price_data: dict, eia_text: str, headlines: list) -> str:
        """Build the Oil Bench LLM prompt."""
        # Specialized commodity analyst prompt:
        # - Price context (WTI, Brent spread, DXY correlation)
        # - EIA fundamental context (inventories, production, imports)
        # - News headlines (oil-specific)
        # - Request JSON: {"allocation": float 0-100, "reasoning": str}
    
    def _query_llm(self, prompt: str) -> dict:
        """Send to Ollama via existing _query_ollama helper."""
        payload = {
            "model": TEXT_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "system": "You are a senior commodity quantitative analyst...",
        }
        return _query_ollama(payload, expected_keys=["allocation", "reasoning"])
    
    def _translate_signal(self, llm_response: dict) -> dict:
        """Convert 0-100 allocation to BUY/SELL/HOLD with confidence."""
        allocation = llm_response.get("allocation", 50.0)
        reasoning = llm_response.get("reasoning", "")
        
        if allocation > self.config.allocation_buy_threshold:
            signal = "BUY"
            confidence = min(1.0, (allocation - 50) / 50)  # 55→0.10, 100→1.0
        elif allocation < self.config.allocation_sell_threshold:
            signal = "SELL"
            confidence = min(1.0, (50 - allocation) / 50)  # 45→0.10, 0→1.0
        else:
            signal = "HOLD"
            confidence = 0.3  # Low confidence for neutral zone
        
        return {
            "signal": signal,
            "confidence": round(confidence, 3),
            "analysis": f"[OilBench] Alloc={allocation:.0f}% | {reasoning[:200]}"
        }
```

### 3.3 LLM Prompt Template

```
You are a senior commodity quantitative analyst specializing in WTI Crude Oil.
Analyze the following data to determine your recommended portfolio allocation (0-100%).

**Price Context:**
- WTI Spot: ${wti_price:.2f} ({wti_change:+.2f}% 5d)
- Brent Spot: ${brent_price:.2f} (Spread: ${spread:.2f})
- DXY: {dxy:.2f} ({dxy_change:+.2f}% 5d)

**EIA Fundamental Data:**
{eia_formatted_text}

**Recent Oil News:**
{headlines_formatted}

**Analysis Framework:**
1. Inventory Analysis: Compare current stocks to 5-year average. Builds > expected = bearish.
2. Supply/Demand Balance: Production trends, import levels, refinery utilization.
3. DXY Impact: Stronger dollar = downward pressure on oil prices.
4. Brent-WTI Spread: Widening spread signals supply chain dynamics.
5. Forward Guidance: STEO outlook for price trajectory.

Return ONLY a JSON object:
{"allocation": <float 0-100>, "reasoning": "<2-sentence analysis>"}
```

---

## 4. Modifications to Existing Files

### 4.1 `src/enhanced_trading_example.py`

**File**: `src/enhanced_trading_example.py`  
**Lines affected**: ~6 lines added in `perform_enhanced_analysis()`

```python
# ADD at top of file (after existing imports):
from eia_client import EIAClient
from oil_bench_model import OilBenchModel

# MODIFY in perform_enhanced_analysis(), after the existing Vincent Ganne block
# (~line 415-419, after `effective_vg_indicators` is computed):

        # --- OIL BENCH MODULE (WTI-ONLY) ---
        oil_bench_decision = None
        if EIAClient.is_oil_ticker(self.analysis_ticker):
            try:
                oil_model = OilBenchModel()
                oil_bench_decision = oil_model.analyze(
                    ticker=self.analysis_ticker,
                    headlines=None  # headlines available via model_predictions if needed
                )
                logger.info(f"OilBench signal: {oil_bench_decision['signal']} "
                           f"(conf={oil_bench_decision['confidence']:.2f})")
            except Exception as e:
                logger.error(f"OilBench model failed (isolated): {e}")
                oil_bench_decision = None
        # --- END OIL BENCH ---
```

Then pass `oil_bench_decision` to `make_enhanced_decision()`.

### 4.2 `src/enhanced_decision_engine.py`

**File**: `src/enhanced_decision_engine.py`  
**Lines affected**: ~15 lines added in `make_enhanced_decision()`

```python
# ADD parameter to make_enhanced_decision():
    oil_bench_decision: Dict = None,

# ADD after the Vincent Ganne block (~line 443-454):
        if oil_bench_decision:
            decisions.append(
                ModelDecision(
                    signal=oil_bench_decision.get('signal', 'HOLD'),
                    confidence=oil_bench_decision.get('confidence', 0.0),
                    strength=self._normalize_signal(oil_bench_decision.get('signal', 'HOLD')),
                    timestamp=timestamp,
                    model_name='oil_bench',
                    reasoning=oil_bench_decision.get('analysis', 'Oil Bench commodity analysis')
                )
            )

# MODIFY __init__ base_weights to include oil_bench:
        self.base_weights = base_weights or {
            'classic': 0.10,
            'llm_text': 0.20,
            'llm_visual': 0.10,
            'sentiment': 0.10,
            'timesfm': 0.25,
            'vincent_ganne': 0.15,
            'oil_bench': 0.10   # NEW: only used when oil_bench_decision is provided
        }
        # Adjusted: vincent_ganne from 0.20→0.15, llm_text from 0.25→0.20
        # to accommodate oil_bench without exceeding 1.0
```

**Critical isolation note**: When `oil_bench_decision` is `None` (NASDAQ tickers), no `ModelDecision` with `model_name='oil_bench'` is appended. The `oil_bench` weight in `base_weights` is simply unused in the weighted score calculation (line 459: `weights.get(decision.model_name, 0.25)` — since there's no decision with `model_name='oil_bench'`, this weight is never applied). The total effective weight for NASDAQ remains ~1.0 across the existing models.

### 4.3 `src/database.py`

**File**: `src/database.py`  
**Line affected**: 1 line change

```python
# MODIFY line 54 (model_signals table CHECK constraint):
# OLD:
model_type TEXT NOT NULL CHECK(model_type IN ('classic', 'llm_text', 'llm_visual', 'sentiment', 'hybrid')),
# NEW:
model_type TEXT NOT NULL CHECK(model_type IN ('classic', 'llm_text', 'llm_visual', 'sentiment', 'hybrid', 'oil_bench', 'vincent_ganne', 'timesfm')),
```

**Migration**: Add a migration function since existing DBs won't have the updated CHECK:

```python
def _migrate_model_signals_table():
    """Adds new model types to the CHECK constraint on model_signals."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    try:
        # SQLite doesn't support ALTER CONSTRAINT, so we recreate
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='model_signals'")
        row = cursor.fetchone()
        if row and 'oil_bench' not in row[0]:
            cursor.execute("DROP TABLE IF EXISTS model_signals_old")
            cursor.execute("ALTER TABLE model_signals RENAME TO model_signals_old")
            cursor.execute('''CREATE TABLE model_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                model_type TEXT NOT NULL CHECK(model_type IN ('classic', 'llm_text', 'llm_visual', 'sentiment', 'hybrid', 'oil_bench', 'vincent_ganne', 'timesfm')),
                signal TEXT NOT NULL CHECK(signal IN ('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL')),
                confidence REAL,
                details TEXT
            )''')
            cursor.execute('''INSERT INTO model_signals SELECT * FROM model_signals_old''')
            cursor.execute("DROP TABLE model_signals_old")
            conn.commit()
            logger.info("Migrated model_signals table: added new model types")
    except Exception as e:
        logger.warning(f"Migration check for model_signals: {e}")
    finally:
        conn.close()
```

Call `_migrate_model_signals_table()` from `init_db()`.

---

## 5. Data Structures

### 5.1 EIA API Response (Expected Shape)

```json
{
  "response": {
    "total": 8,
    "dateFormat": "YYYY-MM-DD",
    "frequency": "weekly",
    "data": [
      {
        "period": "2026-04-10",
        "duoarea": "R5X2",
        "product": "EPC0",
        "value": 435.214,
        "units": "million barrels"
      }
    ]
  }
}
```

### 5.2 Internal Fundamental Data (EIAClient.get_fundamental_context())

```python
{
    "inventories": {
        "current": 435.2,          # million barrels
        "previous": 433.1,
        "wow_change": 2.1,
        "wow_change_pct": 0.48,
        "vs_5yr_avg_pct": 5.3,     # % above/below 5yr average
        "history": pd.DataFrame,    # last 8 weeks
    },
    "production": {
        "current": 13.2,           # million bbl/day
        "mom_change_pct": 0.76,
    },
    "imports": {
        "current": 6.1,            # million bbl/day
        "mom_change_pct": -2.3,
    },
    "steo": {
        "wti_forecast_next_q": 72.0,
        "global_demand_forecast": 102.5,
        "demand_change_pct": 1.2,
    },
    "as_of": "2026-04-16T10:00:00",
}
```

### 5.3 Oil Bench Output (matches existing contract)

```python
{
    "signal": "BUY",           # "BUY" | "SELL" | "HOLD"
    "confidence": 0.35,        # float 0.0-1.0
    "analysis": "[OilBench] Alloc=67% | Inventories below 5yr avg by 3.2%, production declining, STEO forecasts $74/bbl. Bullish supply-demand balance."
}
```

---

## 6. Execution Flow (WTI Run)

```
main.py --ticker CRUDP.PA --t212
  │
  ├─ EnhancedTradingSystem(ticker="CRUDP.PA")
  │    └─ analysis_ticker = "CL=F"  (via ANALYSIS_MAPPING)
  │
  ├─ prepare_data_and_features()
  │    ├─ get_etf_data("CL=F") → hist_data, VIX
  │    ├─ create_technical_indicators() → RSI, MACD, BB, etc.
  │    └─ get_vincent_ganne_indicators() → all cross-asset prices
  │
  ├─ train_classic_model() → model, scaler
  │
  ├─ get_model_predictions()
  │    ├─ classic → prediction, confidence
  │    ├─ LLM text → signal, confidence, analysis
  │    ├─ LLM visual → signal, confidence, analysis
  │    ├─ sentiment → signal, confidence
  │    └─ TimesFM → signal, confidence
  │
  ├─ perform_enhanced_analysis()
  │    │
  │    │  ★ GATE: EIAClient.is_oil_ticker("CL=F") → True
  │    │
  │    ├─ [NEW] OilBenchModel.analyze(ticker="CL=F")
  │    │    ├─ EIAClient.get_fundamental_context()
  │    │    │    ├─ get_crude_inventories(weeks=8)   ← EIA API call (cached)
  │    │    │    ├─ get_crude_production()            ← EIA API call (cached)
  │    │    │    ├─ get_imports()                     ← EIA API call (cached)
  │    │    │    ├─ get_steo_outlook()                ← EIA API call (cached)
  │    │    │    └─ return fundamental_context dict
  │    │    ├─ _fetch_price_data() → WTI, Brent, DXY from yfinance
  │    │    ├─ _construct_prompt(price_data, eia_text, headlines)
  │    │    ├─ _query_llm(prompt) → Ollama (local)
  │    │    └─ _translate_signal() → {"signal", "confidence", "analysis"}
  │    │
  │    │  Vincent Ganne: DISABLED for oil tickers
  │    │  (existing code at line 416-419 already handles this)
  │    │
  │    ├─ EnhancedDecisionEngine.make_enhanced_decision(
  │    │    ..., oil_bench_decision=oil_bench_decision
  │    │  )
  │    │    └─ decisions list includes oil_bench ModelDecision
  │    │    └─ Weighted consensus includes oil_bench (weight=0.10)
  │    │
  │    └─ Risk management, position sizing, etc. (unchanged)
  │
  └─ Execute trade via T212 / log to CSV
```

### 6.1 Execution Flow (NASDAQ Run) — Unchanged

```
main.py --ticker SXRV.DE --t212
  │
  ├─ EnhancedTradingSystem(ticker="SXRV.DE")
  │    └─ analysis_ticker = "^NDX"
  │
  ├─ ... (same as before, no EIA imports loaded at module level)
  │
  ├─ perform_enhanced_analysis()
  │    │
  │    │  ★ GATE: EIAClient.is_oil_ticker("^NDX") → False
  │    │
  │    │  OilBenchModel: NOT INSTANTIATED
  │    │  EIA API: NOT CALLED
  │    │  oil_bench_decision = None
  │    │
  │    │  Vincent Ganne: ACTIVE (normal behavior)
  │    │
  │    ├─ EnhancedDecisionEngine.make_enhanced_decision(
  │    │    ..., oil_bench_decision=None
  │    │  )
  │    │    └─ decisions list is IDENTICAL to current behavior
  │    │    └─ oil_bench weight (0.10) is NEVER applied
  │    │
  │    └─ ... (unchanged)
```

---

## 7. New File: `tests/test_eia_client.py`

### Test Categories

1. **Unit: `is_oil_ticker()` gate**
   - `assert EIAClient.is_oil_ticker("CL=F") == True`
   - `assert EIAClient.is_oil_ticker("CRUDP.PA") == True`
   - `assert EIAClient.is_oil_ticker("BZ=F") == True`
   - `assert EIAClient.is_oil_ticker("^NDX") == False`
   - `assert EIAClient.is_oil_ticker("SXRV.DE") == False`
   - `assert EIAClient.is_oil_ticker("QQQ") == False`
   - `assert EIAClient.is_oil_ticker("") == False`

2. **Unit: API response parsing**
   - Mock `requests.get` → parse inventory JSON → validate DataFrame shape
   - Mock malformed response → graceful degradation → empty DataFrame

3. **Unit: Cache behavior**
   - First call → API hit + cache write
   - Second call (within TTL) → cache hit, no API call
   - Expired cache → API hit + cache refresh

4. **Unit: `format_for_llm()` output**
   - Given known data dict → verify output string contains expected metrics

5. **Integration: Error scenarios**
   - No API key → critical log, empty context
   - Network timeout → retry, cache fallback
   - Rate limit (429) → cache fallback
   - Invalid JSON → parse error handling

## 8. New File: `tests/test_oil_bench_model.py`

### Test Categories

1. **Unit: Ticker validation**
   - `OilBenchModel().analyze("CL=F")` → produces valid signal dict
   - `OilBenchModel().analyze("^NDX")` → returns HOLD + confidence=0 + "skipped"

2. **Unit: Signal translation**
   - allocation=70 → BUY, confidence≈0.40
   - allocation=55 → BUY, confidence≈0.10
   - allocation=50 → HOLD, confidence=0.30
   - allocation=45 → SELL, confidence≈0.10
   - allocation=20 → SELL, confidence≈0.60
   - allocation=0 → SELL, confidence=1.00

3. **Unit: LLM response handling**
   - Valid JSON → correct parsing
   - Missing "allocation" key → default to 50 (HOLD)
   - Invalid allocation (negative, >100) → clamp to 0-100

4. **Integration: Full analyze() with mocked dependencies**
   - Mock EIAClient, mock _query_ollama → validate end-to-end flow

---

## 9. File Summary

| File | Action | Lines Changed (est.) |
|------|--------|---------------------|
| `src/eia_client.py` | **CREATE** | ~250 lines |
| `src/oil_bench_model.py` | **CREATE** | ~180 lines |
| `tests/test_eia_client.py` | **CREATE** | ~150 lines |
| `tests/test_oil_bench_model.py` | **CREATE** | ~120 lines |
| `src/enhanced_trading_example.py` | **MODIFY** | +12 lines |
| `src/enhanced_decision_engine.py` | **MODIFY** | +18 lines |
| `src/database.py` | **MODIFY** | +25 lines (migration) |
| **Total** | | ~755 lines |

No changes to: `data.py`, `llm_client.py`, `classic_model.py`, `features.py`, `timesfm_model.py`, `sentiment_analysis.py`, `news_fetcher.py`, `web_researcher.py`, `t212_executor.py`, `main.py`, `schedule.py`.

---

## 10. Implementation Order

| Step | Task | Dependency | Risk |
|------|------|------------|------|
| 1 | Create `src/eia_client.py` with cache, error handling, all 4 endpoints | EIA_API_KEY in .env (done) | Low |
| 2 | Create `tests/test_eia_client.py` and validate against live EIA API | Step 1 | Low |
| 3 | Create `src/oil_bench_model.py` with LLM prompt, signal translation | Step 1 + Ollama running | Medium |
| 4 | Create `tests/test_oil_bench_model.py` | Step 3 | Low |
| 5 | Modify `src/database.py` — add migration for model_signals | None | Low |
| 6 | Modify `src/enhanced_decision_engine.py` — add oil_bench param + weight | Step 3 | Medium |
| 7 | Modify `src/enhanced_trading_example.py` — add OilBench gate + call | Steps 1-3, 6 | Medium |
| 8 | Run `uv run ruff check src/` and `uv run pytest tests/` | Steps 1-7 | Low |
| 9 | Manual E2E test: `uv run main.py --ticker CRUDP.PA --simul` | Steps 1-8 | Medium |
| 10 | Verify no regression: `uv run main.py --ticker SXRV.DE --simul` | Step 9 | Critical |

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| EIA API down or key revoked | Graceful degradation: empty context → OilBench returns HOLD. System functions with existing models |
| Ollama LLM produces invalid allocation | `_translate_signal()` defaults to 50 (HOLD) on parse failure. `_query_ollama` already has 3x retry |
| OilBench model exception crashes analysis | Wrapped in try/except in `perform_enhanced_analysis()`. On failure: `oil_bench_decision = None`, analysis proceeds without it |
| Weight rebalance affects NASDAQ | `oil_bench` weight only applied when oil_bench_decision is non-None. Total weight normalization already handles missing models |
| DB migration breaks existing data | Migration preserves all existing rows. Transaction + rollback on failure |
| Scheduler runs both tickers sequentially | Each ticker gets fresh `EnhancedTradingSystem` instance. No shared mutable state between runs |

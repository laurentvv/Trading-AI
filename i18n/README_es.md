<p align="center">
  <a href="README.md">English</a> |
  <a href="i18n/README_zh.md">中文</a> |
  <a href="i18n/README_hi.md">हिंदी</a> |
  <a href="i18n/README_es.md">Español</a> |
  <a href="i18n/README_fr.md">Français</a> |
  <a href="i18n/README_ar.md">العربية</a> |
  <a href="i18n/README_bn.md">বাংলা</a> |
  <a href="i18n/README_ru.md">Русский</a> |
  <a href="i18n/README_pt.md">Português</a> |
  <a href="i18n/README_id.md">Bahasa Indonesia</a>
</p>

<p align="center">
  <img src="assets/banner.png" alt="Banner de Trading IA Híbrido" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Sistema de Trading con IA Híbrida 📈</h1>
  <p>
    Un sistema experto de apoyo a la decisión para el trading de ETFs de NASDAQ y Petróleo (WTI), aprovechando una inteligencia artificial híbrida de 12 modelos para señales de trading robustas y matizadas.
  </p>
</div>

<div align="center">

[![Estado del proyecto](https://img.shields.io/badge/status-en--desarrollo-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Versión de Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Licencia](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Tabla de Contenidos

- [🌟 Acerca del proyecto](#-acerca-del-proyecto)
  - [🚀 Estrategia Dual-Ticker (Análisis vs. Trading)](#-estrategia-dual-ticker-análisis-vs-trading)
  - [🧠 Motor de IA Híbrido](#-motor-de-ia-híbrido)
  - [🧘 Filosofía de Decisión: "Prudencia Cognitiva"](#-filosofía-de-decisión--prudencia-cognitiva)
  - [✨ Características clave](#-características-clave)
  - [💻 Stack tecnológico](#-stack-tecnológico)
  - [⚙️ Rendimiento y Hardware](#️-rendimiento-y-hardware)
  - [🧠 Arquitectura de IA y LLM (Gemini + Fallback Local)](#-arquitectura-de-ia-y-llm-gemini--fallback-local)
  - [🧠 FinAcumen (Memoria Financiera)](#-finacumen-memoria-financiera)
- [📂 Estructura del proyecto](#-estructura-del-proyecto)
- [🚀 Inicio rápido](#-inicio-rápido)
  - [✅ Requisitos previos](#-requisitos-previos)
  - [⚙️ Instalación](#️-instalación)
- [🛠️ Uso](#️-uso)
  - [Modo Simulación (Paper Trading)](#modo-simulación-paper-trading)
  - [Ejecución Real (Trading 212)](#ejecución-real-trading-212)
- [🧪 Backtesting en Producción](#-backtesting-en-producción)
  - [Características](#características)
  - [Uso](#uso)
- [🤝 Contribuir](#-contribuir)
- [📜 Licencia](#-licencia)
- [📧 Contacto](#-contacto)

---

## 🌟 Acerca del proyecto

Un sistema experto de apoyo a la decisión para el trading de ETFs de NASDAQ y Petróleo (WTI), aprovechando una inteligencia artificial híbrida de 12 modelos.

### 🚀 Estrategia Dual-Ticker (Análisis vs. Trading)

El sistema **analiza un índice global** (p. ej. `^NDX` para el Nasdaq-100, `CL=F` para el WTI) pero **ejecuta en un ETF cotizado en EUR** (p. ej. `SXRV.DE`, `CRUDP.PA`). Esta disociación garantiza un análisis sobre datos de alta fidelidad y una ejecución real sobre activos accesibles vía Trading 212.

### 🧠 Motor de IA Híbrido

El motor combina modelos heterogéneos en un **consenso ponderado**:

1. **Modelos Scikit-Learn** (RandomForest, GradientBoosting, LogisticRegression) — validados con `TimeSeriesSplit` para evitar la fuga de datos. Señal cuantitativa agresiva (25 % del peso cognitivo).
2. **TimesFM 2.5** (Google Research) — modelo fundacional para el forecast de series temporales.
3. **TensorTrade / PPO** (stable-baselines3) — agente de Reinforcement Learning en un entorno Gymnasium a medida.
4. **Gemma 4 12B** (Ollama) — análisis **textual** (macro/noticias) y **visual** (charts técnicos); la **defensa JSON de dos capas** garantiza un JSON limpio a pesar del modo pensamiento `<|think|>` activo.
5. **Análisis de Sentimiento** híbrido (Alpha Vantage + AlphaEar + Hyperliquid).
6. **Vincent Ganne Model** — cerrojo geopolítico (WTI, Brent, Gas, Urea, DXY) que genera señales de COMPRA únicamente para validar los mínimos del Nasdaq.
7. **OilBenchModel** — modelo cognitivo especializado para el WTI (indicadores técnicos + fundamentales EIA + sentimiento).

### 🧘 Filosofía de Decisión: "Prudencia Cognitiva"

Los modelos cognitivos (Gemma 4, sentimiento, Vincent Ganne) poseen el **75 %** del peso de decisión, frente al **25 %** del modelo cuantitativo agresivo. Esta sobreponderación deliberada garantiza que el contexto cualitativo templa las señales cuantitativas. Una señal solo se ejecuta si la confianza global supera el **40 %**; entre el 20 % y el 40 %, se degrada a HOLD.

### ✨ Características clave

- **Arquitectura LLM Híbrida Cloud/Local**: integración `free-llm-api-keys` para aprovechar "Frontier Models" muy inteligentes (DeepSeek, Claude, Gemini) para el análisis textual, con un fallback 100 % robusto al Ollama local (que sigue siendo el motor exclusivo para los charts visuales).
- **Enfoque Dual-Ticker**: analizar el índice, operar el ETF.
- **Precios en vivo T212**: recuperación en tiempo real de precios EUR vía la API de Trading 212 (0,2 s), con fallback yfinance y caché parquet.
- **Spread Brent Dated**: seguimiento de la tensión del mercado físico vía el spread entre el Brent Spot (Dated) y el Brent de futuros.
- **Resiliencia de red**: circuit breaker yfinance con trackers separados (info vs. download), timeout de 10 s en todas las llamadas de red.
- **Auto-invalidación de caché**: la caché Parquet detecta su caducidad (> 2 días) y fuerza una actualización. Use `refresh_cache.py` para un vaciado manual.
- **Paralelización de llamadas LLM**: las llamadas de modelos independientes (`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`) se ejecutan en un `ThreadPoolExecutor` para superponer la inferencia Ollama con la E/S. Camino crítico típicamente 4–6 min en CPU frente a 10+ min en secuencia.
- **Caché de 24h de consultas de búsqueda**: la consulta de búsqueda web generada por el LLM se cachea en `data_cache/search_queries/<ticker>_<date>_<price-sig>.json`. Clave por fecha + firma de acción de precios (bucketing log2 del close + bucket RSI), por lo que un cambio de régimen la invalida. Las consultas de fallback **nunca** se cachean (un fallo transitorio de Ollama no puede envenenar la caché durante 24h).
- **Timeout estricto de ciclo**: cada ciclo por ticker se envuelve en un presupuesto de 40 min (`CYCLE_TIMEOUT_SECONDS` en `main.py`). En timeout, el hilo de trabajo se `shutdown(wait=F)` para que el siguiente ticker arranque inmediatamente; se aplica HOLD al ticker expirado. Los futures individuales tienen sus propios timeouts por tarea (búsqueda 240 s, visual 300 s, texto 240 s, modelos CPU 180 s cada uno, noticias 90 s, crawl web 30 s).
- **Seguridad anti-hilo huérfano**: en timeout de ciclo, un `threading.Event` por ticker se activa para que el worker huérfano aborte antes de cualquier llamada `execute_t212_trade` — impidiendo operaciones con dinero real tras haber mostrado al usuario el panel "HOLD aplicado". Un `threading.Lock` por ticker además serializa la colocación de órdenes T212, eliminando el riesgo de doble operación bajo solapamiento del scheduler o invocaciones `--ticker` duplicadas.
- **Centinela de fallo LLM**: cuando `_query_ollama` agota todos sus reintentos, el diccionario de fallback lleva un marcador `"failed": True` para que la lógica de consenso aguas abajo pueda distinguir "el modelo eligió HOLD" de "el modelo crasheó" (actualmente propagado pero no filtrado — un seguimiento conocido).
- **Cognición avanzada**: uso de **Gemma 4 12B** con **defensa JSON de dos capas**:
  1. **Aplicación del esquema en servidor** (`format: SCHEMA_*` con `additionalProperties: false`) — la capa portante; pasada vía el parámetro `format` de Ollama en cada sitio de llamada. Esquemas definidos en `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`).
  2. **Sufijo defensivo del system prompt** (`"...never add a 'thought' key."`) — segunda línea redundante-pero-inofensiva, conservada como belt-and-braces ante cualquier regresión futura de la capa esquema.

  El token de razonamiento `<|think|>` está **activo** en los cuatro system prompts de producción (reactivado el 2026-06-06 en `main` tras validación en la rama `think-mode`). Es la capa esquema la que realmente neutraliza el defecto histórico de escombros JSON `<|channel>thought` (causa raíz mayo 2026): `tests/check_llm_json.py` confirma que los casos schema-strict (`v3_schema`, `v6_schema`, `v7_schema_strict`) producen JSON limpio incluso con `<|think|>` activo, mientras que las variantes loose `format:json` fallan. Vea `docs/ADR-001-think-mode-dual-layer-defence.md` para el análisis completo y el procedimiento de reversal.
- **Agente autónomo de Morning Brief**: un workflow nocturno basado en `smolagents` (`morning_brief/morning_brief.py`) programado automáticamente a las 01:00 vía `schedule.py`. Rastrea independientemente los logs diarios de la API, descarga datos fundamentales de inventario EIA, y arbitra un debate *Bull vs Bear*. El informe markdown resultante (`morning_market_brief.md`) se inyecta automáticamente en el system prompt del LLM textual durante el ciclo de trading diario, otorgando a la IA principal una memoria contextual profunda y una conciencia fundamental sin ralentizar la ejecución en mercado en vivo.
- **🏛️ Weekend Council (Memoria Estratégica)**: una retrospectiva LLM multi-persona semanal (`src/council/weekend_council.py`) que se ejecuta cada **sábado a las 01:00** vía `schedule.py`. Seis personas — cada una en una **familia de modelo Ollama distinta** (Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / LFM 2.5 / Mistral Nemo 12B) para una diversidad de razonamiento genuina — deliberan en un protocolo de 4 rondas (Problem Restate Gate → Analysis con STANCE explícita → Debate 1-vs-1 → Síntesis del Juez) con mecanismos anti-groupthink (cuota de disidencia, veredicto unresolved-first). El Juez (Qwen3.5-9B-MTP) emite una postura por ticker que se convierte en el **11.º voto ponderado** (9,5 %) del consenso en tiempo real, con una confianza que decae linealmente en 7 días. Presupuestos de tokens generosos (`num_predict` hasta 12000, `num_ctx` hasta 65536) y una ventana de scheduler de 48 horas acomodan los thinking models en CPU. El council analiza datos PROD reales: precisión de modelos (`model_performance.db`), métricas de cartera y alertas críticas (`performance_monitor.db`), y el diario de trading ejecutado. Instale los 6 modelos requeridos con `uv run python setup_council_models.py`. Vea `docs/ADR-003-weekend-council-11th-voice.md`.
- **Noticias y Sentimiento Blockchain**: integración de **AlphaEar** y **Hyperliquid** para capturar el sentimiento social y especulativo.
- **Scheduler automatizado**: script `schedule.py` para ejecución continua (8:30 – 18:00) en un servidor.
- **Gestión de riesgo centralizada**: el `AdvancedRiskManager` centraliza la lógica Anti-Loss (Stop-Loss) y Trailing Stop. Los modelos individuales ya no gestionan estos riesgos, garantizando una estrategia unificada y estricta de protección del capital a través de los regímenes de mercado.
- **Contratos de datos estrictos**: todos los modelos de IA están totalmente estandarizados para devolver una dataclass fuertemente tipada `ModelResult` (`signal`, `confidence`, `reasoning`), asegurando 100 % de uniformidad a través del motor de consenso.
- **Salud del código auditada**: el proyecto mantiene un estándar de salud de código **Grade B** vía auditorías automatizadas (0 código muerto, alto índice de mantenibilidad).
- **Backtesting de producción**: motor de backtest autónomo (`backtest_prod.py`) que reproduce las señales reales de prod contra precios reales con comisiones T212 — sin dependencias externas.
- **Control del dump de depuración**: establezca `TRADING_DEBUG_DUMP=0` para desactivar el dump (limitado a 5 MB) `data_cache/llm_debug_fail.txt` de fallos LLM.

### 💻 Stack tecnológico

- **Lenguaje**: `Python 3.12+`
- **Cálculos y Datos**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **IA y LLM**: `google-genai` (Gemini), `requests`, `ollama`
- **Scraping Web y Búsqueda**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualización**: `matplotlib` (backend Agg para thread safety), `seaborn`, `mplfinance`
- **Utilidades**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Rendimiento y Hardware
El sistema está diseñado para ser **rendimiento en hardware de consumo** sin requerir una GPU dedicada.
- **Solo CPU**: la inferencia LLM (Gemma 4 12B Q6_K vía Ollama) y TimesFM se ejecutan enteramente en CPU. El rendimiento es de ~3–4 tokens/s en una CPU moderna de 8 núcleos.
- **RAM recomendada**: 16 GB mínimo (32 GB sugeridos para ejecutar cómodamente Gemma 4 12B junto a TimesFM y TensorTrade).
- **Concurrencia Ollama**: establezca `OLLAMA_NUM_PARALLEL=8` (ya en el `.env` recomendado) para que múltiples llamadas LLM compartan la carga del modelo. Con el presupuesto de contexto por defecto de 4 GB, los slots paralelos obtienen ~512 tokens cada uno — Ollama serializará si los prompts superan el ctx por slot, pero el `ThreadPoolExecutor` mantiene el solapamiento wall-clock beneficioso para los pasos ligados a E/S (fetch de noticias, crawl web, modelos CPU).
- **Tiempo de ejecución**: ~6 a 9 minutos por ticker en CPU (en frío), ~3 a 5 minutos por ticker con un hit en la caché de consultas de búsqueda. La ejecución por defecto abarca dos tickers (CRUDP.PA + SXRV.DE), así que prevea ~15 min en total.
- **Timeout de ciclo**: cada ciclo por ticker está limitado a 40 min (`CYCLE_TIMEOUT_SECONDS`). Si se excede, se aplica HOLD y el siguiente ticker arranca inmediatamente.
- **Velocidad de API**: integración con Trading 212 ultrarrápida (<1 s para la recuperación del precio en vivo).

### 🧠 Arquitectura de IA y LLM (Gemini + Fallback Local)
El sistema aprovecha una arquitectura multinivel muy robusta para asegurar un uptime máximo y una toma de decisiones inteligente, profundamente integrada en `main.py` y el Weekend Council.

- **Cascada de Fallback de 4 Niveles**:
  1. **Nivel Gemini de Pago (`GEMINI_API_KEY_PAY`)**: Máxima prioridad. Usa modelos avanzados como Gemini 2.5 Pro para razonamiento complejo, visión de charts técnicos y decisiones finales de trading.
  2. **Nivel Gemini Gratuito (`GEMINI_API_KEY`)**: Usado para tareas más ligeras de alto volumen como la síntesis de contexto web.
  3. **Proxies de API LLM gratuitos**: Respaldo vía `free-llm-api-keys`.
  4. **Ollama Local**: Fallback en CPU offline 100 % robusto si todos los servicios cloud caen.
- **Protección de costes**: el nivel de pago está limitado por un presupuesto de coste de 30 días móvil (`GEMINI_PAY_MONTHLY_BUDGET_EUR`, por defecto 8,6 €/mes) — el coste de cada llamada se calcula a partir del uso real de tokens × el precio del modelo y se acumula; cuando se alcanza el presupuesto, las llamadas caen al nivel gratuito / Ollama. Un backstop diario (`GEMINI_PAY_DAILY_CAP`, por defecto 200) protege contra bucles descontrolados.
- **Integración**: el motor principal de ejecución diaria (`main.py`) usa Gemini para el consenso multimodelo en tiempo real, mientras que el Weekend Council asíncrono (`council`) integra Gemini específicamente para ciertos roles (como el Juez y el Escéptico) junto a diversos modelos Ollama locales.

### 🧠 FinAcumen (Memoria Financiera)
La arquitectura FinAcumen se ha integrado para dotar a los modelos de IA locales de una **memoria de experiencia** y de herramientas deterministas. Esto resuelve el problema de la amnesia de los LLMs.
- FinAcumen funciona **de forma asíncrona por la noche** (vía `schedule.py`) para beneficiarse de toda la potencia de la CPU sin bloquear los ciclos de trading.
- Su informe cualitativo profundo se añade automáticamente al **Morning Market Brief** para guiar al LLM de decisión a lo largo de la jornada de trading.

## 📂 Estructura del proyecto

El proyecto está organizado de forma modular para una mejor mantenibilidad.

```
Trading-AI/
├── morning_brief/                   # Agente autónomo nocturno de análisis fundamental profundo
│   ├── morning_brief.py             # Orquestador de agentes y configuración smolagents
│   └── output/                      # Informes markdown diarios generados (morning_market_brief.md)
├── src/                             # Módulos núcleo
│   ├── adaptive_weight_manager.py   # Ponderación dinámica de modelos según rendimiento
│   ├── advanced_risk_manager.py     # Gestión de riesgo Trend-Aware y sizing
│   ├── bootstrap.py                 # Lógica de inicialización del núcleo
│   ├── chart_generator.py           # Genera charts técnicos para el LLM visual
│   ├── classic_model.py             # Conjunto de modelos cuantitativos Scikit-learn
│   ├── config_weights.py            # Configuración de pesos base del motor híbrido
│   ├── data.py                      # Fetch, caché y preprocesamiento de datos
│   ├── database.py                  # Gestión de base SQLite para métricas
│   ├── eia_client.py                # Cliente API Energy Information Administration
│   ├── enhanced_decision_engine.py  # Motor de fusión híbrida que orquesta todos los modelos
│   ├── enhanced_trading_example.py  # Scripts de ejemplo de uso de los modelos
│   ├── features.py                  # Ingeniería de features técnicas y macroeconómicas
│   ├── grebenkov_model.py           # Modelo matemático Trend-Following (Agnostic Risk Parity)
│   ├── hmm_model.py                 # Hidden Markov Model para detección de régimen
│   ├── llm_client.py                # Integración Ollama para inferencia LLM local
│   ├── news_fetcher.py              # Crawl y parseo de noticias financieras
│   ├── oil_bench_model.py           # Modelo de trading WTI especializado en energía
│   ├── performance_monitor.py       # Seguimiento de precisión e historial de modelos
│   ├── read_simul.py                # Herramientas para leer salidas de simulación
│   ├── sentiment_analysis.py        # Integración de sentimiento Alpha Vantage y AlphaEar
│   ├── t212_executor.py             # Ejecución real API Trading 212 y cartera
│   ├── tensortrade_model.py         # Señal de Reinforcement Learning (PPO)
│   ├── timesfm_model.py             # Integración de forecast de series temporales TimesFM 2.5
│   └── web_researcher.py            # Scraping macroeconómico web con Crawl4AI
├── data_cache/                       # Todas las cachés (gitignorado)
│   ├── *.parquet                     # Datos OHLCV por ticker (yfinance)
│   ├── macro/                        # Series temporales macro (FRED, multifuentes)
│   ├── search_queries/               # Caché 24h de consultas de búsqueda LLM (por ticker+fecha+price-sig)
│   └── llm_debug_fail.txt            # Dump (limitado 5 MB) de fallos LLM — desactivar con TRADING_DEBUG_DUMP=0
├── tests/                            # Scripts de test y validación
│   ├── test_full_cycle.py            # Test end-to-end T212 compra/espera/venta
│   ├── test_enhanced_decision_engine.py # Tests del motor de fusión híbrida
│   ├── check_llm_json.py             # Diagnóstico JSON-schema LLM (testa los 4 sitios de llamada Ollama)
│   ├── check_live.py                 # Script de verificación de precios de mercado en vivo
│   └── ...                           # Otros tests unitarios y de integración
├── i18n/                            # Internacionalización (READMEs traducidos)
├── assets/                          # Recursos estáticos (imágenes, banners)
├── memory-bank/                     # Estado determinista de 4 archivos + contexto long-form (ver AGENTS.md §1)
├── backtest_prod.py                 # Motor de backtest de producción autónomo
├── main.py                          # Punto de entrada único (Análisis y Trading)
├── pyproject.toml                   # Dependencias y configuración del proyecto (uv)
├── refresh_cache.py                 # Utilidad CLI para forzar el refresco de la caché Parquet
├── schedule.py                      # Scheduler en vivo para ejecución automatizada
├── setup_timesfm.py                 # Script de instalación del vendor TimesFM 2.5
├── .env.example                     # Ejemplo de variables de entorno
└── README.md                        # Esta documentación
```

---

## 🚀 Inicio rápido

Siga estos pasos para configurar su entorno de desarrollo local.

### ✅ Requisitos previos

- Python 3.12+ (vía `uv`)
- [Ollama](https://ollama.com/) instalado y en ejecución localmente.
- Modelo LLM descargado: `ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`
- **Modelos del Weekend Council** (opcionales, pero requeridos para la diversidad de razonamiento del council): el council ejecuta cada persona en una familia de modelo *distinta* (Gemma / GLM / Qwen / LFM). Instalelos todos a la vez con `uv run python setup_council_models.py`.

### ⚙️ Instalación

1.  **Clonar el repositorio:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Instalar `uv` (si aún no lo ha hecho):**
    Vea [astral.sh/uv](https://astral.sh/uv) para las instrucciones de instalación.

3.  **Crear y activar el entorno virtual (PASO CRUCIAL):**
    Debe crear y activar el `.venv` antes de instalar los modelos fundacionales.
    ```bash
    uv venv
    source .venv/bin/activate  # En Windows, use `.\.venv\Scripts\activate.ps1`
    ```

4.  **Instalar los modelos fundacionales:**
    Ejecute los scripts de instalación para clonar los modelos en `vendor/` y aplicar parches:
    ```bash
    python setup_timesfm.py
    ```

5.  **Inicializar y sincronizar el entorno:**
    ```bash
    uv sync
    ```

6.  **Instalar navegadores para la investigación Web (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

7.  **Configurar sus claves API:**
    Cree un archivo `.env` en la raíz del proyecto:
    ```
    ALPHA_VANTAGE_API_KEY="SU_CLAVE"
    EIA_API_KEY="SU_CLAVE"

    # Opcional pero muy recomendado: Integración Gemini AI
    GEMINI_API_KEY_PAY="SU_CLAVE_NIVEL_PAGO"  # Para razonamiento/visión complejos (Gemini 2.5 Pro)
    GEMINI_API_KEY="SU_CLAVE_NIVEL_GRATUITO"   # Para tareas más ligeras (síntesis)
    GEMINI_PAY_MONTHLY_BUDGET_EUR=8.6        # Presupuesto de coste de 30 días móvil (€) — guardia de facturación principal
    GEMINI_PAY_DAILY_CAP=200                 # Backstop: máx. de llamadas API de pago por día
    ```

---

## 🛠️ Uso

El sistema entrena sus modelos con los datos más recientes en cada ejecución antes de dar una decisión.

### Modo Simulación (Paper Trading)

Para probar el sistema sin riesgo con un capital ficticio de 1000 €, use el flag `--simul`. El sistema gestionará un historial estricto de compras y ventas.

```sh
# Ejecutar un análisis simulado (Por defecto: SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Ejecutar en Petróleo (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Ejecución Real (Trading 212)

El sistema ahora está **totalmente integrado** con Trading 212:
- **Verificación de cartera**: antes de cualquier acción, el robot consulta su cash y posiciones reales.
- **Gestión de la API**: incluye mecanismos de reintento automático contra los límites de solicitudes (Rate Limiting).

```sh
# Ejecutar análisis con ejecución real (Demo o Real según .env)
uv run main.py --t212
```

---

## 🧪 Backtesting en Producción

El sistema incluye un **motor de backtest de producción autónomo** (`backtest_prod.py`) que reproduce las señales reales de prod de `logs_prod/trading_journal.csv` contra los precios reales de los archivos Parquet de `data_cache/`.

### Características
- **Señales reales**: reproduce las decisiones exactas del motor híbrido de 12 modelos.
- **Precios reales**: usa datos OHLCV reales de los ETF (SXRV.DE, CRUDP.PA) — sin proxies US.
- **Comisiones T212**: simula el modelo de comisiones de Trading 212 del 0,1 % por operación.
- **Comparación baseline**: calcula automáticamente el rendimiento buy-and-hold como benchmark.
- **Métricas**: Sharpe Ratio, Drawdown Máximo, Win Rate, Alpha, Retorno Total por ticker.

### Uso

```bash
uv run python backtest_prod.py
```

Resultados guardados en `logs_prod/backtest_report.json` con curvas de equity en CSV.

---

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! No dude en hacer un fork del proyecto y abrir una Pull Request.

---

## 📜 Licencia

Distribuido bajo la Licencia MIT.

---

## 📧 Contacto

Enlace del proyecto: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

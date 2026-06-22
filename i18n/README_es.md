<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_zh.md">中文</a> |
  <a href="README_hi.md">हिंदी</a> |
  <a href="README_es.md">Español</a> |
  <a href="README_fr.md">Français</a> |
  <a href="README_ar.md">العربية</a> |
  <a href="README_bn.md">বাংলা</a> |
  <a href="README_ru.md">Русский</a> |
  <a href="README_pt.md">Português</a> |
  <a href="README_id.md">Bahasa Indonesia</a>
</p>

<p align="center">
  <img src="../assets/banner.png" alt="Hybrid AI Trading Banner" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Sistema de Trading con IA Híbrida 📈</h1>
  <p>
    Un sistema experto de apoyo a la toma de decisiones para el trading de ETFs de NASDAQ y Petróleo (WTI), que aprovecha una inteligencia artificial híbrida trimodal para obtener señales de trading robustas y matizadas.
  </p>
</div>

<div align="center">

[![Project Status](https://img.shields.io/badge/status-in--development-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Tabla de Contenidos

- [🌟 Acerca del Proyecto](#-acerca-del-proyecto)
  - [✨ Características Principales](#-características-principales)
  - [💻 Stack Tecnológico](#-stack-tecnológico)
  - [⚙️ Rendimiento y Hardware](#️-rendimiento-y-hardware)
- [📂 Estructura del Proyecto](#-estructura-del-proyecto)
- [🚀 Inicio Rápido](#-inicio-rápido)
  - [✅ Requisitos Previos](#-requisitos-previos)
  - [⚙️ Instalación](#️-instalación)
- [🛠️ Uso](#️-uso)
  - [Modo de Simulación (Paper Trading)](#modo-de-simulación-paper-trading)
  - [Ejecución Real (Trading 212)](#ejecución-real-trading-212)
- [🧪 Backtesting en Producción](#-backtesting-en-producción)
- [🤝 Contribución](#-contribución)
- [📜 Licencia](#-licencia)
- [📧 Contacto](#-contacto)

---

## 🌟 Acerca del Proyecto

Este proyecto es un sistema experto de apoyo a la toma de decisiones para el trading de ETFs, utilizando un enfoque de IA híbrida trimodal. Está diseñado para proporcionar un análisis exhaustivo y robusto combinando varias perspectivas de IA.

### 🚀 Estrategia de Doble Ticker (Análisis vs. Trading)
El sistema utiliza un enfoque innovador para maximizar la precisión del modelo:
- **Análisis de Alta Fidelidad**: Los modelos de IA analizan **índices de referencia globales** (`^NDX` para Nasdaq, `CL=F` para el crudo WTI). Estos índices ofrecen un historial más largo y tendencias "más puras", sin el ruido relacionado con los horarios de negociación o las comisiones de los ETFs.
- **Ejecución en ETFs**: Las órdenes reales se colocan en los tickers correspondientes en **Trading 212** (`SXRV.DE`, `CRUDP.PA`), utilizando **precios en vivo de T212** (a través de la API de posiciones) para el tamaño de la posición. El estado de la cartera se sincroniza directamente desde T212 (`sync_state_from_t212()`), y los precios en vivo se inyectan en el flujo de análisis (`_inject_t212_live_price()` en `src/data.py`).

### 🧠 Motor de IA Híbrida
El sistema fusiona once señales distintas:
1.  **Modelo Cuantitativo Clásico**: Conjunto de RandomForest/GradientBoosting/LogisticRegression entrenado con indicadores técnicos y macroeconómicos.
2.  **TimesFM 2.5 (Google Research)**: Modelo fundacional de última generación para la predicción de series temporales.
3.  **TensorTrade / PPO (Aprendizaje por Refuerzo)**: Agente de RL (stable-baselines3) que entrena una política PPO en un entorno de trading personalizado de Gymnasium con persistencia a través de ciclos.
4.  **Modelo Oil-Bench (Gemma 4 12B (Unsloth))**: Modelo especializado en energía que fusiona datos fundamentales de la **EIA** (Existencias, Importaciones, Utilización de refinerías) y el sentimiento para el trading de WTI.
5.  **LLM Textual (Gemma 4 12B (Unsloth))**: Análisis contextual de datos brutos, noticias en tiempo real a través de la habilidad **AlphaEar**, e integración de **búsqueda web macroeconómica** dinámica. Consume explícitamente el informe nocturno del **Morning Brief** para adquirir una conciencia fundamental profunda antes de tomar decisiones.
6.  **LLM Visual (Gemma 4 12B (Unsloth))**: Análisis directo de gráficos técnicos (`enhanced_trading_chart.png`).
7.  **Análisis de Sentimiento**: Análisis híbrido combinando Alpha Vantage y tendencias "candentes" de **AlphaEar** (Weibo, WallstreetCN).
8.  **Datos Descentralizados (Hyperliquid)**: Análisis del sentimiento especulativo sobre el Petróleo (WTI) a través de la *Tasa de Fondeo* (Funding Rate) y el *Interés Abierto* (Open Interest).
9.  **Modelo Vincent Ganne**: Análisis geopolítico y multi-activos (WTI, Brent, Gas, DXY, MA200) para detectar suelos macroeconómicos.
10. **Modelo Grebenkov**: Modelo matemático de seguimiento de tendencias calibrado para el análisis de múltiples activos utilizando Paridad de Riesgo Agnóstica.
11. **Motor de Fusión Híbrido**: El meta-modelo que orquesta la ponderación dinámica y el consenso cognitivo entre todos los submodelos.

El objetivo es producir una decisión final (`BUY`, `SELL`, `HOLD`) con una prioridad absoluta en **Precisión Primero**.

### 🧘 Filosofía de Decisión: "Prudencia Cognitiva"
A diferencia de los algoritmos de trading clásicos que entran en pánico tan pronto como explota la volatilidad, este sistema aplica un enfoque de inversor informado:
- **Fuerte Consenso Requerido**: Un modelo cuantitativo (Clásico) puede dar una falsa alarma (`SELL`), pero si los modelos cognitivos (LLM Textual, Visual, TimesFM) se mantienen neutrales, el sistema preferirá `HOLD`.
- **Filtro de Confianza**: Una decisión de movimiento (Comprar o Vender) solo se valida si la confianza global supera un umbral de seguridad (generalmente 40%). Por debajo de este, el sistema considera la señal como "ruido" y permanece a la espera.
- **Protección del Capital**: En modo de riesgo `VERY_HIGH`, `HOLD` sirve como un escudo. Evita entrar en un mercado inestable y salir prematuramente en una simple corrección técnica si los fundamentos (Noticias/Visión/Hyperliquid) no confirman una caída inminente.

### ✨ Características Principales

- **Enfoque de Doble Ticker**: Analiza el índice, opera el ETF.
- **Precios en Vivo de T212**: Recuperación en tiempo real de precios en EUR a través de la API de Trading 212 (0.2s), con yfinance como respaldo y caché parquet.
- **Diferencial de Brent Fechado (Dated Brent Spread)**: Monitoreo de la tensión del mercado físico a través del diferencial entre el Brent Spot (Fechado) y los Futuros del Brent.
- **Resiliencia de Red**: Cortacircuitos de yfinance con rastreadores separados (información vs. descarga), tiempo de espera de 10s en todas las llamadas de red.
- **Auto-Invalidación de Caché**: La caché parquet auto-detecta la obsolescencia (> 2 días) y fuerza una actualización. Use `refresh_cache.py` para limpiar manualmente la caché.
- **Paralelización de Llamadas LLM**: Llamadas de modelos independientes (`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`) se ejecutan en un `ThreadPoolExecutor` para superponer la inferencia de Ollama con la E/S. La ruta crítica típicamente es de 4–6 min en CPU frente a más de 10 min de forma secuencial.
- **Caché de Consultas de Búsqueda de 24h**: La consulta de búsqueda web generada por el LLM se almacena en la caché bajo `data_cache/search_queries/<ticker>_<date>_<price-sig>.json`. Clave generada por fecha + una firma de acción de precio (agrupación log2 de cierre + rango RSI), por lo que un cambio de régimen la invalida. Las consultas de respaldo **nunca** se almacenan en caché (un fallo temporal de Ollama no puede corromper la caché por 24h).
- **Límite de Tiempo de Ciclo Estricto**: Cada ciclo de ticker está envuelto en un presupuesto de 15 minutos (`CYCLE_TIMEOUT_SECONDS` en `main.py`). Al exceder el tiempo de espera, el hilo de trabajo recibe `shutdown(wait=False)` para que el siguiente ticker comience de inmediato; se aplica HOLD al ticker cuyo tiempo expiró. Los *futures* individuales tienen sus propios límites de tiempo por tarea (búsqueda 240s, visual 300s, texto 240s, modelos CPU 180s cada uno, noticias 90s, rastreo web 30s).
- **Seguridad de Hilo Huérfano**: En el tiempo de espera del ciclo, se establece un `threading.Event` por ticker para que el trabajador huérfano aborte antes de cualquier llamada a `execute_t212_trade`, previniendo operaciones con dinero real después de que se le haya mostrado al usuario el panel de "HOLD aplicado". Un `threading.Lock` por ticker serializa aún más la colocación de órdenes de T212, eliminando el riesgo de operaciones dobles bajo superposición del planificador o invocaciones duplicadas de `--ticker`.
- **Centinela de Fallo de LLM**: Cuando `_query_ollama` agota todos los reintentos, el diccionario de respaldo lleva una bandera `"failed": True` para que la lógica de consenso posterior pueda distinguir "el modelo eligió HOLD" de "el modelo se bloqueó" (actualmente propagado pero no filtrado — un seguimiento conocido).
- **Cognición Avanzada**: Uso de **Gemma 4 12B** con **defensa JSON de doble capa**:
  1. **Aplicación de esquemas del lado del servidor** (`format: SCHEMA_*` con `additionalProperties: false`) — la capa que soporta la carga; pasada a través del parámetro `format` de Ollama en cada sitio de llamada. Esquemas definidos en `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`).
  2. **Sufijo de indicador de sistema defensivo** (`"...never add a 'thought' key."`) — segunda línea redundante pero inofensiva, mantenida como precaución extrema contra cualquier regresión futura de la capa del esquema.

  El token de razonamiento `<|think|>` está **activo** en los cuatro prompts del sistema de producción (reactivado el 2026-06-06 en `main` tras validación en la rama `think-mode`). La capa de esquema es lo que realmente neutraliza el defecto histórico de residuos JSON `<|channel>thought` (causa raíz de mayo de 2026): `tests/check_llm_json.py` confirma que los casos de esquema estricto (`v3_schema`, `v6_schema`, `v7_schema_strict`) producen JSON limpio incluso con `<|think|>` activado, mientras que las variantes laxas `format:json` fallan. Consulta `docs/ADR-001-think-mode-dual-layer-defence.md` para el análisis completo y el procedimiento de reversión.
- **Agente Autónomo Morning Brief**: Un proceso nocturno basado en `smolagents` (`morning_brief/morning_brief.py`) programado para ejecutarse automáticamente a la 01:00 AM a través de `schedule.py`. Analiza de forma independiente los registros diarios de la API, descarga datos fundamentales de inventarios de la EIA y arbitra un debate *Bull vs Bear*. El informe en formato markdown generado (`morning_market_brief.md`) se inyecta automáticamente en el prompt del sistema del LLM Textual durante el ciclo de trading diario, otorgando a la IA principal una memoria contextual y una conciencia fundamental profundas sin ralentizar la ejecución en el mercado en vivo.
- **Sentimiento de Noticias y Blockchain**: Integración de **AlphaEar** y **Hyperliquid** para capturar el sentimiento social y especulativo.
- **Planificador Automatizado**: Script `schedule.py` para ejecución continua (8:30 AM - 6:00 PM) en un servidor.
- **Gestión de Riesgos Centralizada**: El `AdvancedRiskManager` centraliza la lógica Anti-Pérdida (Stop-Loss) y Trailing Stop. Los modelos individuales ya no gestionan estos riesgos, lo que garantiza una estrategia de protección de capital unificada y estricta a través de varios regímenes de mercado.
- **Contratos de Datos Estrictos**: Todos los modelos de IA están completamente estandarizados para devolver una clase de datos `ModelResult` fuertemente tipada (`signal`, `confidence`, `reasoning`), asegurando un 100% de uniformidad en todo el motor de consenso.
- **Salud del Código Auditada**: El proyecto mantiene un estándar de salud del código **Grado B** a través de auditorías automatizadas (0 código muerto, alto índice de mantenibilidad).
- **Backtesting de Producción**: Motor de backtest independiente (`backtest_prod.py`) que reproduce señales reales de producción frente a precios reales con comisiones de T212 — sin dependencias externas.
- **Control de Volcado de Depuración**: Establezca `TRADING_DEBUG_DUMP=0` para deshabilitar el volcado de fallos de LLM (`data_cache/llm_debug_fail.txt`) que tiene un límite de 5 MB.

### 💻 Stack Tecnológico

- **Lenguaje**: `Python 3.12+`
- **Cálculos y Datos**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **IA y LLM**: `requests`, `ollama`
- **Scraping y Búsqueda Web**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualización**: `matplotlib` (backend Agg para seguridad de hilos), `seaborn`, `mplfinance`
- **Utilidades**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Rendimiento y Hardware
El sistema está diseñado para ser **eficiente en hardware de consumo** sin requerir una GPU dedicada.
- **Solo CPU**: La inferencia de LLM (Gemma 4 12B Q6_K a través de Ollama) y TimesFM se ejecutan completamente en la CPU. El rendimiento es de ~3–4 tokens/s en una CPU moderna de 8 núcleos.
- **RAM Recomendada**: 16 GB como mínimo (se sugieren 32 GB para ejecutar Gemma 4 12B cómodamente junto con TimesFM y TensorTrade).
- **Concurrencia de Ollama**: Establezca `OLLAMA_NUM_PARALLEL=8` (ya incluido en el archivo `.env` recomendado) para que múltiples llamadas a LLM puedan compartir la carga del modelo. Con el presupuesto de contexto predeterminado de 4 GB, los *slots* paralelos obtienen ~512 tokens cada uno — Ollama serializará si los *prompts* exceden el ctx por slot, pero el `ThreadPoolExecutor` mantiene la superposición en el reloj de pared beneficiosa para pasos limitados por E/S (obtención de noticias, rastreo web, modelos CPU).
- **Tiempo de Ejecución**: ~6 a 9 minutos por ticker en CPU (frío), ~3 a 5 minutos por ticker con acierto en la caché de consulta de búsqueda. El valor predeterminado ejecuta dos tickers (CRUDP.PA + SXRV.DE), por lo que se estima ~15 min en total.
- **Límite de Tiempo de Ciclo**: Cada ciclo de ticker está limitado a 15 min (`CYCLE_TIMEOUT_SECONDS`). Si se excede, se aplica HOLD y el siguiente ticker comienza de inmediato.
- **Velocidad de la API**: Integración ultrarrápida con Trading 212 (<1s para la recuperación de precios en vivo).

---

## 📂 Estructura del Proyecto

El proyecto está organizado de manera modular para un mejor mantenimiento.

```
Trading-AI/
├── morning_brief/                   # Agente autónomo nocturno para un análisis fundamental profundo
│   ├── morning_brief.py             # Orquestador del agente y configuración de smolagents
│   └── output/                      # Informes diarios en markdown generados (morning_market_brief.md)
├── src/                             # Módulos principales
│   ├── adaptive_weight_manager.py   # Ponderación dinámica de modelos basada en el rendimiento
│   ├── advanced_risk_manager.py     # Gestión de riesgos y dimensionamiento basados en tendencias
│   ├── chart_generator.py           # Generación de gráficos técnicos para el LLM visual
│   ├── classic_model.py             # Conjunto de modelos cuantitativos Scikit-learn
│   ├── data.py                      # Obtención de datos, caché y preprocesamiento
│   ├── database.py                  # Gestión de la base de datos SQLite para métricas
│   ├── eia_client.py                # Cliente de la API de la Administración de Información de Energía (EIA)
│   ├── enhanced_decision_engine.py  # Motor de fusión híbrida que orquesta todos los modelos
│   ├── features.py                  # Ingeniería de características técnicas y macroeconómicas
│   ├── grebenkov_model.py           # Modelo matemático de seguimiento de tendencias (Paridad de Riesgo Agnóstica)
│   ├── llm_client.py                # Integración de Ollama para la inferencia local de LLM
│   ├── news_fetcher.py              # Rastreo y análisis de noticias financieras
│   ├── oil_bench_model.py           # Modelo de trading de WTI especializado en energía
│   ├── performance_monitor.py       # Seguimiento de la precisión de los modelos e historial
│   ├── sentiment_analysis.py        # Integración de sentimientos de Alpha Vantage y AlphaEar
│   ├── t212_executor.py             # Ejecución real en la API de Trading 212 y cartera
│   ├── tensortrade_model.py         # Señal de Aprendizaje por Refuerzo (PPO)
│   ├── timesfm_model.py             # Integración de predicción de series temporales TimesFM 2.5
│   └── web_researcher.py            # Scraping web macroeconómico con Crawl4AI
├── data_cache/                       # Todas las cachés (ignoradas en git)
│   ├── *.parquet                     # Datos OHLCV por ticker (yfinance)
│   ├── macro/                        # Series temporales macro (FRED, múltiples fuentes)
│   ├── search_queries/               # Caché de búsqueda web LLM de 24h (por ticker+fecha+firma de precio)
│   └── llm_debug_fail.txt            # Volcado de fallos de LLM con límite (5 MB) — deshabilitar con TRADING_DEBUG_DUMP=0
├── tests/                            # Scripts de prueba y validación
│   ├── test_full_cycle.py            # Prueba completa de extremo a extremo T212 comprar/esperar/vender
│   ├── test_enhanced_decision_engine.py # Pruebas para el motor de fusión híbrida
│   ├── check_llm_json.py             # Diagnóstico de esquema JSON LLM (prueba los 4 sitios de llamada a Ollama)
│   ├── check_live.py                 # Script de verificación de precios de mercado en vivo
│   └── ...                           # Otras pruebas unitarias y de integración
├── i18n/                            # Internacionalización (READMEs traducidos)
├── assets/                          # Recursos estáticos (imágenes, banners)
├── memory-bank/                     # Memoria y contexto del asistente de IA
├── backtest_prod.py                 # Motor de backtesting de producción independiente
├── main.py                          # Punto de entrada único (Análisis y Trading)
├── pyproject.toml                   # Dependencias y configuración del proyecto (uv)
├── refresh_cache.py                 # Utilidad CLI para forzar la actualización de la caché Parquet
├── schedule.py                      # Planificador en vivo para ejecución automatizada
├── setup_timesfm.py                 # Script de instalación para el modelo de proveedor TimesFM 2.5
├── .env.example                     # Ejemplo de variables de entorno
└── README.md                        # Esta documentación
```

---

## 🚀 Inicio Rápido

Siga estos pasos para configurar su entorno de desarrollo local.

### ✅ Requisitos Previos

- Python 3.12+ (a través de `uv`)
- [Ollama](https://ollama.com/) instalado y funcionando localmente.
- Modelo LLM descargado: `ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`

### ⚙️ Instalación

1.  **Clonar el repositorio:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Instalar `uv` (si no lo ha hecho):**
    Consulte [astral.sh/uv](https://astral.sh/uv) para obtener instrucciones de instalación.

3.  **Crear y activar el entorno virtual (Paso CRUCIAL):**
    Debe crear y activar el `.venv` antes de instalar los modelos fundacionales.
    ```bash
    uv venv
    source .venv/bin/activate  # En Windows, use `.\.venv\Scripts\activate.ps1`
    ```

4.  **Instalar Modelos Fundacionales:**
    Ejecute los scripts de instalación para clonar los modelos en `vendor/` y aplicar los parches:
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

7.  **Configurar sus claves de API:**
    Cree un archivo `.env` en la raíz del proyecto:
    ```
    ALPHA_VANTAGE_API_KEY="SU_CLAVE"
    EIA_API_KEY="SU_CLAVE"
    ```

---

## 🛠️ Uso

El sistema entrena sus modelos con los datos más recientes en cada ejecución antes de tomar una decisión.

### Modo de Simulación (Paper Trading)

Para probar el sistema sin riesgos con un capital ficticio de 1000 €, use la bandera `--simul`. El sistema gestionará un historial estricto de compras y ventas.

```sh
# Ejecutar un análisis simulado (Predeterminado: SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Ejecutar sobre el Petróleo (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Ejecución Real (Trading 212)

El sistema ahora está **totalmente integrado** con Trading 212:
- **Verificación de Cartera**: Antes de cualquier acción, el robot consulta su efectivo y posiciones reales.
- **Gestión de la API**: Incluye mecanismos de reintento automático frente a límites de solicitud (Rate Limiting).

```sh
# Ejecutar análisis con ejecución real (Demo o Real según el .env)
uv run main.py --t212
```

---

## 🧪 Backtesting en Producción

El sistema incluye un **motor de backtest de producción independiente** (`backtest_prod.py`) que reproduce señales de producción reales desde `logs_prod/trading_journal.csv` frente a precios reales desde los archivos Parquet en `data_cache/`.

### Características
- **Señales reales**: Reproduce las decisiones exactas del motor híbrido de 12 modelos.
- **Precios reales**: Utiliza datos reales de ETF OHLCV (SXRV.DE, CRUDP.PA) — sin usar proxies de EE. UU.
- **Comisiones de T212**: Simula el modelo de comisiones del 0.1% por operación de Trading 212.
- **Comparación de referencia**: Calcula automáticamente el rendimiento de "comprar y mantener" (*buy-and-hold*) como punto de referencia.
- **Métricas**: Ratio de Sharpe, Drawdown Máximo, Tasa de Aciertos (Win Rate), Alfa, Rendimiento Total por ticker.

### Uso

```bash
uv run python backtest_prod.py
```

Resultados guardados en `logs_prod/backtest_report.json` junto con archivos CSV de las curvas de capital.

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! No dude en hacer un fork del proyecto y abrir un Pull Request.

---

## 📜 Licencia

Distribuido bajo la Licencia MIT.

---

## 📧 Contacto

Enlace del Proyecto: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

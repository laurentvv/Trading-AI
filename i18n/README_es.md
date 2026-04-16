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
  <img src="../assets/banner.png" alt="Banner de Trading con IA Híbrida" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Sistema de Trading con IA Híbrida 📈</h1>
  <p>
    Un sistema experto de apoyo a la decisión para el trading de ETFs de NASDAQ y Petróleo (WTI), que aprovecha una inteligencia artificial híbrida trimodal para obtener señales de trading robustas y matizadas.
  </p>
</div>

<div align="center">

[![Estado del Proyecto](https://img.shields.io/badge/status-en--desarrollo-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Versión de Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Licencia](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="../enhanced_performance_dashboard.png" alt="Panel de Rendimiento" width="800"/>
</p>

---

## 📚 Tabla de Contenidos

- [🌟 Sobre el Proyecto](#-sobre-el-proyecto)
  - [✨ Características Clave](#-características-clave)
  - [💻 Stack Tecnológico](#-stack-tecnológico)
  - [⚙️ Rendimiento y Hardware](#️-rendimiento-y-hardware)
- [📂 Estructura del Proyecto](#-estructura-del-proyecto)
- [🚀 Inicio Rápido](#-inicio-rápido)
  - [✅ Requisitos Previos](#-requisitos-previos)
  - [⚙️ Instalación](#️-instalación)
- [🛠️ Uso](#️-uso)
  - [Análisis Manual](#-análisis-manual)
  - [Análisis Automatizado con el Programador Inteligente](#-análisis-automatizado-con-el-programador-inteligente)
- [🤝 Contribución](#-contribución)
- [📜 Licencia](#-licencia)
- [📧 Contacto](#-contacto)

---

## 🌟 Sobre el Proyecto

Este proyecto es un sistema experto de apoyo a la decisión para el trading de ETFs, que utiliza un enfoque de IA híbrida trimodal. Está diseñado para proporcionar un análisis exhaustivo y robusto combinando varias perspectivas de IA.

### 🚀 Estrategia de Ticker Dual (Análisis vs. Trading)
El sistema utiliza un enfoque innovador para maximizar la precisión del modelo:
- **Análisis de Alta Fidelidad**: Los modelos de IA analizan **índices de referencia globales** (`^NDX` para Nasdaq, `CL=F` para crudo WTI). Estos índices ofrecen un historial más largo y tendencias más "puras", sin el ruido relacionado con las horas de trading o las comisiones de los ETFs.
- **Ejecución de ETFs**: Las órdenes reales se colocan en los tickers correspondientes en **Trading 212** (`SXRV.DE`, `CRUDP.PA`), utilizando **precios en vivo de T212** (vía API de posiciones) para el dimensionamiento de las posiciones.

### 🧠 Motor de IA Híbrida
El sistema fusiona ocho señales distintas:
1.  **Modelo Cuantitativo Clásico**: Conjunto de RandomForest/GradientBoosting/LogisticRegression entrenado en indicadores técnicos y macroeconómicos.
2.  **TimesFM 2.5 (Google Research)**: Modelo de base de vanguardia para la previsión de series temporales.
3.  **Modelo Oil-Bench (Gemma 4:e4b)**: Modelo especializado en energía que fusiona datos fundamentales de la **EIA** (Existencias, Importaciones, Utilización de refinerías) y el sentimiento para el trading de WTI.
4.  **LLM Textual (Gemma 4:e4b)**: Análisis contextual de datos brutos, noticias en tiempo real a través de la herramienta **AlphaEar**, e integración de **investigación web macroeconómica** dinámica.
5.  **LLM Visual (Gemma 4:e4b)**: Análisis directo de gráficos técnicos (`enhanced_trading_chart.png`).
6.  **Análisis de Sentimiento**: Análisis híbrido que combina Alpha Vantage y tendencias actuales de **AlphaEar** (Weibo, WallstreetCN).
7.  **Datos Descentralizados (Hyperliquid)**: Análisis del sentimiento especulativo sobre el Petróleo (WTI) a través de la *Tasa de Financiación* e *Interés Abierto*.
8.  **Modelo de Vincent Ganne**: Análisis geopolítico y cross-asset (WTI, Brent, Gas, DXY, MA200) para detectar suelos macroeconómicos.

El objetivo es producir una decisión final (`BUY`, `SELL`, `HOLD`) con una prioridad absoluta en **la precisión primero**.

### 🧘 Filosofía de Decisión: "Prudencia Cognitiva"
A diferencia de los algoritmos de trading clásicos que entran en pánico en cuanto estalla la volatilidad, este sistema aplica un enfoque de inversor informado:
- **Se requiere un consenso fuerte**: Un modelo cuantitativo (Clásico) puede dar la alarma (`SELL`), pero si los modelos cognitivos (LLM de Texto, Visión, TimesFM) permanecen neutrales, el sistema preferirá `HOLD`.
- **Filtro de Confianza**: Una decisión de movimiento (Comprar o Vender) solo se valida si la confianza global supera un umbral de seguridad (generalmente 40%). Por debajo de este, el sistema considera la señal como "ruido" y permanece a la espera.
- **Protección del Capital**: En el modo de riesgo `VERY_HIGH`, `HOLD` sirve como escudo. Evita entrar en un mercado inestable y evita salir prematuramente ante una simple corrección técnica si los fundamentos (Noticias/Visión/Hyperliquid) no confirman un desplome inminente.

### ✨ Características Clave

- **Enfoque de Ticker Dual**: Analiza el índice, opera el ETF.
- **Precios en vivo de T212**: Recuperación en tiempo real de precios en EUR a través de la API de Trading 212 (0.2s), con respaldo de yfinance y caché parquet.
- **Spread de Brent Dated**: Monitoreo de la tensión del mercado físico a través del diferencial entre el Brent Spot (Dated) y los Futuros de Brent.
- **Resiliencia de Red**: Interruptor automático de yfinance con rastreadores separados (información vs. descarga), tiempo de espera de 10s en todas las llamadas de red.
- **Cognición Avanzada**: Uso de **Gemma 4** para una mejor síntesis técnica/fundamental.
- **Noticias y Sentimiento Blockchain**: Integración de **AlphaEar** y **Hyperliquid** para capturar el sentimiento social y especulativo.
- **Programador Automatizado**: Script `schedule.py` para ejecución continua (8:30 AM - 6:00 PM) en un servidor.
- **Gestión de Riesgos Avanzada**: Ajuste automático de señales basado en la volatilidad y el régimen de mercado.

### 💻 Stack Tecnológico

- **Lenguaje**: `Python 3.12+`
- **Cálculos y Datos**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Aprendizaje Automático**: `scikit-learn`, `shap`
- **IA y LLM**: `requests`, `ollama`
- **Web Scraping y Búsqueda**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualización**: `matplotlib`, `seaborn`, `mplfinance`
- **Utilidades**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Rendimiento y Hardware
El sistema está diseñado para ser **eficiente en hardware de consumo** sin requerir una GPU dedicada.
- **Solo CPU**: La inferencia de LLM (Gemma 4 vía Ollama) y TimesFM están optimizados para una ejecución rápida en CPU si hay suficiente RAM disponible.
- **RAM Recomendada**: Mínimo 16 GB (se sugieren 32 GB para ejecutar Gemma 4 cómodamente).
- **Tiempo de Ejecución**: ~2 a 5 minutos para un ciclo completo (incluyendo crawling web, entrenamiento de ML, predicciones de TimesFM y 3 análisis de LLM).
- **Velocidad de la API**: Integración ultra rápida con Trading 212 (<1s para la recuperación de precios en vivo).

---

## 📂 Estructura del Proyecto

El proyecto está organizado de forma modular para un mejor mantenimiento.

```
Trading-AI/
├── src/                     # Módulos principales
│   ├── eia_client.py               # Cliente de datos fundamentales de energía
│   ├── oil_bench_model.py          # Modelo especializado en energía
│   ├── enhanced_decision_engine.py # Motor de fusión y modelo de Vincent Ganne
│   ├── advanced_risk_manager.py    # Gestión de riesgos sensible a la tendencia
│   ├── adaptive_weight_manager.py  # Gestión dinámica de pesos de modelos
│   ├── t212_executor.py            # Ejecución real en Trading 212
│   ├── timesfm_model.py            # Integración de TimesFM 2.5
│   └── ...                         # Datos, Características, Cliente LLM
├── tests/                   # Scripts de prueba y validación
├── data_cache/              # Datos de mercado y macro (Parquet)
├── main.py                  # Punto de entrada único (Análisis y Trading)
├── schedule.py              # Programador en vivo (8:30 AM - 6:00 PM)
├── backtest_engine.py       # Motor de backtesting histórico
├── .env                     # Claves API (Alpha Vantage, T212, EIA)
└── README.md                # Esta documentación
```

---

## 🚀 Inicio Rápido

Siga estos pasos para configurar su entorno de desarrollo local.

### ✅ Requisitos Previos

- Python 3.12+ (vía `uv`)
- [Ollama](https://ollama.com/) instalado y funcionando localmente.
- Modelo LLM descargado: `ollama pull gemma4:e4b`

### ⚙️ Instalación

1.  **Clonar el repositorio:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Instalar `uv` (si aún no lo ha hecho):**
    Consulte [astral.sh/uv](https://astral.sh/uv) para las instrucciones de instalación.

3.  **Instalar y parchear TimesFM 2.5 (Paso CRUCIAL):**
    Ejecute el script de instalación para clonar el modelo en `vendor/` y aplicar los parches:
    ```bash
    python setup_timesfm.py
    ```

4.  **Inicializar y sincronizar el entorno:**
    ```bash
    uv sync
    ```

5.  **Instalar navegadores para la investigación web (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

6.  **Configurar sus claves API:**
    Cree un archivo `.env` en la raíz del proyecto:
    ```
    ALPHA_VANTAGE_API_KEY="SU_CLAVE"
    EIA_API_KEY="SU_CLAVE"
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

El sistema está ahora **totalmente integrado** con Trading 212:
- **Verificación de Cartera**: Antes de cualquier acción, el robot consulta su efectivo y posiciones reales.
- **Gestión de API**: Incluye mecanismos de reintento automático contra los límites de solicitudes (Rate Limiting).

```sh
# Ejecutar análisis con ejecución real (Demo o Real según el .env)
uv run main.py --t212
```

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Siéntase libre de hacer un fork del proyecto y abrir una Pull Request.

---

## 📜 Licencia

Distribuido bajo la Licencia MIT.

---

## 📧 Contacto

Enlace del proyecto: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

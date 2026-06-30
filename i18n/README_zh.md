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
  <img src="assets/banner.png" alt="混合 AI 交易系统横幅" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 混合 AI 交易系统 📈</h1>
  <p>
    一个面向纳斯达克（NASDAQ）和原油（WTI）ETF 交易的专家级决策支持系统，利用由 12 个模型组成的混合人工智能，产生稳健且细腻的交易信号。
  </p>
</div>

<div align="center">

[![项目状态](https://img.shields.io/badge/status-开发中-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python 版本](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![许可证](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 目录

- [🌟 关于本项目](#-关于本项目)
  - [🚀 双标的策略（分析 vs. 交易）](#-双标的策略分析-vs-交易)
  - [🧠 混合 AI 引擎](#-混合-ai-引擎)
  - [🧘 决策哲学：“认知谨慎”](#-决策哲学认知谨慎)
  - [✨ 核心特性](#-核心特性)
  - [💻 技术栈](#-技术栈)
  - [⚙️ 性能与硬件](#️-性能与硬件)
  - [🧠 AI 与 LLM 架构（Gemini + 本地兜底）](#-ai-与-llm-架构gemini--本地兜底)
  - [🧠 FinAcumen（金融记忆）](#-finacumen金融记忆)
- [📂 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
  - [✅ 前置要求](#-前置要求)
  - [⚙️ 安装](#️-安装)
- [🛠️ 使用](#️-使用)
  - [模拟模式（Paper Trading）](#模拟模式paper-trading)
  - [实盘执行（Trading 212）](#实盘执行trading-212)
- [🧪 生产环境回测](#-生产环境回测)
  - [特性](#特性)
  - [用法](#用法)
- [🤝 贡献](#-贡献)
- [📜 许可证](#-许可证)
- [📧 联系方式](#-联系方式)

---

## 🌟 关于本项目

一个面向纳斯达克（NASDAQ）和原油（WTI）ETF 交易的专家级决策支持系统，利用由 12 个模型组成的混合人工智能。

### 🚀 双标的策略（分析 vs. 交易）

系统**分析全球指数**（例如纳斯达克-100 的 `^NDX`、WTI 的 `CL=F`），但**在以欧元计价的 ETF 上执行**（例如 `SXRV.DE`、`CRUDP.PA`）。这种解耦确保了基于高保真数据的分析，以及在 Trading 212 可交易资产上的真实执行。

### 🧠 混合 AI 引擎

引擎将异构模型组合成一个**加权共识**：

1. **Scikit-Learn 模型**（RandomForest、GradientBoosting、LogisticRegression）——通过 `TimeSeriesSplit` 验证以防止数据泄漏。激进的量化信号（占认知权重的 25%）。
2. **TimesFM 2.5**（Google Research）——用于时间序列预测的基础模型。
3. **TensorTrade / PPO**（stable-baselines3）——在自定义 Gymnasium 环境中的强化学习智能体。
4. **Gemma 4 12B**（Ollama）——**文本**（宏观/新闻）与**视觉**（技术图表）分析；**双层 JSON 防御**在 `<|think|>` 思考模式激活时仍保证纯净的 JSON。
5. **混合情感分析**（Alpha Vantage + AlphaEar + Hyperliquid）。
6. **Vincent Ganne Model** —— 地缘政治锁（WTI、Brent、天然气、尿素、DXY），仅生成 BUY 信号以验证纳斯达克底部。
7. **OilBenchModel** —— 专为 WTI 设计的认知模型（技术指标 + EIA 基本面 + 情绪）。

### 🧘 决策哲学：“认知谨慎”

认知模型（Gemma 4、情感、Vincent Ganne）持有 **75%** 的决策权重，而激进的量化模型仅占 **25%**。这种刻意的加权确保定性上下文能够调和量化信号。只有当全局置信度超过 **40%** 时信号才会被执行；在 20%–40% 之间则降级为 HOLD。

### ✨ 核心特性

- **混合云/本地 LLM 架构**：集成 `free-llm-api-keys`，利用高智商“前沿模型”（DeepSeek、Claude、Gemini）进行文本分析，并提供 100% 稳健的本地 Ollama 兜底（视觉图表仍专属 Ollama）。
- **双标的策略**：分析指数、交易 ETF。
- **T212 实时价格**：通过 Trading 212 API 实时获取欧元价格（0.2 秒），并具备 yfinance 兜底与 parquet 缓存。
- **Dated Brent 价差**：通过 Brent Spot（Dated）与 Brent 期货之间的价差监控实物市场紧张度。
- **网络韧性**：yfinance 断路器，采用独立追踪器（info vs. download），所有网络调用 10 秒超时。
- **缓存自动失效**：Parquet 缓存检测过期（> 2 天）并强制刷新。使用 `refresh_cache.py` 手动清空。
- **LLM 调用并行化**：独立的模型调用（`text_llm`、`visual_llm`、`search_query`、`timesfm`、`tensortrade`、`grebenkov`）在 `ThreadPoolExecutor` 中运行，将 Ollama 推理与 I/O 重叠。在 CPU 上关键路径通常为 4–6 分钟，而串行需 10+ 分钟。
- **24 小时搜索查询缓存**：LLM 生成的网络搜索查询缓存于 `data_cache/search_queries/<ticker>_<date>_<price-sig>.json`。键由日期 + 价格行为签名（收盘价 log2 桶 + RSI 桶）组成，因此行情切换会使其失效。兜底查询**从不**缓存（一次瞬态 Ollama 故障不会污染缓存 24 小时）。
- **严格周期超时**：每个标的的周期被包裹在 15 分钟预算内（`main.py` 中的 `CYCLE_TIMEOUT_SECONDS`）。超时时，工作线程被 `shutdown(wait=F)` 以便下一个标的立即开始；对超时标的应用 HOLD。各 future 有各自的超时（搜索 240 秒、视觉 300 秒、文本 240 秒、CPU 模型各 180 秒、新闻 90 秒、网页爬取 30 秒）。
- **孤儿线程安全**：周期超时时，设置一个按标的的 `threading.Event`，使孤儿工作线程在任何 `execute_t212_trade` 调用之前中止——防止在用户已看到“已应用 HOLD”面板后进行真实资金交易。按标的的 `threading.Lock` 进一步串行化 T212 下单，消除调度重叠或重复 `--ticker` 调用下的双重交易风险。
- **LLM 失败哨兵**：当 `_query_ollama` 耗尽所有重试时，兜底字典带有 `"failed": True` 标记，使下游共识逻辑可区分“模型选择了 HOLD”与“模型崩溃”（目前已传播但未过滤——一个已知后续事项）。
- **高级认知**：使用 **Gemma 4 12B**，具备**双层 JSON 防御**：
  1. **服务端模式强制**（`format: SCHEMA_*`，`additionalProperties: false`）——承载层；通过 Ollama 的 `format` 参数在每个调用点传递。模式定义于 `src/llm_client.py`（`SCHEMA_TRADING_DECISION`、`SCHEMA_SEARCH_QUERY`、`SCHEMA_OIL_ALLOCATION`）。
  2. **防御性 system prompt 后缀**（`"...never add a 'thought' key."`）——冗余但无害的第二道防线，作为 belt-and-braces 防止模式层未来任何回归。

  `<|think|>` 推理令牌在所有四个生产 system prompt 中均**激活**（于 2026-06-06 在 `think-mode` 分支验证后重新启用到 `main`）。正是模式层真正中和了历史上 `<|channel>thought` JSON 碎片的缺陷（2026 年 5 月根因）：`tests/check_llm_json.py` 证实 schema-strict 用例（`v3_schema`、`v6_schema`、`v7_schema_strict`）即使在 `<|think|>` 激活时也产生纯净 JSON，而 loose 的 `format:json` 变体会失败。完整分析与回退程序见 `docs/ADR-001-think-mode-dual-layer-defence.md`。
- **自主 Morning Brief 智能体**：一个基于 `smolagents` 的隔夜工作流（`morning_brief/morning_brief.py`），通过 `schedule.py` 在凌晨 01:00 自动运行。它独立爬取每日 API 日志、下载 EIA 库存基本面数据，并仲裁一场 *Bull vs Bear* 辩论。生成的 markdown 报告（`morning_market_brief.md`）会在每日交易周期自动注入到文本 LLM 的 system prompt 中，赋予主 AI 深度上下文记忆与基本面认知，而不拖慢实盘执行。
- **🏛️ Weekend Council（战略记忆）**：每周一次的多人格 LLM 回顾（`src/council/weekend_council.py`），通过 `schedule.py` 在每个**周六 01:00** 运行。六个人格——每个运行在**不同的 Ollama 模型家族**上（Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / LFM 2.5 / Mistral Nemo 12B），以获得真正的推理多样性——按照 4 轮协议（Problem Restate Gate → 带明确 STANCE 的 Analysis → 1 对 1 辩论 → Judge 综合）进行审议，并具备反群体思维机制（异议配额、未决优先裁决）。Judge（Qwen3.5-9B-MTP）给出每个标的的立场，成为实时共识中的**第 11 个加权投票**（9.5%），置信度在 7 天内线性衰减。慷慨的 token 预算（`num_predict` 最高 12000、`num_ctx` 最高 65536）和 48 小时调度窗口适应 CPU 上的思考模型。Council 分析真实的 PROD 数据：模型准确率（`model_performance.db`）、投资组合指标与关键告警（`performance_monitor.db`），以及已执行的交易日志。使用 `uv run python setup_council_models.py` 安装所需的 6 个模型。参见 `docs/ADR-003-weekend-council-11th-voice.md`。
- **新闻与区块链情绪**：集成 **AlphaEar** 与 **Hyperliquid** 以捕获社交与投机情绪。
- **自动化调度器**：`schedule.py` 脚本，用于在服务器上持续执行（8:30 – 18:00）。
- **集中化风险管理**：`AdvancedRiskManager` 集中化 Anti-Loss（止损）与 Trailing Stop 逻辑。单个模型不再管理这些风险，确保跨市场环境统一严格的资本保护策略。
- **严格数据契约**：所有 AI 模型被完全标准化，返回强类型 `ModelResult` dataclass（`signal`、`confidence`、`reasoning`），确保共识引擎中 100% 的一致性。
- **已审计的代码健康度**：项目通过自动化审计维持 **Grade B** 代码健康标准（0 死代码，高可维护性指数）。
- **生产环境回测**：独立回测引擎（`backtest_prod.py`），用 T212 手续费回放真实生产信号并对照真实价格——无外部依赖。
- **调试转储控制**：设置 `TRADING_DEBUG_DUMP=0` 以禁用有上限（5 MB）的 `data_cache/llm_debug_fail.txt` LLM 失败转储。

### 💻 技术栈

- **语言**：`Python 3.12+`
- **计算与数据**：`pandas`、`numpy`、`yfinance`、`pyarrow`、`pandas_datareader`、`hyperliquid-python-sdk`
- **机器学习**：`scikit-learn`、`shap`
- **AI 与 LLM**：`google-genai`（Gemini）、`requests`、`ollama`
- **网页抓取与搜索**：`beautifulsoup4`、`duckduckgo_search`、`crawl4ai`
- **可视化**：`matplotlib`（Agg 后端以保证线程安全）、`seaborn`、`mplfinance`
- **工具**：`tqdm`、`rich`、`python-dotenv`、`schedule`

### ⚙️ 性能与硬件
本系统设计为**在消费级硬件上表现良好**，无需专用 GPU。
- **仅 CPU**：LLM 推理（通过 Ollama 的 Gemma 4 12B Q6_K）与 TimesFM 完全在 CPU 上运行。在现代 8 核 CPU 上吞吐量约为 ~3–4 tokens/s。
- **推荐内存**：最低 16 GB（建议 32 GB，以便同时舒适运行 Gemma 4 12B、TimesFM 和 TensorTrade）。
- **Ollama 并发**：设置 `OLLAMA_NUM_PARALLEL=8`（已在推荐的 `.env` 中），以便多个 LLM 调用共享模型负载。在默认 4 GB 上下文预算下，并行槽位每个获得 ~512 tokens——当提示超过每个槽位的 ctx 时 Ollama 会串行化，但 `ThreadPoolExecutor` 仍为 I/O 密集步骤（新闻抓取、网页爬取、CPU 模型）保持有益的墙上时间重叠。
- **执行时间**：CPU 上每个标的约 ~6 到 9 分钟（冷启动），搜索查询缓存命中时每个标的约 ~3 到 5 分钟。默认运行两个标的（CRUDP.PA + SXRV.DE），因此请预留总共 ~15 分钟。
- **周期超时**：每个标的的周期上限为 15 分钟（`CYCLE_TIMEOUT_SECONDS`）。若超过，则应用 HOLD 并立即开始下一个标的。
- **API 速度**：超快的 Trading 212 集成（实时价格获取 <1 秒）。

### 🧠 AI 与 LLM 架构（Gemini + 本地兜底）
系统采用高度稳健的多层架构，以确保最大正常运行时间与智能决策，并深度集成于 `main.py` 与 Weekend Council。

- **4 层级联兜底**：
  1. **Gemini 付费层（`GEMINI_API_KEY_PAY`）**：最高优先级。使用 Gemini 2.5 Pro 等高级模型进行复杂推理、技术图表视觉和最终交易决策。
  2. **Gemini 免费层（`GEMINI_API_KEY`）**：用于较轻量、高吞吐的任务，例如网页上下文摘要。
  3. **免费 LLM API 代理**：通过 `free-llm-api-keys` 兜底。
  4. **本地 Ollama**：当所有云服务都失败时，100% 稳健的离线 CPU 兜底。
- **成本保护**：付费层受滚动 30 天成本预算（`GEMINI_PAY_MONTHLY_BUDGET_EUR`，默认 8.6 €/月）约束——每次调用的成本根据实际 token 使用量 × 模型价格计算并累计；当预算用尽时，调用回落到免费层 / Ollama。每日上限兜底（`GEMINI_PAY_DAILY_CAP`，默认 200）防止失控循环。
- **集成**：主日常执行引擎（`main.py`）使用 Gemini 进行实时多模型共识，而异步的 Weekend Council（`council`）则专门针对某些角色（如 Judge 与 Sceptique）集成 Gemini，并搭配多样的本地 Ollama 模型。

### 🧠 FinAcumen（金融记忆）
FinAcumen 架构已被集成，以赋予本地 AI 模型**经验记忆**与确定性工具。这解决了 LLM 的失忆问题。
- FinAcumen **在夜间异步运行**（通过 `schedule.py`），以在不阻塞交易周期的情况下充分利用 CPU 算力。
- 其深度定性报告会自动追加到 **Morning Market Brief** 中，在整个交易日里指导决策 LLM。

## 📂 项目结构

项目以模块化方式组织，以获得更好的可维护性。

```
Trading-AI/
├── morning_brief/                   # 用于深度基本面分析的隔夜自主智能体
│   ├── morning_brief.py             # 智能体编排器与 smolagents 配置
│   └── output/                      # 生成的每日 markdown 报告（morning_market_brief.md）
├── src/                             # 核心模块
│   ├── adaptive_weight_manager.py   # 基于性能的动态模型加权
│   ├── advanced_risk_manager.py     # Trend-Aware 风险管理与仓位 sizing
│   ├── bootstrap.py                 # 核心初始化逻辑
│   ├── chart_generator.py           # 为视觉 LLM 生成技术图表
│   ├── classic_model.py             # Scikit-learn 量化模型集成
│   ├── config_weights.py            # 混合引擎的基础权重配置
│   ├── data.py                      # 数据获取、缓存与预处理
│   ├── database.py                  # 用于指标的 SQLite 数据库管理
│   ├── eia_client.py                # Energy Information Administration API 客户端
│   ├── enhanced_decision_engine.py  # 编排所有模型的混合融合引擎
│   ├── enhanced_trading_example.py  # 模型使用示例脚本
│   ├── features.py                  # 技术与宏观经济特征工程
│   ├── grebenkov_model.py           # Trend-Following 数学模型（Agnostic Risk Parity）
│   ├── hmm_model.py                 # 用于 regime 检测的 Hidden Markov Model
│   ├── llm_client.py                # 用于本地 LLM 推理的 Ollama 集成
│   ├── news_fetcher.py              # 金融新闻抓取与解析
│   ├── oil_bench_model.py           # 能源专精的 WTI 交易模型
│   ├── performance_monitor.py       # 追踪模型准确率与历史
│   ├── read_simul.py                # 读取模拟输出的工具
│   ├── sentiment_analysis.py        # Alpha Vantage 与 AlphaEar 情绪集成
│   ├── t212_executor.py             # Trading 212 API 真实执行与投资组合
│   ├── tensortrade_model.py         # 强化学习（PPO）信号
│   ├── timesfm_model.py             # TimesFM 2.5 时间序列预测集成
│   └── web_researcher.py            # 使用 Crawl4AI 进行宏观经济网页抓取
├── data_cache/                       # 所有缓存（已 gitignore）
│   ├── *.parquet                     # 每个标的的 OHLCV 数据（yfinance）
│   ├── macro/                        # 宏观时间序列（FRED，多源）
│   ├── search_queries/               # 24 小时 LLM 搜索查询缓存（按 标的+日期+price-sig）
│   └── llm_debug_fail.txt            # 有上限（5 MB）的 LLM 失败转储——用 TRADING_DEBUG_DUMP=0 禁用
├── tests/                            # 测试与验证脚本
│   ├── test_full_cycle.py            # 端到端 T212 买入/等待/卖出测试
│   ├── test_enhanced_decision_engine.py # 混合融合引擎测试
│   ├── check_llm_json.py             # LLM JSON-schema 诊断（测试全部 4 个 Ollama 调用点）
│   ├── check_live.py                 # 实时市场价格验证脚本
│   └── ...                           # 其他单元与集成测试
├── i18n/                            # 国际化（已翻译的 README）
├── assets/                          # 静态资源（图片、横幅）
├── memory-bank/                     # 确定性 4 文件状态 + long-form 上下文（见 AGENTS.md §1）
├── backtest_prod.py                 # 独立的生产环境回测引擎
├── main.py                          # 单一入口点（分析与交易）
├── pyproject.toml                   # 项目依赖与配置（uv）
├── refresh_cache.py                 # 强制刷新 Parquet 缓存的 CLI 工具
├── schedule.py                      # 用于自动执行的实时调度器
├── setup_timesfm.py                 # TimesFM 2.5 vendor 安装脚本
├── .env.example                     # 环境变量示例
└── README.md                        # 本文档
```

---

## 🚀 快速开始

按以下步骤配置本地开发环境。

### ✅ 前置要求

- Python 3.12+（通过 `uv`）
- 已安装并在本地运行的 [Ollama](https://ollama.com/)。
- 已下载的 LLM 模型：`ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`
- **Weekend Council 模型**（可选，但 council 的推理多样性需要）：council 让每个人格运行在*不同*的模型家族上（Gemma / GLM / Qwen / LFM）。使用 `uv run python setup_council_models.py` 一次性安装全部。

### ⚙️ 安装

1.  **克隆仓库：**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **安装 `uv`（如果尚未安装）：**
    安装说明见 [astral.sh/uv](https://astral.sh/uv)。

3.  **创建并激活虚拟环境（关键步骤）：**
    必须在安装基础模型之前创建并激活 `.venv`。
    ```bash
    uv venv
    source .venv/bin/activate  # 在 Windows 上使用 `.\.venv\Scripts\activate.ps1`
    ```

4.  **安装基础模型：**
    运行安装脚本将模型克隆到 `vendor/` 并应用补丁：
    ```bash
    python setup_timesfm.py
    ```

5.  **初始化并同步环境：**
    ```bash
    uv sync
    ```

6.  **为网页研究安装浏览器（Crawl4AI）：**
    ```bash
    uv run python -m playwright install chromium
    ```

7.  **配置 API 密钥：**
    在项目根目录创建 `.env` 文件：
    ```
    ALPHA_VANTAGE_API_KEY="你的密钥"
    EIA_API_KEY="你的密钥"

    # 可选但强烈推荐：Gemini AI 集成
    GEMINI_API_KEY_PAY="你的付费层密钥"  # 用于复杂推理/视觉（Gemini 2.5 Pro）
    GEMINI_API_KEY="你的免费层密钥"      # 用于较轻的任务（摘要）
    GEMINI_PAY_MONTHLY_BUDGET_EUR=8.6        # 滚动 30 天成本预算 (€) —— 承载计费守护
    GEMINI_PAY_DAILY_CAP=200                 # 兜底：每天最大付费 API 调用次数
    ```

---

## 🛠️ 使用

系统在每次执行的最新数据上训练其模型，然后再给出决策。

### 模拟模式（Paper Trading）

使用 `--simul` 标志，以 1000 欧元的虚拟资金无风险地测试系统。系统将管理严格的买卖历史。

```sh
# 运行一次模拟分析（默认：SXRV.DE - 纳斯达克 100 EUR）
uv run main.py --simul

# 在原油（WTI）上运行
uv run main.py --ticker CRUDP.PA --simul
```

### 实盘执行（Trading 212）

该系统现已**完全集成** Trading 212：
- **投资组合核对**：在任何动作之前，机器人会查询你的真实现金和持仓。
- **API 管理**：包含针对请求限制（Rate Limiting）的自动重试机制。

```sh
# 运行带真实执行的分析（Demo 或 Real 根据 .env）
uv run main.py --t212
```

---

## 🧪 生产环境回测

系统包含一个**独立的生产环境回测引擎**（`backtest_prod.py`），将 `logs_prod/trading_journal.csv` 中的真实生产信号对照 `data_cache/` Parquet 文件中的真实价格进行回放。

### 特性
- **真实信号**：回放 12 模型混合引擎的精确决策。
- **真实价格**：使用 ETF 的真实 OHLCV 数据（SXRV.DE、CRUDP.PA）——无 US 代理。
- **T212 手续费**：模拟 Trading 212 每笔交易 0.1% 的手续费模型。
- **基准比较**：自动计算 buy-and-hold 表现作为基准。
- **指标**：每个标的的 Sharpe Ratio、最大回撤、Win Rate、Alpha、总收益。

### 用法

```bash
uv run python backtest_prod.py
```

结果保存到 `logs_prod/backtest_report.json`，并附带 CSV 权益曲线。

---

## 🤝 贡献

欢迎贡献！欢迎 fork 本项目并提交 Pull Request。

---

## 📜 许可证

基于 MIT 许可证分发。

---

## 📧 联系方式

项目链接：[https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

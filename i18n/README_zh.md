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
  <h1>📈 混合人工智能交易系统 📈</h1>
  <p>
    一个针对纳斯达克和石油 (WTI) ETF 交易的专家决策支持系统，利用三模态混合人工智能提供稳健而细微的交易信号。
  </p>
</div>

<div align="center">

[![Project Status](https://img.shields.io/badge/status-in--development-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 目录

- [🌟 关于项目](#-关于项目)
  - [✨ 核心特性](#-核心特性)
  - [💻 技术栈](#-技术栈)
  - [⚙️ 性能与硬件](#️-性能与硬件)
- [📂 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
  - [✅ 前置条件](#-前置条件)
  - [⚙️ 安装步骤](#️-安装步骤)
- [🛠️ 使用指南](#️-使用指南)
  - [手动分析](#-手动分析)
  - [智能调度器自动分析](#-智能调度器自动分析)
- [🤝 参与贡献](#-参与贡献)
- [📜 许可证](#-许可证)
- [📧 联系方式](#-联系方式)

---

## 🌟 关于项目

本项目是一个用于 ETF 交易的专家决策支持系统，采用三模态混合 AI 方法。它旨在通过结合多种 AI 视角提供全面且稳健的分析。

### 🚀 双标的策略（分析 vs 交易）
系统使用创新方法最大化模型准确性：
- **高保真分析**：AI 模型分析**全球基准指数**（纳斯达克为 `^NDX`，WTI 原油为 `CL=F`）。这些指数提供了更长的历史数据和“更纯粹”的趋势，不受交易时间或 ETF 管理费用的噪音影响。
- **ETF 执行**：在**Trading 212**上对相应标的（`SXRV.DE`, `CRUDP.PA`）执行实际订单，使用 **T212 实时价格**（通过持仓 API）进行仓位管理。投资组合状态直接从 T212 同步（`sync_state_from_t212()`），实时价格被注入到分析管道中（`src/data.py` 中的 `_inject_t212_live_price()`）。

### 🧠 混合 AI 引擎
系统融合了十一种不同的信号：
1. **经典量化模型**：基于技术和宏观经济指标训练的随机森林 (RandomForest) / 梯度提升 (GradientBoosting) / 逻辑回归 (LogisticRegression) 组合。
2. **TimesFM 2.5 (Google Research)**：用于时间序列预测的最先进基础模型。
3. **TensorTrade / PPO (强化学习)**：在自定义 Gymnasium 交易环境中训练 PPO 策略的 RL 代理 (stable-baselines3)，具有跨周期的持久性。
4. **Oil-Bench 模型 (Gemma 4 12B (Unsloth))**：能源专用模型，融合 **EIA** 基本面数据（库存、进口、炼油厂利用率）和 WTI 交易情绪。
5. **文本 LLM (Gemma 4 12B (Unsloth))**：分析原始数据的上下文，通过 **AlphaEar** 技能获取实时新闻，并集成动态的**宏观经济网络研究**。它会显式读取隔夜的 **Morning Brief** 报告，以在做出决策前获得深入的宏观基本面认知。
6. **视觉 LLM (Gemma 4 12B (Unsloth))**：直接分析技术图表（`enhanced_trading_chart.png`）。
7. **情绪分析**：结合 Alpha Vantage 和来自 **AlphaEar**（微博、华尔街见闻）“热门”趋势的混合分析。
8. **去中心化数据 (Hyperliquid)**：通过*资金费率 (Funding Rate)*和*未平仓合约 (Open Interest)*分析石油 (WTI) 的投机情绪。
9. **Vincent Ganne 模型**：用于检测宏观经济底部的地缘政治和跨资产分析（WTI、布伦特、天然气、DXY、MA200）。
10. **Grebenkov 模型**：使用不可知风险平价 (Agnostic Risk Parity) 为跨资产分析校准的趋势跟踪数学模型。
11. **混合融合引擎**：协调所有子模型之间动态权重和认知共识的元模型。

目标是产出最终决策（`BUY`、`SELL`、`HOLD`），并以**准确性优先**为绝对原则。

### 🧘 决策哲学：“认知审慎”
不同于一遇到波动率飙升就恐慌的传统交易算法，本系统应用了明智的投资者方法：
- **需要强烈共识**：量化模型（经典）可能会发出虚假警报（`SELL`），但如果认知模型（文本 LLM、视觉、TimesFM）保持中立，系统将倾向于 `HOLD`。
- **信心过滤器**：只有当全局信心超过安全阈值（通常为 40%）时，系统才会验证移动决策（买入或卖出）。低于此值，系统将该信号视为“噪音”并保持待机状态。
- **资本保护**：在 `VERY_HIGH` 风险模式下，`HOLD` 可作为护盾。它防止进入不稳定的市场，并避免在简单的技术回调中过早退出，前提是基本面（新闻/视觉/Hyperliquid）未确认即将发生崩盘。

### ✨ 核心特性

- **双标的方法**：分析指数，交易 ETF。
- **T212 实时价格**：通过 Trading 212 API 实时恢复欧元价格（0.2 秒），带有 yfinance 回退机制和 parquet 缓存。
- **即期布伦特价差 (Dated Brent Spread)**：通过即期布伦特（Dated）与布伦特期货之间的价差监控实物市场的紧张局势。
- **网络韧性**：yfinance 断路器，具有独立的跟踪器（信息 vs 下载），所有网络调用的超时时间为 10 秒。
- **缓存自动失效**：Parquet 缓存自动检测过期数据（> 2 天）并强制刷新。使用 `refresh_cache.py` 可手动清除缓存。
- **LLM 调用并行化**：独立的模型调用（`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`）在 `ThreadPoolExecutor` 中运行，以使 Ollama 推理与 I/O 重叠。关键路径在 CPU 上通常需要 4-6 分钟，而串行执行需要 10 分钟以上。
- **24小时搜索查询缓存**：LLM 生成的网络搜索查询缓存在 `data_cache/search_queries/<ticker>_<date>_<price-sig>.json` 中。其键由日期和一个价格行为签名（收盘价的 log2 桶 + RSI 桶）组成，因此状态的改变会使其失效。回退查询**绝不**缓存（一次短暂的 Ollama 故障不会污染 24 小时的缓存）。
- **硬周期超时**：每个标的周期被限制在 15 分钟内（在 `main.py` 中的 `CYCLE_TIMEOUT_SECONDS`）。超时后，工作线程会触发 `shutdown(wait=False)`，以便下一个标的立即启动；超时的标的将应用 HOLD 策略。各个 Future 都有其单独的任务超时（搜索 240 秒，视觉 300 秒，文本 240 秒，每个 CPU 模型 180 秒，新闻 90 秒，网络爬取 30 秒）。
- **孤儿线程安全**：在周期超时时，会设置一个每标的的 `threading.Event`，以便孤儿工作线程在任何 `execute_t212_trade` 调用之前退出——防止在向用户展示“HOLD appliqué”面板后还进行真金白银的交易。每标的的 `threading.Lock` 进一步序列化 T212 订单放置，消除了在调度器重叠或重复使用 `--ticker` 调用时的双重交易风险。
- **LLM 故障哨兵**：当 `_query_ollama` 耗尽所有重试次数时，回退字典将携带一个 `"failed": True` 标志，以便下游的共识逻辑能够区分“模型选择了 HOLD”和“模型崩溃”（目前仅传播未过滤——这是一个已知的后续工作）。
- **高级认知**：使用带有**双层 JSON 防御**的 **Gemma 4 12B**：
  1. **服务端模式强制执行**（`format: SCHEMA_*` 和 `additionalProperties: false`）——承载层；在每个调用站点的 Ollama 的 `format` 参数中传递。模式在 `src/llm_client.py` 中定义（`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`）。
  2. **防御性系统提示后缀**（`"...never add a 'thought' key."`）——冗余但无害的第二道防线，保留以防未来模式层的任何退化。

  在所有四个生产环境的系统提示中，  `<|think|>` 推理 token 在所有四个生产系统提示词中**均处于激活状态**（在 `think-mode` 分支上验证后，于 2026-06-06 在 `main` 分支上重新启用）。Schema 层是真正中和历史遗留的 `<|channel>thought` JSON 碎片缺陷（2026 年 5 月的根本原因）的机制：`tests/check_llm_json.py` 证实了严格遵循 schema 的用例（`v3_schema`、`v6_schema`、`v7_schema_strict`）即使在启用 `<|think|>` 的情况下也能生成干净的 JSON，而宽松的 `format:json` 变体则会失败。如需完整的分析和回滚流程，请参阅 `docs/ADR-001-think-mode-dual-layer-defence.md`。
- **自主晨报代理 (Autonomous Morning Brief Agent)**：这是一个基于 `smolagents` 的隔夜工作流 (`morning_brief/morning_brief.py`)，由 `schedule.py` 在凌晨 1:00 自动触发。它能独立爬取每日API日志、下载EIA基本面库存数据，并推演*牛市与熊市*的辩论。生成的报告 (`morning_market_brief.md`) 将在白天交易周期内自动注入到文本LLM的系统提示词中，赋予主AI深度记忆与基本面认知，且不会拖慢实时市场执行速度。
- **新闻与区块链情绪**：集成 **AlphaEar** 和 **Hyperliquid**，以捕捉社交和投机情绪。
- **自动调度器**：`schedule.py` 脚本用于在服务器上持续执行（上午 8:30 - 下午 6:00）。
- **集中式风险管理**：`AdvancedRiskManager` 集中管理止损（Anti-Loss）和追踪止损逻辑。个体模型不再管理这些风险，确保了在不同市场机制下统一且严格的资本保护策略。
- **严格的数据契约**：所有 AI 模型完全标准化以返回强类型的 `ModelResult` 数据类（`signal`, `confidence`, `reasoning`），确保了共识引擎间 100% 的一致性。
- **代码健康审计**：项目通过自动审计保持 **B 级 (Grade B)** 代码健康标准（0 死代码，高可维护性指数）。
- **生产环境回测**：独立的回测引擎（`backtest_prod.py`），使用 T212 费用和真实价格回放实际生产信号——没有外部依赖。
- **调试转储控制**：将 `TRADING_DEBUG_DUMP=0` 设置为禁用带上限的（5 MB）`data_cache/llm_debug_fail.txt` LLM 故障转储。

### 💻 技术栈

- **语言**：`Python 3.12+`
- **计算与数据**：`pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **机器学习**：`scikit-learn`, `shap`
- **AI 与 LLM**：`requests`, `ollama`
- **网页爬取与搜索**：`beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **可视化**：`matplotlib` (Agg 后端用于线程安全), `seaborn`, `mplfinance`
- **工具与实用程序**：`tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ 性能与硬件
系统设计为在**消费者硬件上表现出色**，无需专用 GPU。
- **仅 CPU**：LLM 推理（通过 Ollama 运行的 Gemma 4 12B Q6_K）和 TimesFM 完全在 CPU 上运行。在现代 8 核 CPU 上，吞吐量约为 3–4 tokens/s。
- **推荐内存**：最低 16 GB（建议 32 GB，以便舒适地同时运行 Gemma 4 12B、TimesFM 和 TensorTrade）。
- **Ollama 并发**：设置 `OLLAMA_NUM_PARALLEL=8`（已包含在推荐的 `.env` 中），使得多个 LLM 调用能够分担模型负载。使用默认的 4 GB 上下文预算，并行槽位每个将获得 ~512 个 tokens——如果提示符超出了单槽位上下文，Ollama 将会进行序列化，但 `ThreadPoolExecutor` 仍能使挂钟重叠在 I/O 密集型步骤（新闻获取、网络爬取、CPU 模型）中保持收益。
- **执行时间**：在 CPU 上每个标的（冷启动）约需 6 到 9 分钟，如果命中搜索查询缓存，每个标的约需 3 到 5 分钟。默认运行两个标的（CRUDP.PA + SXRV.DE），因此总计需要大约 15 分钟。
- **周期超时**：每个标的周期被限制在 15 分钟内（`CYCLE_TIMEOUT_SECONDS`）。如果超出限制，将应用 HOLD，并立即开始下一个标的。
- **API 速度**：超快的 Trading 212 集成（<1 秒获取实时价格）。

---

## 📂 项目结构

项目采用模块化组织，以提升可维护性。

```
Trading-AI/
├── morning_brief/                   # 隔夜自动代理，用于深度基本面分析
│   ├── morning_brief.py             # 代理编排与 smolagents 配置
│   └── output/                      # 生成的每日 Markdown 报告 (morning_market_brief.md)
├── src/                             # 核心模块
│   ├── adaptive_weight_manager.py   # 基于表现的动态模型权重管理
│   ├── advanced_risk_manager.py     # 趋势感知的风险管理和仓位控制
│   ├── chart_generator.py           # 为视觉 LLM 生成技术图表
│   ├── classic_model.py             # Scikit-learn 量化模型组合
│   ├── data.py                      # 数据获取、缓存和预处理
│   ├── database.py                  # 用于指标管理的 SQLite 数据库
│   ├── eia_client.py                # 美国能源信息署 API 客户端
│   ├── enhanced_decision_engine.py  # 协调所有模型的混合融合引擎
│   ├── features.py                  # 技术和宏观经济特征工程
│   ├── grebenkov_model.py           # 趋势跟踪数学模型 (不可知风险平价)
│   ├── llm_client.py                # 整合 Ollama 进行本地 LLM 推理
│   ├── news_fetcher.py              # 财经新闻抓取和解析
│   ├── oil_bench_model.py           # 专门针对 WTI 交易的能源模型
│   ├── performance_monitor.py       # 跟踪模型准确性和历史记录
│   ├── sentiment_analysis.py        # Alpha Vantage & AlphaEar 情绪分析整合
│   ├── t212_executor.py             # Trading 212 API 真实执行和投资组合
│   ├── tensortrade_model.py         # 强化学习 (PPO) 信号
│   ├── timesfm_model.py             # 整合 TimesFM 2.5 时间序列预测
│   └── web_researcher.py            # 使用 Crawl4AI 进行宏观经济网络研究
├── data_cache/                       # 所有缓存（gitignored）
│   ├── *.parquet                     # 每个标的 OHLCV 数据 (yfinance)
│   ├── macro/                        # 宏观时间序列 (FRED，多数据源)
│   ├── search_queries/               # 24小时 LLM 搜索查询缓存 (按 ticker+date+price-sig 分类)
│   └── llm_debug_fail.txt            # 有上限的 (5 MB) LLM 故障转储 — 通过 TRADING_DEBUG_DUMP=0 禁用
├── tests/                            # 测试和验证脚本
│   ├── test_full_cycle.py            # 端到端 T212 买入/等待/卖出测试
│   ├── test_enhanced_decision_engine.py # 混合融合引擎的测试
│   ├── check_llm_json.py             # LLM JSON 模式诊断 (测试所有 4 个 Ollama 调用点)
│   ├── check_live.py                 # 实时市场价格验证脚本
│   └── ...                           # 其他单元测试和集成测试
├── i18n/                            # 国际化 (翻译的 README)
├── assets/                          # 静态资源 (图片、横幅)
├── memory-bank/                     # AI 助手记忆和上下文
├── backtest_prod.py                 # 独立的生产环境回测引擎
├── main.py                          # 单一入口点 (分析与交易)
├── pyproject.toml                   # 项目依赖和配置 (uv)
├── refresh_cache.py                 # CLI 实用程序，用于强制刷新 Parquet 缓存
├── schedule.py                      # 用于自动执行的实时调度器
├── setup_timesfm.py                 # TimesFM 2.5 供应商依赖安装脚本
├── .env.example                     # 环境变量示例
└── README.md                        # 本文档
```

---

## 🚀 快速开始

请按照以下步骤设置您的本地开发环境。

### ✅ 前置条件

- Python 3.12+ (通过 `uv`)
- 已在本地安装并运行 [Ollama](https://ollama.com/)。
- 已下载 LLM 模型: `ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`

### ⚙️ 安装步骤

1.  **克隆仓库:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **安装 `uv` (如果尚未安装):**
    请参阅 [astral.sh/uv](https://astral.sh/uv) 获取安装说明。

3.  **创建并激活虚拟环境 (关键步骤):**
    必须在安装基础模型前创建并激活 `.venv`。
    ```bash
    uv venv
    source .venv/bin/activate  # Windows 下使用 `.\.venv\Scripts\activate.ps1`
    ```

4.  **安装基础模型:**
    运行安装脚本将模型克隆到 `vendor/` 并应用补丁:
    ```bash
    python setup_timesfm.py
    ```

5.  **初始化并同步环境:**
    ```bash
    uv sync
    ```

6.  **安装用于网络研究 (Crawl4AI) 的浏览器:**
    ```bash
    uv run python -m playwright install chromium
    ```

7.  **配置您的 API 密钥:**
    在项目根目录创建一个 `.env` 文件:
    ```
    ALPHA_VANTAGE_API_KEY="YOUR_KEY"
    EIA_API_KEY="YOUR_KEY"
    ```

---

## 🛠️ 使用指南

系统在每次执行前会对最新数据进行训练，然后给出决策。

### 模拟模式 (模拟交易)

如需使用 1000 欧元虚拟资金在无风险下测试系统，请使用 `--simul` 标志。系统将管理严格的买卖历史记录。

```sh
# 运行模拟分析 (默认: SXRV.DE - 纳斯达克 100 欧元)
uv run main.py --simul

# 在石油 (WTI) 上运行
uv run main.py --ticker CRUDP.PA --simul
```

### 真实执行 (Trading 212)

系统现在已与 Trading 212 **完全集成**：
- **投资组合验证**：在采取任何行动之前，机器人会咨询您的真实资金和持仓。
- **API 管理**：包括针对请求限制（速率限制）的自动重试机制。

```sh
# 运行带有真实执行的分析（根据 .env 使用演示或真实账户）
uv run main.py --t212
```

---

## 🧪 生产环境回测

系统包含一个**独立的生产环境回测引擎**（`backtest_prod.py`），它可以针对 `data_cache/` 中 Parquet 文件的真实价格重放 `logs_prod/trading_journal.csv` 中的实际生产信号。

### 特性
- **实际信号**：回放 12 模型混合引擎的确切决策。
- **真实价格**：使用实际的 ETF OHLCV 数据（SXRV.DE, CRUDP.PA）——没有使用美国代理数据。
- **T212 费用**：模拟 Trading 212 的每笔交易 0.1% 的费用模型。
- **基准比较**：自动计算买入并持有（buy-and-hold）的表现在作为基准。
- **指标**：夏普比率 (Sharpe Ratio)、最大回撤 (Maximum Drawdown)、胜率 (Win Rate)、Alpha、每个标的的总回报 (Total Return)。

### 使用方法

```bash
uv run python backtest_prod.py
```

结果及权益曲线 CSV 文件将保存至 `logs_prod/backtest_report.json`。

---

## 🤝 参与贡献

欢迎贡献力量！随时欢迎对该项目进行 Fork 并提交 Pull Request。

---

## 📜 许可证

根据 MIT 许可证分发。

---

## 📧 联系方式

项目链接: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

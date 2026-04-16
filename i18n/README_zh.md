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
  <img src="../assets/banner.png" alt="混合 AI 交易橫幅" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 混合 AI 交易系統 📈</h1>
  <p>
    一個用於 納斯達克 (NASDAQ) 和原油 (WTI) ETF 交易的專家決策支持系統，利用三模態混合人工智能提供穩健且細緻的交易信號。
  </p>
</div>

<div align="center">

[![項目狀態](https://img.shields.io/badge/status-开发中-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python 版本](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![許可證](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="../enhanced_performance_dashboard.png" alt="性能儀錶板" width="800"/>
</p>

---

## 📚 目錄

- [🌟 關於項目](#-關於項目)
  - [✨ 核心特性](#-核心特性)
  - [💻 技術棧](#-技術棧)
  - [⚙️ 性能與硬體](#️-性能與硬體)
- [📂 項目結構](#-項目結構)
- [🚀 快速開始](#-快速開始)
  - [✅ 前提條件](#-前提條件)
  - [⚙️ 安裝](#️-安裝)
- [🛠️ 使用方法](#️-使用方法)
  - [手動分析](#-手動分析)
  - [使用智慧排程器進行自動分析](#-使用智慧排程器進行自動分析)
- [🤝 貢獻](#-貢獻)
- [📜 許可證](#-許可證)
- [📧 聯繫方式](#-聯繫方式)

---

## 🌟 關於項目

該項目是一個用於 ETF 交易的專家決策支持系統，採用三模態混合 AI 方法。它旨在通過結合多個 AI 視角提供全面且穩健的分析。

### 🚀 雙代碼策略 (分析 vs. 交易)
該系統採用創新方法以最大化模型準確性：
- **高保真分析**：AI 模型分析 **全球參考指數**（納斯達克為 `^NDX`，WTI 原油為 `CL=F`）。這些指數提供更長的歷史記錄和更「純粹」的趨勢，沒有與交易時間或 ETF 費用相關的噪音。
- **ETF 執行**：真實訂單放置在 **Trading 212** 上相應的代碼（`SXRV.DE`、`CRUDP.PA`），使用 **T212 即時價格**（通過持倉 API）進行倉位管理。

### 🧠 混合 AI 引擎
系統融合了八個不同的信號：
1.  **經典量化模型**：在技術和宏觀經濟指標上訓練的 RandomForest/GradientBoosting/LogisticRegression 集成。
2.  **TimesFM 2.5 (Google Research)**：用於時間序列預測的最先進基礎模型。
3.  **Oil-Bench 模型 (Gemma 4:e4b)**：能源專業模型，融合了 **EIA** 基本面數據（庫存、進口、煉油廠利用率）和 WTI 交易情緒。
4.  **文本 LLM (Gemma 4:e4b)**：對原始數據進行背景分析，通過 **AlphaEar** 技能獲取即時新聞，並整合動態的 **宏觀經濟網絡研究**。
5.  **視覺 LLM (Gemma 4:e4b)**：直接分析技術圖表 (`enhanced_trading_chart.png`)。
6.  **情緒分析**：結合 Alpha Vantage 和來自 **AlphaEar** (微博、華爾街見聞) 的「熱點」趨勢的混合分析。
7.  **去中心化數據 (Hyperliquid)**：通過 *資金費率* 和 *未平倉合約* 分析原油 (WTI) 的投機情緒。
8.  **Vincent Ganne 模型**：地緣政治和跨資產分析（WTI、布倫特原油、天然氣、DXY、MA200），用於檢測宏觀經濟底部。

目標是以 **準確性優先** 為絕對原則，產生最終決定（`買入`、`賣出`、`持有`）。

### 🧘 決策哲學：「認知謹慎」
與一旦波動性爆炸就恐慌的經典交易算法不同，該系統採用知情投資者的方法：
- **需要強大共識**：量化模型（經典）可能會發出警告（`賣出`），但如果認知模型（文本 LLM、視覺、TimesFM）保持中立，系統將傾向於 `持有`。
- **置信度過濾器**：只有當全局置信度超過安全閾值（通常為 40%）時，移動決策（買入或賣出）才會被驗證。低於此閾值，系統將信號視為「噪音」並保持待命。
- **資本保護**：在 `極高` 風險模式下，`持有` 起到盾牌作用。它防止進入不穩定的市場，並避免在基本面（新聞/視覺/Hyperliquid）未確認即將崩盤的情況下，因簡單的技術調整而過早退出。

### ✨ 核心特性

- **雙代碼方法**：分析指數，交易 ETF。
- **T212 即時價格**：通過 Trading 212 API（0.2秒）實時獲取歐元價格，具備 yfinance 備用方案和 parquet 快取。
- **Dated Brent 价差**：通过布伦特现货 (Dated) 与布伦特期货之间的价差监控实物市场压力。
- **網絡韌性**：具備獨立追蹤器（信息 vs. 下載）的 yfinance 斷路器，所有網絡調用均有 10秒 超時設置。
- **高級認知**：使用 **Gemma 4** 進行更好的技術/基本面綜合。
- **新聞與區塊鏈情緒**：整合 **AlphaEar** 和 **Hyperliquid** 以捕捉社交和投機情緒。
- **自動排程器**：用於在伺服器上持續執行（上午 8:30 - 下午 6:00）的 `schedule.py` 腳本。
- **高級風險管理**：基於波動性和市場體系的自動信號調整。

### 💻 技術棧

- **語言**：`Python 3.12+`
- **計算與數據**：`pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **機器學習**：`scikit-learn`, `shap`
- **AI 與 LLM**：`requests`, `ollama`
- **網絡爬蟲與搜索**：`beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **可視化**：`matplotlib`, `seaborn`, `mplfinance`
- **工具**：`tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ 性能與硬體
該系統旨在 **在消費級硬體上表現良好**，無需專用 GPU。
- **僅限 CPU**：如果 RAM 充足，LLM 推理（通過 Ollama 的 Gemma 4）和 TimesFM 已針對快速 CPU 執行進行優化。
- **推薦 RAM**：至少 16 GB（建議 32 GB 以流暢運行 Gemma 4）。
- **執行時間**：一個完整週期約需 2 到 5 分鐘（包括網絡爬取、ML 訓練、TimesFM 預測和 3 個 LLM 分析）。
- **API 速度**：極速 Trading 212 集成（獲取即時價格 < 1秒）。

---

## 📂 項目結構

項目採用模組化組織，以便於維護。

```
Trading-AI/
├── src/                     # 核心模組
│   ├── eia_client.py               # 能源基本面數據客戶端
│   ├── oil_bench_model.py          # 能源專業模型
│   ├── enhanced_decision_engine.py # 融合引擎和 Vincent Ganne 模型
│   ├── advanced_risk_manager.py    # 趨勢感知風險管理
│   ├── adaptive_weight_manager.py  # 動態模型權重管理
│   ├── t212_executor.py            # 在 Trading 212 上的真實執行
│   ├── timesfm_model.py            # TimesFM 2.5 集成
│   └── ...                         # 數據、特徵、LLM 客戶端
├── tests/                   # 測試與驗證腳本
├── data_cache/              # 市場與宏觀數據 (Parquet)
├── main.py                  # 單一入口點 (分析與交易)
├── schedule.py              # 即時排程器 (上午 8:30 - 下午 6:00)
├── backtest_engine.py       # 歷史回測引擎
├── .env                     # API 密鑰 (Alpha Vantage, T212, EIA)
└── README.md                # 此文檔
```

---

## 🚀 快速開始

按照以下步驟設置您的本地開發環境。

### ✅ 前提條件

- Python 3.12+ (通過 `uv`)
- 已安裝並在本地運行的 [Ollama](https://ollama.com/)。
- 已下載的 LLM 模型：`ollama pull gemma4:e4b`

### ⚙️ 安裝

1.  **克隆倉庫：**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **安裝 `uv`（如果尚未安裝）：**
    安裝說明請參見 [astral.sh/uv](https://astral.sh/uv)。

3.  **安裝並修補 TimesFM 2.5（關鍵步驟）：**
    運行安裝腳本將模型克隆到 `vendor/` 並應用修補程序：
    ```bash
    python setup_timesfm.py
    ```

4.  **初始化並同步環境：**
    ```bash
    uv sync
    ```

5.  **為網絡研究安裝瀏覽器 (Crawl4AI)：**
    ```bash
    uv run python -m playwright install chromium
    ```

6.  **配置您的 API 密鑰：**
    在項目根目錄下創建一個 `.env` 文件：
    ```
    ALPHA_VANTAGE_API_KEY="您的密鑰"
    EIA_API_KEY="您的密鑰"
    ```

---

## 🛠️ 使用方法

系統在每次執行時都會在最新數據上訓練其模型，然後給出決策。

### 模擬模式 (模擬交易)

要在沒有風險的情況下使用 1000 歐元的虛擬資本測試系統，請使用 `--simul` 標誌。系統將管理嚴格的買賣歷史。

```sh
# 運行模擬分析（默認：SXRV.DE - 納斯達克 100 EUR）
uv run main.py --simul

# 在原油 (WTI) 上運行
uv run main.py --ticker CRUDP.PA --simul
```

### 真實執行 (Trading 212)

系統現在已與 Trading 212 **完全整合**：
- **投資組合驗證**：在採取任何行動之前，機器人會諮詢您的真實現金和持倉。
- **API 管理**：包含針對請求限制的自動重試機制（速率限制）。

```sh
# 進行真實執行的分析（根據 .env 文件決定是模擬環境還是真實環境）
uv run main.py --t212
```

---

## 🤝 貢獻

歡迎貢獻！請隨時 fork 項目並提交 Pull Request。

---

## 📜 許可證

根據 MIT 許可證分發。

---

## 📧 聯繫方式

項目連結：[https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

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
  <img src="../assets/banner.png" alt="Banner de Trading com IA Híbrida" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Sistema de Trading com IA Híbrida 📈</h1>
  <p>
    Um sistema especialista de apoio à decisão para o trading de ETFs de NASDAQ e Petróleo (WTI), aproveitando uma inteligência artificial híbrida trimodal para sinais de trading robustos e detalhados.
  </p>
</div>

<div align="center">

[![Status do Projeto](https://img.shields.io/badge/status-em--desenvolvimento-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Versão do Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Licença](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="../enhanced_performance_dashboard.png" alt="Painel de Desempenho" width="800"/>
</p>

---

## 📚 Tabela de Conteúdos

- [🌟 Sobre o Projeto](#-sobre-o-projeto)
  - [✨ Principais Características](#-principais-características)
  - [💻 Stack Tecnológica](#-stack-tecnológica)
  - [⚙️ Desempenho e Hardware](#️-desempenho-e-hardware)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [🚀 Início Rápido](#-início-rápido)
  - [✅ Pré-requisitos](#-pré-requisitos)
  - [⚙️ Instalação](#️-instalação)
- [🛠️ Uso](#️-uso)
  - [Análise Manual](#-análise-manual)
  - [Análise Automatizada com Agendador Inteligente](#-análise-automatizada-com-agendador-inteligente)
- [🤝 Contribuição](#-contribuição)
- [📜 Licença](#-licença)
- [📧 Contato](#-contato)

---

## 🌟 Sobre o Projeto

Este projeto é um sistema especialista de apoio à decisão para o trading de ETFs, utilizando uma abordagem de IA híbrida trimodal. Ele foi projetado para fornecer uma análise abrangente e robusta, combinando várias perspectivas de IA.

### 🚀 Estratégia de Ticker Duplo (Análise vs. Trading)
O sistema utiliza uma abordagem inovadora para maximizar a precisão do modelo:
- **Análise de Alta Fidelidade**: Os modelos de IA analisam **índices de referência globais** (`^NDX` para Nasdaq, `CL=F` para petróleo bruto WTI). Esses índices oferecem um histórico mais longo e tendências mais "puras", sem o ruído relacionado ao horário de trading ou taxas de ETFs.
- **Execução de ETFs**: Ordens reais são colocadas nos tickers correspondentes na **Trading 212** (`SXRV.DE`, `CRUDP.PA`), utilizando **preços ao vivo da T212** (via API de posições) para o dimensionamento das posições.

### 🧠 Motor de IA Híbrida
O sistema funde oito sinais distintos:
1.  **Modelo Quantitativo Clássico**: Conjunto de RandomForest/GradientBoosting/LogisticRegression treinado em indicadores técnicos e macroeconômicos.
2.  **TimesFM 2.5 (Google Research)**: Modelo de base de última geração para previsão de séries temporais.
3.  **Modelo Oil-Bench (Gemma 4:e4b)**: Modelo especializado em energia que funde dados fundamentais da **EIA** (Estoques, Importações, Utilização de refinarias) e sentimento para o trading de WTI.
4.  **LLM Textual (Gemma 4:e4b)**: Análise contextual de dados brutos, notícias em tempo real via ferramenta **AlphaEar** e integração de **pesquisa web macroeconômica** dinâmica.
5.  **LLM Visual (Gemma 4:e4b)**: Análise direta de gráficos técnicos (`enhanced_trading_chart.png`).
6.  **Análise de Sentimento**: Análise híbrida que combina Alpha Vantage e tendências atuais da **AlphaEar** (Weibo, WallstreetCN).
7.  **Dados Descentralizados (Hyperliquid)**: Análise do sentimento especulativo sobre o Petróleo (WTI) via *Taxa de Financiamento* e *Interesse Aberto*.
8.  **Modelo de Vincent Ganne**: Análise geopolítica e cross-asset (WTI, Brent, Gás, DXY, MA200) para detectar fundos macroeconômicos.

O objetivo é produzir uma decisão final (`BUY`, `SELL`, `HOLD`) com prioridade absoluta na **Precisão Primeiro**.

### 🧘 Filosofia de Decisão: "Prudência Cognitiva"
Ao contrário dos algoritmos de trading clássicos que entram em pânico assim que a volatilidade explode, este sistema aplica uma abordagem de investidor informado:
- **Consenso Forte Necessário**: Um modelo quantitativo (Clássico) pode alertar para venda (`SELL`), mas se os modelos cognitivos (LLM de Texto, Visão, TimesFM) permanecerem neutros, o sistema preferirá `HOLD`.
- **Filtro de Confiança**: Uma decisão de movimento (Compra ou Venda) só é validada se a confiança global exceder um limite de segurança (geralmente 40%). Abaixo disso, o sistema considera o sinal como "ruído" e permanece em espera.
- **Proteção de Capital**: No modo de risco `VERY_HIGH`, o `HOLD` serve como um escudo. Ele evita entrar em um mercado instável e evita sair prematuramente em uma simples correção técnica se os fundamentos (Notícias/Visão/Hyperliquid) não confirmarem um crash iminente.

### ✨ Principais Características

- **Abordagem de Ticker Duplo**: Analiza o índice, opera o ETF.
- **Preços ao Vivo T212**: Recuperação em tempo real de preços em EUR via API da Trading 212 (0.2s), com fallback do yfinance e cache parquet.
- **Spread do Brent Dated**: Monitorização da tensão no mercado físico através do diferencial entre o Brent Spot (Dated) e os Futuros de Brent.
- **Resiliência de Rede**: Interruptor automático do yfinance com rastreadores separados (informação vs. download), tempo limite de 10s em todas as chamadas de rede.
- **Cognição Avançada**: Uso do **Gemma 4** para melhor síntese técnica/fundamental.
- **Notícias e Sentimento Blockchain**: Integração do **AlphaEar** e **Hyperliquid** para capturar o sentimento social e especulativo.
- **Agendador Automatizado**: Script `schedule.py` para execução contínua (8:30 - 18:00) em um servidor.
- **Gestão de Riscos Avançada**: Ajuste automático de sinais com base na volatilidade e no regime de mercado.

### 💻 Stack Tecnológica

- **Linguagem**: `Python 3.12+`
- **Cálculos e Dados**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **IA e LLM**: `requests`, `ollama`
- **Web Scraping e Busca**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualização**: `matplotlib`, `seaborn`, `mplfinance`
- **Utilitários**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Desempenho e Hardware
O sistema foi projetado para ser **eficiente em hardware de consumo** sem a necessidade de uma GPU dedicada.
- **Apenas CPU**: A inferência de LLM (Gemma 4 via Ollama) e o TimesFM são otimizados para execução rápida em CPU se houver RAM suficiente disponível.
- **RAM Recomendada**: Mínimo de 16 GB (sugere-se 32 GB para rodar o Gemma 4 confortavelmente).
- **Tempo de Execução**: ~2 a 5 minutos para um ciclo completo (incluindo rastreamento web, treinamento de ML, previsões TimesFM e 3 análises de LLM).
- **Velocidade da API**: Integração ultra-rápida com a Trading 212 (<1s para recuperação de preço ao vivo).

---

## 📂 Estrutura do Projeto

O projeto é organizado de forma modular para melhor manutenção.

```
Trading-AI/
├── src/                     # Módulos principais
│   ├── eia_client.py               # Cliente de dados fundamentais de energia
│   ├── oil_bench_model.py          # Modelo especializado em energia
│   ├── enhanced_decision_engine.py # Motor de fusão e modelo de Vincent Ganne
│   ├── advanced_risk_manager.py    # Gestão de riscos sensível à tendência
│   ├── adaptive_weight_manager.py  # Gestão dinâmica de pesos dos modelos
│   ├── t212_executor.py            # Execução real na Trading 212
│   ├── timesfm_model.py            # Integração do TimesFM 2.5
│   └── ...                         # Dados, Características, Cliente LLM
├── tests/                   # Scripts de teste e validação
├── data_cache/              # Dados de mercado e macro (Parquet)
├── main.py                  # Ponto de entrada único (Análise e Trading)
├── schedule.py              # Agendador ao vivo (8:30 - 18:00)
├── backtest_engine.py       # Motor de backtesting histórico
├── .env                     # Chaves de API (Alpha Vantage, T212, EIA)
└── README.md                # Esta documentação
```

---

## 🚀 Início Rápido

Siga estas etapas para configurar seu ambiente de desenvolvimento local.

### ✅ Pré-requisitos

- Python 3.12+ (via `uv`)
- [Ollama](https://ollama.com/) instalado e rodando localmente.
- Modelo LLM baixado: `ollama pull gemma4:e4b`

### ⚙️ Instalação

1.  **Clonar o repositório:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Instalar o `uv` (se ainda não o fez):**
    Veja [astral.sh/uv](https://astral.sh/uv) para instruções de instalação.

3.  **Instalar e aplicar patch no TimesFM 2.5 (Passo CRUCIAL):**
    Execute o script de instalação para clonar o modelo em `vendor/` e aplicar os patches:
    ```bash
    python setup_timesfm.py
    ```

4.  **Inicializar e sincronizar o ambiente:**
    ```bash
    uv sync
    ```

5.  **Instalar navegadores para pesquisa web (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

6.  **Configurar suas chaves de API:**
    Crie um arquivo `.env` na raiz do projeto:
    ```
    ALPHA_VANTAGE_API_KEY="SUA_CHAVE"
    EIA_API_KEY="SUA_CHAVE"
    ```

---

## 🛠️ Uso

O sistema treina seus modelos com os dados mais recentes em cada execução antes de tomar uma decisão.

### Modo Simulação (Paper Trading)

Para testar o sistema sem risco com um capital fictício de 1000 €, use a flag `--simul`. O sistema gerenciará um histórico rigoroso de compras e vendas.

```sh
# Executar uma análise simulada (Padrão: SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Executar em Petróleo (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Execução Real (Trading 212)

O sistema agora está **totalmente integrado** com a Trading 212:
- **Verificação de Portfólio**: Antes de qualquer ação, o robô consulta seu caixa e posições reais.
- **Gestão de API**: Inclui mecanismos de repetição automática contra limites de solicitações (Rate Limiting).

```sh
# Executar análise com execução real (Demo ou Real de acordo com o .env)
uv run main.py --t212
```

---

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para fazer um fork do projeto e abrir um Pull Request.

---

## 📜 Licença

Distribuído sob a Licença MIT.

---

## 📧 Contato

Link do Projeto: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

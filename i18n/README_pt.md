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
  <img src="assets/banner.png" alt="Banner de Trading com IA Híbrida" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Sistema de Trading com IA Híbrida 📈</h1>
  <p>
    Um sistema especialista de apoio à decisão para o trading de ETFs de NASDAQ e Petróleo (WTI), aproveitando uma inteligência artificial híbrida de 12 modelos para sinais de trading robustos e detalhados.
  </p>
</div>

<div align="center">

[![Estado do Projeto](https://img.shields.io/badge/status-em--desenvolvimento-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Versão do Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Licença](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Tabela de Conteúdos

- [🌟 Sobre o Projeto](#-sobre-o-projeto)
  - [🚀 Estratégia de Ticker Duplo (Análise vs. Trading)](#-estratégia-de-ticker-duplo-análise-vs-trading)
  - [🧠 Motor de IA Híbrida](#-motor-de-ia-híbrida)
  - [🧘 Filosofia de Decisão: "Prudência Cognitiva"](#-filosofia-de-decisão-prudência-cognitiva)
  - [✨ Principais Características](#-principais-características)
  - [💻 Stack Tecnológica](#-stack-tecnológica)
  - [⚙️ Desempenho e Hardware](#️-desempenho-e-hardware)
  - [🧠 Arquitetura de IA e LLM (Gemini + Fallback Local)](#-arquitetura-de-ia-e-llm-gemini--fallback-local)
  - [🧠 FinAcumen (Memória Financeira)](#-finacumen-memória-financeira)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [🚀 Início Rápido](#-início-rápido)
  - [✅ Pré-requisitos](#-pré-requisitos)
  - [⚙️ Instalação](#️-instalação)
- [🛠️ Uso](#️-uso)
  - [Modo Simulação (Paper Trading)](#modo-simulação-paper-trading)
  - [Execução Real (Trading 212)](#execução-real-trading-212)
- [🧪 Backtesting de Produção](#-backtesting-de-produção)
  - [Características](#características)
  - [Utilização](#utilização)
- [🤝 Contribuição](#-contribuição)
- [📜 Licença](#-licença)
- [📧 Contato](#-contato)

---

## 🌟 Sobre o Projeto

Este projeto é um sistema especialista de apoio à decisão para o trading de ETFs, utilizando uma abordagem de IA híbrida de 12 modelos. Ele foi projetado para fornecer uma análise abrangente e robusta, combinando várias perspectivas de IA.

### 🚀 Estratégia de Ticker Duplo (Análise vs. Trading)
O sistema utiliza uma abordagem inovadora para maximizar a precisão do modelo:
- **Análise de Alta Fidelidade**: Os modelos de IA analisam **índices de referência globais** (`^NDX` para Nasdaq, `CL=F` para petróleo bruto WTI). Esses índices oferecem um histórico mais longo e tendências mais "puras", sem o ruído relacionado ao horário de trading ou taxas de ETFs.
- **Execução de ETFs**: Ordens reais são colocadas nos tickers correspondentes na **Trading 212** (`SXRV.DE`, `CRUDP.PA`), utilizando **preços ao vivo da T212** (via API de posições) para o dimensionamento das posições. O estado da carteira é sincronizado diretamente da T212 (`sync_state_from_t212()`), e os preços ao vivo são injetados no pipeline de análise (`_inject_t212_live_price()` em `src/data.py`).

### 🧠 Motor de IA Híbrida
O sistema funde treze sinais distintos (mais um meta-modelo):
1.  **Modelo Quantitativo Clássico**: Conjunto de RandomForest/GradientBoosting/LogisticRegression treinado em indicadores técnicos e macroeconômicos.
2.  **TimesFM 2.5 (Google Research)**: Modelo de base de última geração para previsão de séries temporais.
3.  **TensorTrade / PPO (Aprendizado por Reforço)**: Agente de RL (stable-baselines3) treinando uma política PPO num ambiente de trading Gymnasium personalizado, com persistência entre ciclos.
4.  **Modelo Oil-Bench (Gemma 4 12B (Unsloth))**: Modelo especializado em energia que funde dados fundamentais da **EIA** (Estoques, Importações, Utilização de refinarias) e sentimento para o trading de WTI.
5.  **LLM Textual (Nuvem/Local Híbrido)**: Análise contextual de dados brutos, notícias em tempo real através da skill **AlphaEar**, e integração de **pesquisa web macroeconômica** dinâmica. Alimentado principalmente por "Modelos de Fronteira" via `free-llm-api-keys`, com um fallback instantâneo para um **Gemma 4 12B** local via Ollama em caso de falha da API. Ele consome explicitamente o relatório noturno **Morning Brief** para adquirir uma profunda consciência fundamental.
6.  **LLM Visual (Gemma 4 12B (Unsloth))**: Análise direta de gráficos técnicos (`enhanced_trading_chart.png`).
7.  **Análise de Sentimento**: Análise híbrida que combina Alpha Vantage e tendências atuais da **AlphaEar** (Weibo, WallstreetCN).
8.  **Dados Descentralizados (Hyperliquid)**: Análise do sentimento especulativo sobre o Petróleo (WTI) via *Taxa de Financiamento* e *Interesse Aberto*.
9.  **Modelo de Vincent Ganne**: Análise geopolítica e cross-asset (WTI, Brent, Gás, DXY, MA200) para detetar fundos macroeconômicos.
10. **Modelo de Grebenkov**: Modelo matemático de Seguir-Tendência calibrado para análise cross-asset usando Agnostic Risk Parity.
11. **Modelo de Markov Oculto (HMM)**: Modelo probabilístico para deteção de regime de mercado (alta/baixa) baseado em variações históricas de preço.
12. **FinAcumen (Motor de Memória de Experiência)**: Um loop de agente ReAct inteligente que avalia as condições de mercado escrevendo e executando queries Python brutas contra conjuntos de dados simulados, equipado com uma "Memória Financeira" vetorial.
13. **🏛️ Conselho de Fim de Semana (Retrospetiva Estratégica)**: Uma deliberação LLM assíncrona, multipessoas e semanal, executada todos os **sábados à 01:00**. Seis personas (Estratega / Gestor de Risco / Quant / Cético / Tático / Comportamentalista) cada uma executada numa **família de modelo Ollama distinta** (Gemma 4 / GLM-4.6V / Qwen 3.5 / LFM 2.5 / Mistral Nemo) para uma genuína diversidade de raciocínio — não mudanças de figurino num único modelo. Um protocolo de 4 rondas (Portão de Reformulação do Problema → Análise com POSICIONAMENTO explícito → Debate 1-vs-1 → Síntese do Juiz) mais mecanismos anti-pensamento de grupo (quota de dissidência, veredito de não-resolvido-primeiro). O Juiz (Qwen3.5-9B-MTP) emite um posicionamento por ticker que se torna o **11.º voto ponderado** no consenso em tempo real (peso de 9,5%, decaído linearmente ao longo de 7 dias). Adaptado de [`0xNyk/council-of-high-intelligence`](https://github.com/0xNyk/council-of-high-intelligence). Ver `docs/ADR-003-weekend-council-11th-voice.md`.
14. **Motor de Fusão Híbrida**: O meta-modelo que orquestra a ponderação dinâmica e o consenso cognitivo entre todos os submodelos.

O objetivo é produzir uma decisão final (`BUY`, `SELL`, `HOLD`) com prioridade absoluta na **Precisão Primeiro**.

### 🧘 Filosofia de Decisão: "Prudência Cognitiva"
Ao contrário dos algoritmos de trading clássicos que entram em pânico assim que a volatilidade explode, este sistema aplica uma abordagem de investidor informado:
- **Consenso Forte Necessário**: Um modelo quantitativo (Clássico) pode alertar para venda (`SELL`), mas se os modelos cognitivos (LLM de Texto, Visão, TimesFM) permanecerem neutros, o sistema preferirá `HOLD`.
- **Filtro de Confiança**: Uma decisão de movimento (Compra ou Venda) só é validada se a confiança global exceder um limite de segurança (geralmente 40%). Abaixo disso, o sistema considera o sinal como "ruído" e permanece em espera.
- **Proteção de Capital**: No modo de risco `VERY_HIGH`, o `HOLD` serve como um escudo. Ele evita entrar num mercado instável e evita sair prematuramente numa simples correção técnica se os fundamentos (Notícias/Visão/Hyperliquid) não confirmarem um crash iminente.

### ✨ Principais Características

- **Arquitetura LLM Nuvem/Local Híbrida**: Integração com `free-llm-api-keys` para aproveitar "Modelos de Fronteira" altamente inteligentes (DeepSeek, Claude, Gemini) para análise textual, com um fallback 100% robusto para Ollama local (que permanece o motor exclusivo para gráficos visuais).
- **Abordagem de Ticker Duplo**: Analisa o índice, opera o ETF.
- **Preços ao Vivo T212**: Recuperação em tempo real de preços em EUR via API da Trading 212 (0,2s), com fallback do yfinance e cache parquet.
- **Spread do Brent Dated**: Monitorização da tensão no mercado físico através do diferencial entre o Brent Spot (Dated) e os Futuros de Brent.
- **Resiliência de Rede**: Interruptor automático do yfinance com rastreadores separados (informação vs. download), tempo limite de 10s em todas as chamadas de rede.
- **Auto-Invalidez do Cache**: O cache parquet deteta automaticamente a desatualização (> 2 dias) e força uma atualização. Use `refresh_cache.py` para limpeza manual do cache.
- **Paralelização de Chamadas LLM**: As chamadas independentes dos modelos (`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`) correm num `ThreadPoolExecutor` para sobrepor a inferência do Ollama com I/O. O caminho crítico é tipicamente de 4–6 min na CPU vs. 10+ min em sequência.
- **Cache de 24h da Query de Pesquisa**: A query de pesquisa web gerada pelo LLM é colocada em cache em `data_cache/search_queries/<ticker>_<date>_<price-sig>.json`. Indexada por data + uma assinatura de ação de preço (agrupamento log2 do fecho + intervalo de RSI), de modo que uma mudança de regime a invalida. As queries de fallback **nunca** são colocadas em cache (uma falha transitória do Ollama não pode envenenar o cache durante 24h).
- **Tempo Limite Rígido do Ciclo**: Cada ciclo de ticker é envolvido num orçamento de 40 min (`CYCLE_TIMEOUT_SECONDS` em `main.py`). Em caso de tempo limite, a thread de trabalho faz `shutdown(wait=False)` para que o próximo ticker comece imediatamente; HOLD é aplicado ao ticker com tempo limite excedido. Os futures individuais têm os seus próprios tempos limite por tarefa (pesquisa 240s, visual 300s, texto 240s, modelos de CPU 180s cada, notícias 90s, crawl web 30s).
- **Segurança de Thread Órfã**: Em caso de tempo limite do ciclo, um `threading.Event` por ticker é definido para que a thread de trabalho órfã aborte antes de qualquer chamada `execute_t212_trade` — impedindo operações com dinheiro real após o painel "HOLD appliqué" ter sido mostrado ao utilizador. Um `threading.Lock` por ticker serializa ainda mais a colocação de ordens na T212, eliminando o risco de operações duplas em sobreposição do agendador ou invocações duplicadas de `--ticker`.
- **Sentinela de Falha do LLM**: Quando o `_query_ollama` esgota todas as tentativas, o dict de fallback transporta uma flag `"failed": True` para que a lógica de consenso a jusante possa distinguir "o modelo escolheu HOLD" de "o modelo falhou" (atualmente propagado, mas não filtrado — um seguimento conhecido).
- **Cognição Avançada**: Uso do **Gemma 4 12B** com **defesa JSON de dupla camada**:
  1. **Aplicação de esquema do lado do servidor** (`format: SCHEMA_*` com `additionalProperties: false`) — a camada de suporte de carga; passada via parâmetro `format` do Ollama em cada ponto de chamada. Esquemas definidos em `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`).
  2. **Sufixo defensivo do prompt de sistema** (`"...never add a 'thought' key."`) — segunda linha redundante-mas-inofensiva, mantida como cinto-e-calaças contra qualquer regressão futura da camada de esquema.

  O token de raciocínio `<|think|>` está **ativo** em todos os quatro prompts de sistema de produção (reativado a 2026-06-06 em `main` após validação no ramo `think-mode`). A camada de esquema é o que realmente neutraliza o defeito histórico de detritos JSON `<|channel>thought` (causa raiz de maio de 2026): `tests/check_llm_json.py` confirma que os casos estritos de esquema (`v3_schema`, `v6_schema`, `v7_schema_strict`) produzem JSON limpo mesmo com `<|think|>` ativado, enquanto as variantes soltas `format:json` falham. Ver `docs/ADR-001-think-mode-dual-layer-defence.md` para a análise completa e o procedimento de reversão.
- **Agente Autónomo Morning Brief**: Um processo noturno baseado em `smolagents` (`morning_brief/morning_brief.py`) agendado para rodar automaticamente à 01:00 AM via `schedule.py`. Ele rastreia de forma independente os logs diários da API, descarrega dados fundamentais de inventários da EIA e arbitra um debate *Bull vs Bear*. O relatório markdown resultante (`morning_market_brief.md`) é automaticamente injetado no prompt de sistema do LLM Textual durante o ciclo diário de negociação, concedendo à IA principal uma profunda memória contextual e consciência fundamental sem desacelerar a execução no mercado em tempo real.
- **🏛️ Conselho de Fim de Semana (Memória Estratégica)**: Uma retrospetiva LLM multipessoas semanal (`src/council/weekend_council.py`) executada todos os **sábados à 01:00** via `schedule.py`. Seis personas — cada uma numa **família de modelo Ollama distinta** (Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / LFM 2.5 / Mistral Nemo 12B) para uma genuína diversidade de raciocínio — deliberam num protocolo de 4 rondas (Portão de Reformulação do Problema → Análise com POSICIONAMENTO explícito → Debate 1-vs-1 → Síntese do Juiz) com mecanismos anti-pensamento de grupo (quota de dissidência, veredito de não-resolvido-primeiro). O Juiz (Qwen3.5-9B-MTP) emite um posicionamento por ticker que se torna o **11.º voto ponderado** (9,5%) no consenso em tempo real, com a confiança decaída linearmente ao longo de 7 dias. Orçamentos de tokens generosos (`num_predict` até 12000, `num_ctx` até 65536) e uma janela de agendador de 48 horas acomodam modelos de raciocínio na CPU. O conselho analisa dados reais de PROD: precisão dos modelos (`model_performance.db`), métricas de carteira e alertas críticos (`performance_monitor.db`) e o diário de trading executado. Instale os 6 modelos necessários com `uv run python setup_council_models.py`. Ver `docs/ADR-003-weekend-council-11th-voice.md`.
- **Sentimento de Notícias e Blockchain**: Integração do **AlphaEar** e **Hyperliquid** para capturar o sentimento social e especulativo.
- **Agendador Automatizado**: Script `schedule.py` para execução contínua (8:30 - 18:00) num servidor.
- **Gestão Centralizada de Risco**: O `AdvancedRiskManager` centraliza a lógica de Anti-Perda (Stop-Loss) e Trailing Stop. Os modelos individuais já não gerem estes riscos, garantindo uma estratégia de proteção de capital unificada e estrita em diferentes regimes de mercado.
- **Contratos de Dados Estritos**: Todos os modelos de IA são totalmente padronizados para devolver um dataclass fortemente tipado `ModelResult` (`signal`, `confidence`, `reasoning`), garantindo 100% de uniformidade no motor de consenso.
- **Saúde do Código Auditada**: O projeto mantém um padrão de saúde de código **Grau B** através de auditorias automatizadas (0 código morto, alto índice de manutenibilidade).
- **Backtesting de Produção**: Motor de backtest autónomo (`backtest_prod.py`) que reproduz sinais reais de produção contra preços reais com taxas da T212 — sem dependências externas.
- **Controlo do Dump de Depuração**: Defina `TRADING_DEBUG_DUMP=0` para desativar o dump de falhas de LLM limitado a 5 MB em `data_cache/llm_debug_fail.txt`.

### 💻 Stack Tecnológica

- **Linguagem**: `Python 3.12+`
- **Cálculos e Dados**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **IA e LLM**: `google-genai` (Gemini), `requests`, `ollama`
- **Web Scraping e Busca**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualização**: `matplotlib` (backend Agg para segurança de threads), `seaborn`, `mplfinance`
- **Utilitários**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Desempenho e Hardware
O sistema foi projetado para ser **eficiente em hardware de consumo** sem a necessidade de uma GPU dedicada.
- **Apenas CPU**: A inferência do LLM (Gemma 4 12B Q6_K via Ollama) e o TimesFM correm inteiramente na CPU. A taxa de transferência é de ~3–4 tokens/s numa CPU moderna de 8 núcleos.
- **RAM Recomendada**: Mínimo de 16 GB (sugere-se 32 GB para rodar o Gemma 4 12B confortavelmente ao lado do TimesFM e do TensorTrade).
- **Concorrência do Ollama**: Defina `OLLAMA_NUM_PARALLEL=8` (já no `.env` recomendado) para que múltiplas chamadas LLM possam partilhar a carga do modelo. Com o orçamento de contexto predefinido de 4 GB, os slots paralelos obtêm ~512 tokens cada — o Ollama serializará se os prompts excederem o contexto por slot, mas o `ThreadPoolExecutor` mantém a sobreposição de tempo útil (wall-clock) benéfica para passos limitados por I/O (busca de notícias, crawl web, modelos de CPU).
- **Tempo de Execução**: ~6 a 9 minutos por ticker na CPU (frio), ~3 a 5 minutos por ticker com acerto no cache da query de pesquisa. A predefinição corre dois tickers (CRUDP.PA + SXRV.DE), portanto preveja ~15 min no total.
- **Tempo Limite do Ciclo**: Cada ciclo de ticker está limitado a 40 min (`CYCLE_TIMEOUT_SECONDS`). Se excedido, HOLD é aplicado e o próximo ticker começa imediatamente.
- **Velocidade da API**: Integração ultra-rápida com a Trading 212 (<1s para recuperação de preço ao vivo).

---



### 🧠 Arquitetura de IA e LLM (Gemini + Fallback Local)
O sistema alavanca uma arquitetura altamente robusta e multinível para garantir o máximo de tempo de atividade e uma tomada de decisão inteligente, profundamente integrada em `main.py` e no Conselho de Fim de Semana.

- **Cascata de Fallback de 4 Níveis**:
  1. **Nível Pago do Gemini (`GEMINI_API_KEY_PAY`)**: Prioridade mais alta. Usa modelos avançados como o Gemini 2.5 Pro para raciocínio complexo, visão de gráficos técnicos e decisões finais de operação.
  2. **Nível Gratuito do Gemini (`GEMINI_API_KEY`)**: Usado para tarefas mais leves e de alto volume, como resumo de contexto web.
  3. **Proxies de API LLM Gratuitos**: Backup via `free-llm-api-keys`.
  4. **Ollama Local**: Fallback na CPU 100% robusto e offline se todos os serviços na nuvem falharem.
- **Proteção de Custos**: O nível pago é limitado por um orçamento de custo de 30 dias rolante (`GEMINI_PAY_MONTHLY_BUDGET_EUR`, padrão 8,6 €/mês) — o custo de cada chamada é calculado a partir do uso real de tokens × o preço do modelo e acumulado; quando o orçamento é atingido, as chamadas recaem para o nível gratuito / Ollama. Um backstop diário (`GEMINI_PAY_DAILY_CAP`, padrão 200) protege contra loops descontrolados.
- **Integração**: O motor principal de execução diária (`main.py`) usa o Gemini para o consenso multi-modelo em tempo real, enquanto o Conselho de Fim de Semana assíncrono (`council`) integra o Gemini especificamente para certos papéis (como o Juiz e o Cético) ao lado de diversos modelos locais do Ollama.

### 🧠 FinAcumen (Memória Financeira)
A arquitetura FinAcumen foi integrada para dotar os modelos de IA locais de uma **memória de experiência** e de ferramentas determinísticas. Isto resolve o problema da amnésia dos LLMs.
- O FinAcumen funciona **de forma assíncrona durante a noite** (via `schedule.py`) para beneficiar de toda a potência da CPU sem bloquear os ciclos de trading.
- O seu relatório qualitativo profundo é automaticamente adicionado ao **Morning Market Brief** para guiar o LLM de decisão ao longo de todo o dia de trading.

## 📂 Estrutura do Projeto

O projeto é organizado de forma modular para melhor manutenção.

```
Trading-AI/
├── morning_brief/                   # Agente autónomo noturno para análise fundamental profunda
│   ├── morning_brief.py             # Orquestrador do agente e configuração smolagents
│   └── output/                      # Relatórios markdown diários gerados (morning_market_brief.md)
├── src/                             # Módulos principais
│   ├── adaptive_weight_manager.py   # Ponderação dinâmica de modelos com base no desempenho
│   ├── advanced_risk_manager.py     # Gestão de risco e dimensionamento sensíveis à tendência
│   ├── bootstrap.py                 # Lógica de inicialização principal
│   ├── chart_generator.py           # Gera gráficos técnicos para o LLM visual
│   ├── classic_model.py             # Conjunto de modelos quantitativos Scikit-learn
│   ├── config_weights.py            # Configuração de pesos base do motor híbrido
│   ├── data.py                      # Busca, cache e pré-processamento de dados
│   ├── database.py                  # Gestão da base de dados SQLite para métricas
│   ├── eia_client.py                # Cliente da API da Energy Information Administration
│   ├── enhanced_decision_engine.py  # Motor de fusão híbrida que orquestra todos os modelos
│   ├── enhanced_trading_example.py  # Scripts de exemplo para utilização de modelos
│   ├── features.py                  # Engenharia de características técnicas e macroeconômicas
│   ├── grebenkov_model.py           # Modelo matemático de Seguir-Tendência (Agnostic Risk Parity)
│   ├── hmm_model.py                 # Modelo de Markov Oculto para deteção de regime
│   ├── llm_client.py                # Integração Ollama para inferência LLM local
│   ├── news_fetcher.py              # Crawling e análise de notícias financeiras
│   ├── oil_bench_model.py           # Modelo de trading de WTI especializado em energia
│   ├── performance_monitor.py       # Registo de precisão e histórico dos modelos
│   ├── read_simul.py                # Ferramentas para leitura de saídas de simulação
│   ├── sentiment_analysis.py        # Integração de sentimento Alpha Vantage e AlphaEar
│   ├── t212_executor.py             # Execução real e carteira via API Trading 212
│   ├── tensortrade_model.py         # Sinal de Aprendizado por Reforço (PPO)
│   ├── timesfm_model.py             # Integração de previsão de séries temporais TimesFM 2.5
│   └── web_researcher.py            # Web scraping macroeconômico com Crawl4AI
├── data_cache/                       # Todos os caches (gitignored)
│   ├── *.parquet                     # Dados OHLCV por ticker (yfinance)
│   ├── macro/                        # Séries temporais macro (FRED, multi-fonte)
│   ├── search_queries/               # Cache LLM de 24h de queries de pesquisa (por ticker+data+price-sig)
│   └── llm_debug_fail.txt            # Dump de falhas LLM limitado a 5 MB — desative com TRADING_DEBUG_DUMP=0
├── tests/                            # Scripts de teste e validação
│   ├── test_full_cycle.py            # Teste end-to-end compra/espera/venda na T212
│   ├── test_enhanced_decision_engine.py # Testes do motor de fusão híbrida
│   ├── check_llm_json.py             # Diagnóstico JSON-schema do LLM (testa todos os 4 pontos de chamada Ollama)
│   ├── check_live.py                 # Script de verificação de preços de mercado ao vivo
│   └── ...                           # Outros testes unitários e de integração
├── i18n/                            # Internacionalização (READMEs traduzidos)
├── assets/                          # Recursos estáticos (imagens, banners)
├── memory-bank/                     # Estado determinístico de 4 ficheiros + contexto detalhado (ver AGENTS.md §1)
├── backtest_prod.py                 # Motor autónomo de backtesting de produção
├── main.py                          # Ponto de entrada único (Análise e Trading)
├── pyproject.toml                   # Dependências e configuração do projeto (uv)
├── refresh_cache.py                 # Utilitário CLI para forçar a atualização do cache Parquet
├── schedule.py                      # Agendador ao vivo para execução automatizada
├── setup_timesfm.py                 # Script de instalação do vendor do TimesFM 2.5
├── .env.example                     # Variáveis de ambiente de exemplo
└── README.md                        # Esta documentação
```

---

## 🚀 Início Rápido

Siga estas etapas para configurar o seu ambiente de desenvolvimento local.

### ✅ Pré-requisitos

- Python 3.12+ (via `uv`)
- [Ollama](https://ollama.com/) instalado e a correr localmente.
- Modelo LLM descarregado: `ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`
- **Modelos do Conselho de Fim de Semana** (opcional, mas necessários para a diversidade de raciocínio do conselho): o conselho executa cada persona numa *família* de modelo diferente (Gemma / GLM / Qwen / LFM). Instale-os todos de uma só vez com `uv run python setup_council_models.py`.

### ⚙️ Instalação

1.  **Clonar o repositório:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Instalar o `uv` (se ainda não o fez):**
    Veja [astral.sh/uv](https://astral.sh/uv) para instruções de instalação.

3.  **Criar e ativar o ambiente virtual (Passo CRUCIAL):**
    Deve criar e ativar o `.venv` antes de instalar os modelos de base.
    ```bash
    uv venv
    source .venv/bin/activate  # No Windows, use `.\.venv\Scripts\activate.ps1`
    ```

4.  **Instalar Modelos de Base:**
    Execute os scripts de instalação para clonar os modelos em `vendor/` e aplicar os patches:
    ```bash
    python setup_timesfm.py
    ```

5.  **Inicializar e sincronizar o ambiente:**
    ```bash
    uv sync
    ```

6.  **Instalar navegadores para pesquisa Web (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

7.  **Configurar as suas chaves de API:**
    Crie um ficheiro `.env` na raiz do projeto:
    ```
    ALPHA_VANTAGE_API_KEY="SUA_CHAVE"
    EIA_API_KEY="SUA_CHAVE"

    # Opcional, mas altamente recomendado: Integração com Gemini AI
    GEMINI_API_KEY_PAY="SUA_CHAVE_NIVEL_PAGO"  # Para raciocínio/visão complexos (Gemini 2.5 Pro)
    GEMINI_API_KEY="SUA_CHAVE_NIVEL_GRATUITO"  # Para tarefas mais leves (resumo)
    GEMINI_PAY_MONTHLY_BUDGET_EUR=8.6        # Orçamento de custo de 30 dias rolante (€) — guarda de faturação principal
    GEMINI_PAY_DAILY_CAP=200                 # Backstop: máximo de chamadas API pagas por dia
    ```

---

## 🛠️ Uso

O sistema treina os seus modelos nos dados mais recentes em cada execução antes de tomar uma decisão.

### Modo Simulação (Paper Trading)

Para testar o sistema sem risco com um capital fictício de 1000 €, use a flag `--simul`. O sistema gerirá um histórico rigoroso de compras e vendas.

```sh
# Executar uma análise simulada (Predefinição: SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Executar em Petróleo (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Execução Real (Trading 212)

O sistema está agora **totalmente integrado** com a Trading 212:
- **Verificação de Portfólio**: Antes de qualquer ação, o robô consulta o seu caixa e posições reais.
- **Gestão da API**: Inclui mecanismos de repetição automática contra limites de solicitações (Rate Limiting).

```sh
# Executar análise com execução real (Demo ou Real de acordo com o .env)
uv run main.py --t212
```

---

## 🧪 Backtesting de Produção

O sistema inclui um **motor autónomo de backtesting de produção** (`backtest_prod.py`) que reproduz os sinais reais de produção de `logs_prod/trading_journal.csv` contra preços reais dos ficheiros Parquet em `data_cache/`.

### Características
- **Sinais reais**: Reproduz as decisões exatas do motor híbrido de 12 modelos.
- **Preços reais**: Usa dados OHLCV reais dos ETFs (SXRV.DE, CRUDP.PA) — sem proxies dos EUA.
- **Taxas da T212**: Simula o modelo de taxas de 0,1% por operação da Trading 212.
- **Comparação de base**: Calcula automaticamente o desempenho de buy-and-hold como referência.
- **Métricas**: Rácio de Sharpe, Drawdown Máximo, Taxa de Vitória, Alfa, Retorno Total por ticker.

### Utilização

```bash
uv run python backtest_prod.py
```

Os resultados são guardados em `logs_prod/backtest_report.json` com curvas de capital em CSV.

---

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para fazer um fork do projeto e abrir um Pull Request.

---

## 📜 Licença

Distribuído sob a Licença MIT.

---

## 📧 Contato

Link do Projeto: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)

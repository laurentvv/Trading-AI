# Product Context

## 1. Problem Statement
Retail and semi-professional traders often rely on a mix of technical analysis and qualitative judgment to make trading decisions. This process can be time-consuming, prone to emotional biases, and difficult to test systematically. Existing tools can be either too simplistic, failing to capture market complexity, or too complex, requiring a steep learning curve.

## 2. Vision
This project aims to create a powerful yet accessible trading decision support system. It bridges the gap between quantitative analysis and qualitative, human-like reasoning by leveraging a hybrid AI model. The system will empower the user to make more informed, data-driven, and systematically-tested trading decisions on NASDAQ ETFs.

## 3. How It Should Work
The user will run a single script from the command line. The script will:
1. Fetch the latest market data for a specified NASDAQ ETF, using a local cache to speed up the process.
2. Process the data to calculate a wide range of technical indicators.
3. Feed this information into two parallel AI models:
    - A `scikit-learn` model trained for signal prediction.
    - An LLM (via Ollama) that provides a signal and a narrative market analysis.
4. Combine the outputs of both models into a single, actionable trading recommendation (e.g., "STRONG BUY", "HOLD", "SELL").
5. Output the decision, the LLM's analysis, and key performance metrics from a backtest to the console.
6. Generate plots visualizing the strategy's performance.

## 4. User Experience Goals
- **Simplicity**: The system should be easy to run with a single command.
- **Clarity**: The output should be clear and provide both a direct signal and the reasoning behind it (via the LLM analysis).
- **Transparency**: The backtesting results and performance metrics should be transparent, allowing the user to understand the strategy's historical performance and risks.

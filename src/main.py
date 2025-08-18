import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import subprocess
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import refactored modules
from data import get_etf_data
from features import create_technical_indicators, create_features, select_features
from classic_model import train_ensemble_model, get_classic_prediction
from llm_client import get_llm_decision, get_visual_llm_decision
from sentiment_analysis import get_sentiment_decision_from_score
from chart_generator import generate_chart_image
from backtest import run_backtest

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants for the Alpha Vantage API ---
# IMPORTANT: It is strongly recommended to use an environment variable for your API key.
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    logger.critical("CRITICAL: The ALPHA_VANTAGE_API_KEY environment variable is not set.")
    logger.critical("Please set it to your Alpha Vantage API key.")
    sys.exit(1)

# --- Constants for the hybrid decision engine ---
W_CLASSIC = 1/4
W_LLM_TEXT = 1/4
W_LLM_VISUAL = 1/4
W_SENTIMENT = 1/4

def get_hybrid_decision(classic_pred: int, classic_conf: float, text_llm_decision: dict, visual_llm_decision: dict, sentiment_decision: dict) -> tuple[str, float]:
    """
    Combines the decisions of the 4 models for a final decision.
    """
    signal_map = {"BUY": 1, "HOLD": 0, "SELL": -1}

    classic_score = (1 if classic_pred == 1 else -1) * classic_conf

    text_llm_score = signal_map.get(text_llm_decision.get('signal', 'HOLD').upper(), 0) * text_llm_decision.get('confidence', 0.0)

    visual_llm_score = signal_map.get(visual_llm_decision.get('signal', 'HOLD').upper(), 0) * visual_llm_decision.get('confidence', 0.0)

    sentiment_score = signal_map.get(sentiment_decision.get('signal', 'HOLD').upper(), 0) * sentiment_decision.get('confidence', 0.0)

    final_score = (classic_score * W_CLASSIC) + (text_llm_score * W_LLM_TEXT) + (visual_llm_score * W_LLM_VISUAL) + (sentiment_score * W_SENTIMENT)

    if final_score > 0.5:
        decision = "STRONG BUY"
    elif 0.1 < final_score <= 0.5:
        decision = "BUY"
    elif -0.1 <= final_score <= 0.1:
        decision = "HOLD"
    elif -0.5 < final_score < -0.1:
        decision = "SELL"
    else:
        decision = "STRONG SELL"
    return decision, final_score

def plot_analysis(data, backtest_data):
    """Advanced visualizations"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('AI Trading System Analysis', fontsize=16, fontweight='bold')
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='Close Price', alpha=0.7)
    ax1.plot(data.index, data['MA_20'], label='MA 20', alpha=0.7, linestyle='--')
    ax1.plot(data.index, data['MA_50'], label='MA 50', alpha=0.7, linestyle='--')
    buy_points = backtest_data['Position'].diff() == 1
    sell_points = backtest_data['Position'].diff() == -1
    ax1.scatter(backtest_data.index[buy_points], backtest_data['Close'][buy_points], color='green', marker='^', s=100, label='Buy', zorder=5)
    ax1.scatter(backtest_data.index[sell_points], backtest_data['Close'][sell_points], color='red', marker='v', s=100, label='Sell', zorder=5)
    ax1.set_title('Trading Signals and Close Price')
    ax1.legend(); ax1.grid(True)
    ax2 = axes[1]
    ax2.plot(backtest_data.index, backtest_data['Cumulative_Strategy'], label='Strategy')
    ax2.plot(backtest_data.index, backtest_data['Cumulative_Returns'], label='Benchmark (Buy & Hold)')
    ax2.set_title('Cumulative Performance')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('backtest_analysis.png')
    logger.info("Analysis chart saved as 'backtest_analysis.png'")
    plt.close()

def run_walk_forward_backtest(data_with_features: pd.DataFrame, train_period_days: int, test_period_days: int):
    """
    Runs a backtest using walk-forward validation and simulates the 3 models.
    """
    all_signals = []
    num_iterations = (len(data_with_features) - train_period_days) // test_period_days

    with tqdm(total=num_iterations, desc="Walk-Forward Backtest") as pbar:
        start_index = 0
        while start_index + train_period_days + test_period_days <= len(data_with_features):
            train_end_index = start_index + train_period_days
            test_end_index = train_end_index + test_period_days
            train_data = data_with_features.iloc[start_index:train_end_index]
            test_data = data_with_features.iloc[train_end_index:test_end_index]

            X_train, y_train, _ = select_features(train_data)
            classic_model, scaler, _, _ = train_ensemble_model(X_train, y_train)

            fold_signals = []
            feature_cols = [col for col in X_train.columns if col in scaler.feature_names_in_]

            for i in range(len(test_data)):
                current_features = test_data.iloc[i:i+1]
                current_features_subset = current_features[feature_cols]
                classic_pred, classic_conf = get_classic_prediction(classic_model, scaler, current_features_subset)

                # Simulate LLM decisions for backtest performance
                if classic_pred == 1 and classic_conf > 0.7:
                    sim_signal = "BUY"; sim_conf = 0.8
                elif classic_pred == 0 and classic_conf > 0.7:
                    sim_signal = "SELL"; sim_conf = 0.8
                else:
                    sim_signal = "HOLD"; sim_conf = 0.5

                text_llm_sim = {"signal": sim_signal, "confidence": sim_conf}
                visual_llm_sim = {"signal": sim_signal, "confidence": sim_conf} # Simple simulation
                sentiment_sim = {"signal": sim_signal, "confidence": sim_conf} # Simple simulation for sentiment

                final_decision, _ = get_hybrid_decision(classic_pred, classic_conf, text_llm_sim, visual_llm_sim, sentiment_sim)

                fold_signals.append(1 if "BUY" in final_decision else 0)

            all_signals.append(pd.Series(fold_signals, index=test_data.index))
            start_index += test_period_days
            pbar.update(1)

    if not all_signals:
        logger.error("Not enough data to perform a walk-forward backtest.")
        return None, None

    final_signals = pd.concat(all_signals)
    aligned_signals = pd.Series(index=data_with_features.index, dtype=int).fillna(0)
    aligned_signals.update(final_signals)
    return run_backtest(data_with_features, aligned_signals)

def main():
    """
    Main function to orchestrate the trading system.
    """
    TICKER = 'QQQ'
    CHART_OUTPUT_PATH = Path("trading_chart.png")

    logger.info("Step 1: Retrieving and preparing data...")
    hist_data, info = get_etf_data(ticker=TICKER)
    data_with_indicators = create_technical_indicators(hist_data)
    data_with_features = create_features(data_with_indicators)

    logger.info("\nStep 2: Running backtest with walk-forward validation...")
    backtest_data, performance = run_walk_forward_backtest(data_with_features, 252, 63)

    if performance:
        logger.info("\n=== WALK-FORWARD BACKTEST RESULTS ===")
        for metric, value in performance.items():
            logger.info(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")

    logger.info("\nStep 3: Generating final decision for today...")
    logger.info("Training final model on all available data...")
    X, y, _ = select_features(data_with_features)
    final_classic_model, final_scaler, _, _ = train_ensemble_model(X, y)

    latest_data = data_with_features.tail(1)
    feature_cols = [col for col in X.columns if col in final_scaler.feature_names_in_]
    latest_features_subset = latest_data[feature_cols]

    # Generate chart for visual analysis
    logger.info(f"Generating analysis chart for {TICKER}...")
    chart_generated = generate_chart_image(data_with_features, CHART_OUTPUT_PATH, title=f"{TICKER} - 6 Month Chart")

    # Get predictions from the four models
    classic_pred, classic_conf = get_classic_prediction(final_classic_model, final_scaler, latest_features_subset)
    text_llm_decision = get_llm_decision(latest_data)
    visual_llm_decision = get_visual_llm_decision(CHART_OUTPUT_PATH) if chart_generated else {"signal": "HOLD", "confidence": 0.0, "analysis": "Chart generation failed."}
    
    # Fetch news and sentiment using the news_fetcher.py script
    logger.info("Fetching live news and sentiment from Alpha Vantage...")
    try:
        # Construct the path to the script and the python executable
        script_path = Path(__file__).parent / "news_fetcher.py"
        python_executable = sys.executable
        
        # Run the script as a subprocess
        process = subprocess.run(
            [python_executable, str(script_path), TICKER, ALPHA_VANTAGE_API_KEY],
            capture_output=True,
            text=True,
            check=True
        )
        news_data = json.loads(process.stdout)
        sentiment_score = news_data.get("sentiment", 0)
        logger.info(f"Successfully fetched news. Overall sentiment score: {sentiment_score:.2f}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to fetch or parse news headlines: {e}")
        sentiment_score = 0

    sentiment_decision = get_sentiment_decision_from_score(sentiment_score)


    # Get final hybrid decision
    final_decision, final_score = get_hybrid_decision(classic_pred, classic_conf, text_llm_decision, visual_llm_decision, sentiment_decision)

    analysis_date = latest_data.index[0].date()
    logger.info(f"\n=== FINAL HYBRID DECISION FOR {analysis_date} ===")
    logger.info(f"Classic Prediction  : {'BUY' if classic_pred == 1 else 'SELL/HOLD'} (Confidence: {classic_conf:.2f})")
    logger.info(f"LLM Decision (Text)  : {text_llm_decision.get('signal')} (Confidence: {text_llm_decision.get('confidence', 0.0):.2f})")
    logger.info(f"LLM Decision (Visual) : {visual_llm_decision.get('signal')} (Confidence: {visual_llm_decision.get('confidence', 0.0):.2f})")
    logger.info(f"LLM Decision (Sentiment) : {sentiment_decision.get('signal')} (Confidence: {sentiment_decision.get('confidence', 0.0):.2f})")
    logger.info(f"Visual Analysis      : {visual_llm_decision.get('analysis')}")
    logger.info("-" * 40)
    logger.info(f"Final Hybrid Score   : {final_score:.4f}")
    logger.info(f"FINAL DECISION       : {final_decision}")
    logger.info("=" * 40)

    if backtest_data is not None:
        logger.info("\nStep 4: Generating backtest analysis charts...")
        plot_analysis(data_with_features, backtest_data)


if __name__ == "__main__":
    main()

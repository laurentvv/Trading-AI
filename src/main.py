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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load environment variables from .env file
load_dotenv()

# Import refactored modules
# Corrected import order and ensured all necessary modules are imported
from data import get_etf_data, fetch_macro_data_for_date
from features import create_technical_indicators, create_features, select_features
from classic_model import train_ensemble_model, get_classic_prediction
from llm_client import get_llm_decision, get_visual_llm_decision
from sentiment_analysis import get_sentiment_decision_from_score
from chart_generator import generate_chart_image
from backtest import run_backtest
from xai_explainer import explain_model_prediction, plot_shap_waterfall

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
W_CLASSIC = 0.4
W_LLM_TEXT = 0.25
W_LLM_VISUAL = 0.25
W_SENTIMENT = 0.1

# Placeholder for future macro weight
W_MACRO = 0.0 # Currently not directly used in final decision, but fetched for context/LMM

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

def run_walk_forward_backtest(data_with_features: pd.DataFrame, train_period_days: int, test_period_days: int, macro_context: dict = None):
    """
    Runs a backtest using walk-forward validation and simulates the 3 models.
    Note: For a full backtest with historical macro data, a more complex alignment is needed.
    This version uses a static macro context for simplicity.
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
    # Initialize macro_context to an empty dict for backtest compatibility
    macro_context = {}
    # Pass macro context to create_features
    data_with_features = create_features(data_with_indicators, macro_context)
    
    # --- NEW: Fetch macroeconomic data for context ---
    analysis_date = data_with_features.index[-1]
    macro_context = fetch_macro_data_for_date(analysis_date)
    logger.info(f"Macro Context for {analysis_date.date()}: {macro_context}")
    # Re-run feature creation with the actual macro context for the final prediction
    data_with_features_final = create_features(data_with_indicators, macro_context)

    logger.info("\nStep 2: Running backtest with walk-forward validation...")
    # For backtest, we currently pass a None or empty macro_context as full historical alignment is complex.
    # The features.py is designed to handle missing macro columns gracefully.
    backtest_data, performance = run_walk_forward_backtest(data_with_features, 252, 63, macro_context={})

    if performance:
        logger.info("\n=== WALK-FORWARD BACKTEST RESULTS ===")
        for metric, value in performance.items():
            logger.info(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")

    logger.info("\nStep 3: Generating final decision for today...")
    logger.info("Training final model on all available data...")
    # Use the data with macro features for the final model
    X, y, _ = select_features(data_with_features_final)
    logger.info(f"Final model training data - X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Final model training data - y value counts:\n{y.value_counts()}")
    
    # Check if we have enough data and at least 2 classes
    if len(y) == 0:
        logger.error("No training data available for the final model. Cannot proceed.")
        return # Or handle this case appropriately
    elif len(y.unique()) < 2:
        logger.warning("Target variable has only one class in the training set. Cannot train the final model properly. Skipping final model training and prediction.")
        # We can't proceed with training/prediction. Let's set default values.
        final_classic_model = None
        final_scaler = None
        classic_pred = 0  # Default to 'SELL/HOLD'
        classic_conf = 0.5 # Default confidence
    else:
        # Check for NaNs in raw features before scaling
        if X.isna().any().any():
            logger.warning("Feature matrix X contains NaNs before scaling. Filling with 0.")
            X = X.fillna(0)
            
        # Train on all data for the final model, no train/test split
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)
        
        # Check for NaNs in scaled data (should not happen after StandardScaler with filled NaNs, but good to be sure)
        if pd.isna(X_scaled).any():
             logger.error("Scaled feature matrix X_scaled contains NaNs. Cannot train the model.")
             final_classic_model = None
             final_scaler = None
             classic_pred = 0
             classic_conf = 0.5
        else:
            # Re-initialize and train the best model type (LogisticRegression) on all data
            # This mimics the logic in train_ensemble_model but without splitting
            final_classic_model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
            final_classic_model.fit(X_scaled, y)

    # For the final prediction, we use the features from the last row of the full dataset (including macro)
    # The target 'y' for this last row is not used in prediction, only its features (X) are.
    latest_data_with_macro = data_with_features_final.tail(1)
    feature_cols = [col for col in X.columns if col in getattr(final_scaler, 'feature_names_in_', X.columns)]
    latest_features_subset = latest_data_with_macro[feature_cols]
    
    # Check for NaNs in the latest features subset before prediction
    if latest_features_subset.isna().any().any():
        logger.warning("Latest feature subset for prediction contains NaNs. Filling with 0.")
        latest_features_subset = latest_features_subset.fillna(0)
        
    # Also prepare data without macro features for LLMs if needed
    # (Currently, LLMs might not use them, but good to have consistency)
    latest_data_without_macro = data_with_features.tail(1)

    # Generate chart for visual analysis
    logger.info(f"Generating analysis chart for {TICKER}...")
    chart_generated = generate_chart_image(data_with_features, CHART_OUTPUT_PATH, title=f"{TICKER} - 6 Month Chart")

    # Get predictions from the four models
    if final_classic_model is not None:
        classic_pred, classic_conf = get_classic_prediction(final_classic_model, final_scaler, latest_features_subset)
        
        # --- NEW: XAI Explanation ---
        logger.info("Generating SHAP explanation for the classic model's prediction...")
        try:
            feature_names_for_model = list(latest_features_subset.columns)
            explanation = explain_model_prediction(
                final_classic_model, 
                final_scaler, 
                latest_features_subset, 
                feature_names_for_model, 
                instance_index=0 # We are explaining the single row in latest_features_subset
            )
            if explanation:
                plot_shap_waterfall(explanation, save_path="shap_waterfall.png")
                logger.info("SHAP waterfall plot saved as 'shap_waterfall.png'")
            else:
                logger.warning("Failed to generate SHAP explanation.")
        except Exception as e:
            logger.error(f"Error during XAI explanation generation or plotting: {e}")
        # --- END NEW ---
        
    else:
        # Use default values if model couldn't be trained
        classic_pred, classic_conf = 0, 0.5 
    text_llm_decision = get_llm_decision(latest_data_without_macro) # Use data without macro for LLMs to keep context consistent
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

    # --- Rich-based Output (French) ---
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    # --- Traductions ---
    signal_translations = {
        "BUY": "ACHAT",
        "SELL": "VENTE",
        "HOLD": "NEUTRE",
        "SELL/HOLD": "VENTE/NEUTRE",
        "STRONG BUY": "ACHAT FORT",
        "STRONG SELL": "VENTE FORTE"
    }
    reliability_translations = {
        "Very High": "Très Élevée",
        "High": "Élevée",
        "Moderate": "Modérée",
        "Low": "Faible"
    }

    # 1. Create a table for the model breakdown
    table = Table(show_header=True, header_style="bold magenta", title_justify="center")
    table.add_column("Modèle", style="dim", width=25)
    table.add_column("Signal", justify="center")
    table.add_column("Confiance", justify="right")

    def get_signal_style(signal):
        if signal is None: return "bold yellow"
        if "BUY" in signal: return "bold green"
        if "SELL" in signal: return "bold red"
        return "bold yellow"

    # Helper to translate signals
    def translate_signal(signal_en):
        return signal_translations.get(signal_en, signal_en)

    classic_signal_en = 'BUY' if classic_pred == 1 else 'SELL/HOLD'
    table.add_row(
        "Quantitatif Classique",
        Text(translate_signal(classic_signal_en), style=get_signal_style(classic_signal_en)),
        f"{classic_conf:.2%}"
    )
    text_llm_signal_en = text_llm_decision.get('signal', 'HOLD')
    table.add_row(
        "LLM (Analyse Textuelle)",
        Text(translate_signal(text_llm_signal_en), style=get_signal_style(text_llm_signal_en)),
        f"{text_llm_decision.get('confidence', 0.0):.2%}"
    )
    visual_llm_signal_en = visual_llm_decision.get('signal', 'HOLD')
    table.add_row(
        "LLM (Analyse Visuelle)",
        Text(translate_signal(visual_llm_signal_en), style=get_signal_style(visual_llm_signal_en)),
        f"{visual_llm_decision.get('confidence', 0.0):.2%}"
    )
    sentiment_signal_en = sentiment_decision.get('signal', 'HOLD')
    table.add_row(
        "LLM (Analyse de Sentiment)",
        Text(translate_signal(sentiment_signal_en), style=get_signal_style(sentiment_signal_en)),
        f"{sentiment_decision.get('confidence', 0.0):.2%}"
    )

    # 2. Determine reliability
    abs_score = abs(final_score)
    if abs_score > 0.6:
        reliability_en = "Very High"
        reliability_style = "bold green"
    elif abs_score > 0.4:
        reliability_en = "High"
        reliability_style = "green"
    elif abs_score > 0.2:
        reliability_en = "Moderate"
        reliability_style = "yellow"
    else:
        reliability_en = "Low"
        reliability_style = "red"
    
    reliability_fr = reliability_translations.get(reliability_en, reliability_en)

    # 3. Create the final decision panel
    decision_text = Text(translate_signal(final_decision), justify="center", style=get_signal_style(final_decision))
    score_text = Text(f"Score: {final_score:.4f}", justify="center")
    reliability_text = Text(f"Fiabilité: {reliability_fr}", justify="center", style=reliability_style)

    output_text = Text.assemble(
        decision_text, "\n",
        score_text, "\n",
        reliability_text
    )

    # 4. Visual Analysis Panel
    visual_analysis_panel = Panel(
        Text(visual_llm_decision.get('analysis', 'N/A'), style="italic"),
        title="[bold]Analyse LLM Visuelle[/bold]",
        border_style="dim"
    )

    console.print("")
    console.print(Panel(
        table,
        title=f"[bold]Analyse de la Décision Hybride pour {TICKER} le {analysis_date.date()}[/bold]",
        border_style="blue"
    ))
    console.print(visual_analysis_panel)
    console.print(Panel(
        output_text,
        title="[bold]Décision Finale[/bold]",
        border_style="bold"
    ))
    console.print("")

    if backtest_data is not None:
        logger.info("\nStep 4: Generating backtest analysis charts...")
        plot_analysis(data_with_features, backtest_data)


if __name__ == "__main__":
    main()
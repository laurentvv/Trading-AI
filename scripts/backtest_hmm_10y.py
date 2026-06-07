import sys
from pathlib import Path
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hmm_model import HMMDecisionModel


def run_benchmark():
    ticker = "QQQ"
    print(f"Téléchargement de 10 ans de données pour {ticker}...")
    data = yf.download(ticker, period="10y", progress=False)

    if data.empty:
        print("Erreur de téléchargement des données.")
        return

    print(f"Données téléchargées : {len(data)} jours de trading.")

    model = HMMDecisionModel(lookback=252, baum_welch_iterations=5)

    # Portfolio setup
    initial_capital = 10000.0
    cash = initial_capital
    position = 0.0

    capital_history = []
    bh_capital_history = []

    bh_cash = initial_capital
    bh_position = 0.0

    # Buy and hold strategy: Buy on first day
    first_price = float(
        data.iloc[0]["Close"].iloc[0] if isinstance(data.iloc[0]["Close"], pd.Series) else data.iloc[0]["Close"]
    )
    bh_position = bh_cash / first_price
    bh_cash = 0.0

    print("Début du backtest (Fenêtre glissante). Cela peut prendre quelques instants...")

    start_idx = 100  # need enough data for the first lookback

    for i in range(start_idx, len(data)):
        current_date = data.index[i]
        current_price = float(
            data.iloc[i]["Close"].iloc[0] if isinstance(data.iloc[i]["Close"], pd.Series) else data.iloc[i]["Close"]
        )

        # Prepare window data
        window_data = data.iloc[i - 100 : i + 1].copy()

        # Avoid multi-index columns from yfinance if present
        if isinstance(window_data.columns, pd.MultiIndex):
            window_data.columns = window_data.columns.get_level_values(0)

        model_input = {"hist_data": window_data}

        result = model.predict(model_input)

        # Execute HMM Strategy Trade
        if result.signal in ["BUY", "STRONG_BUY"] and cash > 0:
            # Buy all we can
            position = cash / current_price
            cash = 0.0
        elif result.signal in ["SELL", "STRONG_SELL"] and position > 0:
            # Sell everything
            cash = position * current_price
            position = 0.0

        # Record Portfolio Values
        current_value = cash + (position * current_price)
        bh_current_value = bh_cash + (bh_position * current_price)

        capital_history.append({"Date": current_date, "Value": current_value, "Strategy": "HMM Model"})
        bh_capital_history.append({"Date": current_date, "Value": bh_current_value, "Strategy": "Buy & Hold"})

        if i % 250 == 0:
            print(f"Année {(i - start_idx) // 250 + 1}/10 complétée...")

    # Create DataFrames
    df_hmm = pd.DataFrame(capital_history).set_index("Date")
    df_bh = pd.DataFrame(bh_capital_history).set_index("Date")

    final_hmm_val = df_hmm.iloc[-1]["Value"]
    final_bh_val = df_bh.iloc[-1]["Value"]

    hmm_return = (final_hmm_val - initial_capital) / initial_capital * 100
    bh_return = (final_bh_val - initial_capital) / initial_capital * 100

    print("\n" + "=" * 40)
    print("RÉSULTATS DU BENCHMARK 10 ANS")
    print("=" * 40)
    print(f"Capital Initial : {initial_capital:.2f} $")
    print(f"HMM Model Final : {final_hmm_val:.2f} $ ({hmm_return:+.2f}%)")
    print(f"Buy & Hold Final: {final_bh_val:.2f} $ ({bh_return:+.2f}%)")
    print("=" * 40)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df_hmm.index, df_hmm["Value"], label="HMM Model (Timing)", color="blue")
    plt.plot(df_bh.index, df_bh["Value"], label="Buy & Hold (Baseline)", color="orange", alpha=0.7)
    plt.title(f"Benchmark sur 10 ans: Modèle HMM vs Buy & Hold ({ticker})")
    plt.xlabel("Date")
    plt.ylabel("Valeur du Portefeuille ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = Path("hmm_benchmark_10y.png")
    plt.savefig(out_path)
    print(f"\nGraphique généré : {out_path.absolute()}")


if __name__ == "__main__":
    run_benchmark()

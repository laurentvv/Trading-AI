import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from zipline.api import order_target_percent, record, symbol
from zipline import run_algorithm
import matplotlib.pyplot as plt
import argparse

from zipline.data.bundles import register, load
from zipline.data.bundles.csvdir import csvdir_equities

# Local AI imports
from classic_model import train_ensemble_model, get_classic_prediction

try:
    register(
        "my-bundle-US",
        csvdir_equities(
            ["daily"],
            "/app/csvdir",
        ),
        calendar_name="XPAR",
    )
except Exception:
    pass


# Minimal robust feature engineering for Zipline
def create_minimal_features(df):
    df = df.copy()

    # Needs Close, High, Low, Volume
    for w in [5, 20, 50, 200]:
        df[f"MA_{w}"] = df["Close"].rolling(window=w).mean()
        df[f"Vol_MA_{w}"] = df["Volume"].rolling(window=w).mean()
        df[f"Ret_{w}"] = df["Close"].pct_change(w)

    df["MA_Cross_5_20"] = (df["MA_5"] > df["MA_20"]).astype(int)

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std()
    df["BB_Up"] = df["BB_Mid"] + (df["BB_Std"] * 2)
    df["BB_Low"] = df["BB_Mid"] - (df["BB_Std"] * 2)

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR_14"] = true_range.rolling(14).mean()

    df.fillna(method="bfill", inplace=True)
    df.fillna(0, inplace=True)
    return df


def initialize(context):
    context.asset = symbol(context.ticker_str)

    print(f"Initializing Zipline Backtest for {context.ticker_str}...")
    import yfinance as yf

    yf_ticker = context.ticker_str.replace("_", ".")

    print(f"Fetching pre-train data for ClassicModel (2014-2018) for {yf_ticker}...")
    pre_train = yf.download(yf_ticker, start="2014-01-01", end="2017-12-31", progress=False)
    if isinstance(pre_train.columns, pd.MultiIndex):
        pre_train.columns = pre_train.columns.droplevel(1)
    pre_train.reset_index(inplace=True)
    pre_train.rename(
        columns={
            "Date": "date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close_adj",
            "Volume": "Volume",
        },
        inplace=True,
    )
    pre_train["Close"] = pre_train["Close_adj"]

    print("Generating features for pre-training...")
    features_df = create_minimal_features(pre_train)

    features_df["Target"] = (features_df["Close"].shift(-1) > features_df["Close"]).astype(int)
    features_df.dropna(inplace=True)

    X = features_df.drop(
        columns=["Target", "date", "Close_adj", "Close", "Open", "High", "Low", "Volume"], errors="ignore"
    )
    X = X.select_dtypes(include=["number"])
    y = features_df["Target"]

    print(f"Training ClassicModel on {len(X)} rows...")
    pipeline, metrics, _ = train_ensemble_model(X, y, walk_forward=False, skip_cache=True)
    context.model_pipeline = pipeline
    print(f"Pre-training complete! Model F1: {metrics.get('f1', 0):.3f}")


def handle_data(context, data):
    hist = data.history(
        context.asset, fields=["price", "open", "high", "low", "close", "volume"], bar_count=250, frequency="1d"
    )

    if len(hist) < 200:
        return

    hist.reset_index(inplace=True)
    hist.rename(
        columns={
            "index": "date",
            "price": "Close",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close_adj",
            "volume": "Volume",
        },
        inplace=True,
    )
    hist["Close"] = hist["Close_adj"]

    try:
        features_df = create_minimal_features(hist)
        latest_features = features_df.iloc[[-1]].copy()
        latest_features = latest_features.select_dtypes(include=["number"])

        cols_to_drop = ["date", "Close_adj", "Close", "Open", "High", "Low", "Volume"]
        latest_features = latest_features.drop(
            columns=[c for c in cols_to_drop if c in latest_features.columns], errors="ignore"
        )

        prediction, confidence = get_classic_prediction(context.model_pipeline, latest_features)
        current_price = data.current(context.asset, "price")

        position = context.portfolio.positions[context.asset]
        current_shares = position.amount
        avg_cost = position.cost_basis

        if confidence > 0.55:
            if prediction == 1:
                # Acheter
                order_target_percent(context.asset, 1.0)
            else:
                # Vendre SEULEMENT si on est en profit (ou pas de position)
                # "Le bench ne doit jamais vendre à perte toujours hold en attente"
                if current_shares > 0:
                    if current_price > avg_cost:
                        order_target_percent(context.asset, 0.0)  # Vendre (profit)
                    else:
                        pass  # HOLD: Ne pas vendre à perte
                else:
                    order_target_percent(context.asset, 0.0)

        record(
            price=current_price,
            prediction=prediction,
            confidence=confidence,
            avg_cost=avg_cost if current_shares > 0 else current_price,
        )

    except Exception:
        pass


def analyze(context, perf):
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))

    perf.portfolio_value.plot(ax=ax[0])
    ax[0].set_title("Portfolio Value")
    ax[0].set_ylabel("Value ($)")

    perf.price.plot(ax=ax[1])
    ax[1].set_title(f"{context.ticker_str} Price")
    ax[1].set_ylabel("Price")

    drawdown = (perf.portfolio_value.cummax() - perf.portfolio_value) / perf.portfolio_value.cummax()
    drawdown.plot(ax=ax[2], color="red")
    ax[2].set_title("Drawdown")
    ax[2].set_ylabel("%")

    plt.tight_layout()
    plt.savefig(f"{context.ticker_str}_backtest.png")
    print(f"Saved performance plot to {context.ticker_str}_backtest.png")

    first_price = perf.price.iloc[0]
    last_price = perf.price.iloc[-1]
    bh_return = (last_price / first_price) - 1

    bh_cummax = perf.price.cummax()
    bh_drawdown = ((bh_cummax - perf.price) / bh_cummax).max()

    model_return = perf.portfolio_value.iloc[-1] / perf.portfolio_value.iloc[0] - 1
    model_dd = drawdown.max()

    print("\n" + "=" * 40)
    print("      BACKTEST PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"Period: {perf.index[0].date()} to {perf.index[-1].date()}")
    print("-" * 40)
    print(f"{'Metric':<20} | {'ClassicModel':<15} | {'Buy & Hold':<15}")
    print("-" * 40)
    print(f"{'Total Return':<20} | {model_return:>14.2%} | {bh_return:>14.2%}")
    print(f"{'Max Drawdown':<20} | {model_dd:>14.2%} | {bh_drawdown:>14.2%}")
    print(f"{'Final Portfolio':<20} | ${perf.portfolio_value.iloc[-1]:>13.2f} | ${10000 * (1 + bh_return):>13.2f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zipline Backtest with ML Model")
    parser.add_argument("--ticker", type=str, default="CRUDP_PA")
    parser.add_argument("--start", type=str, default="2018-01-02")
    parser.add_argument("--end", type=str, default="2023-12-29")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    print(f"Starting Backtest for {args.ticker} from {start} to {end}")

    bundle_data = load("my-bundle-US")

    def initialize_with_args(context):
        context.ticker_str = args.ticker
        initialize(context)

    result = run_algorithm(
        start=start,
        end=end,
        initialize=initialize_with_args,
        capital_base=10000,
        handle_data=handle_data,
        analyze=analyze,
        bundle="my-bundle-US",
        data_frequency="daily",
    )

    result.to_csv(f"{args.ticker}_backtest_results.csv")

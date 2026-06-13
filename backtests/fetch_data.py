import yfinance as yf
import pandas as pd
import os
import argparse


def fetch_data(ticker, start, end, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Fetching data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end)

    # If using newer yfinance versions, it returns a MultiIndex when a single ticker is passed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.reset_index(inplace=True)
    df.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"},
        inplace=True,
    )

    # Needs: date, open, high, low, close, volume, dividend, split
    df["dividend"] = 0.0
    df["split"] = 1.0

    df = df[["date", "open", "high", "low", "close", "volume", "dividend", "split"]]
    out_file = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical data for Zipline")
    parser.add_argument("--ticker", type=str, default="CRUDP.PA")
    parser.add_argument("--start", type=str, default="2016-01-01")  # Give enough runway for 2018 backtest!
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--out", type=str, default="csvdir/daily")
    args = parser.parse_args()

    # Ensure correct Zipline naming format
    clean_ticker = args.ticker.replace(".", "_")
    fetch_data(args.ticker, args.start, args.end, args.out)

    # Rename correctly for Zipline
    import shutil

    if os.path.exists(f"{args.out}/{args.ticker}.csv") and clean_ticker != args.ticker:
        shutil.move(f"{args.out}/{args.ticker}.csv", f"{args.out}/{clean_ticker}.csv")

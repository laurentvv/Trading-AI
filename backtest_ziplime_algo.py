from ziplime.api import order_target_percent
import pandas as pd
import pytz

SIGNALS = None


async def initialize(context):
    global SIGNALS
    try:
        df = pd.read_csv("logs_prod/trading_journal.csv")
        df["date"] = pd.to_datetime(df["date"])
        SIGNALS = df.set_index(["date", "Ticker"])
        print(f"Loaded {len(SIGNALS)} signals for ZipLime")
    except Exception as e:
        print("Could not load trading_journal.csv:", e)
        SIGNALS = pd.DataFrame()

    context.tickers = ["AAPL", "WTI"]


async def handle_data(context, data):
    if SIGNALS is None or SIGNALS.empty:
        return

    current_dt = context.get_datetime()
    if current_dt.tzinfo is not None:
        current_date = current_dt.astimezone(pytz.UTC).replace(tzinfo=None)
    else:
        current_date = current_dt

    current_date = pd.Timestamp(current_date).normalize()

    for ticker in context.tickers:
        try:
            asset = data.asset_finder.lookup_symbol(ticker, as_of_date=None)
        except Exception:
            continue

        try:
            signal_row = SIGNALS.loc[(current_date, ticker)]
            if isinstance(signal_row, pd.DataFrame):
                signal_row = signal_row.iloc[-1]

            signal = signal_row.get("FINAL_SIGNAL", "HOLD")

            if signal in ["BUY", "STRONG_BUY"]:
                order_target_percent(asset, 0.5)
            elif signal in ["SELL", "STRONG_SELL"]:
                order_target_percent(asset, 0.0)
        except KeyError:
            pass

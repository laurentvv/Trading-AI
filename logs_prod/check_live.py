import yfinance as yf

print("=== LIVE MARKET DATA ===")
for ticker in ["CRUDP.PA", "SXRV.DE", "CL=F", "^NDX"]:
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if not hist.empty:
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last
        change_pct = ((last["Close"] - prev["Close"]) / prev["Close"]) * 100
        date_str = last.name.strftime("%Y-%m-%d")
        price = last["Close"]
        print(f"  {ticker:>10}: {price:>10.2f} ({change_pct:+.2f}%) @ {date_str}")
    else:
        print(f"  {ticker:>10}: NO DATA")

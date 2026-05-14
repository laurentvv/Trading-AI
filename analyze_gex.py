import pandas as pd
import yfinance as yf
import io
import requests

def fetch_gex_data():
    url = "https://squeezemetrics.com/monitor/static/DIX.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def fetch_nasdaq_data(start_date):
    nasdaq = yf.download("^IXIC", start=start_date, progress=False)
    if isinstance(nasdaq.columns, pd.MultiIndex):
        nasdaq.columns = nasdaq.columns.droplevel(1)
    nasdaq.index = pd.to_datetime(nasdaq.index)
    return nasdaq

def analyze_gex():
    print("Fetching SqueezeMetrics GEX Data...")
    gex_df = fetch_gex_data()

    start_date = gex_df.index.min().strftime('%Y-%m-%d')
    print(f"Fetching NASDAQ Data since {start_date}...")
    nasdaq_df = fetch_nasdaq_data(start_date)

    # Merge datasets
    df = gex_df.join(nasdaq_df['Close'], how='inner').rename(columns={'Close': 'nasdaq_close'})

    # Calculate Forward Returns for NASDAQ
    df['fwd_1d_ret'] = df['nasdaq_close'].pct_change().shift(-1)
    df['fwd_5d_ret'] = df['nasdaq_close'].pct_change(5).shift(-5)
    df['fwd_20d_ret'] = df['nasdaq_close'].pct_change(20).shift(-20)

    df['gex_zscore'] = (df['gex'] - df['gex'].rolling(252).mean()) / df['gex'].rolling(252).std()

    # Divide into GEX regimes: Positive vs Negative
    gex_pos = df[df['gex'] > 0]
    gex_neg = df[df['gex'] <= 0]

    report_lines = []
    report_lines.append("# GEX (Gamma Exposure) Utility Analysis for NASDAQ Trading\n")
    report_lines.append("## Overview\n")
    report_lines.append("This analysis explores the relationship between the S&P 500 Gamma Exposure (GEX), as provided by SqueezeMetrics, and the forward returns of the NASDAQ (^IXIC).\n")

    report_lines.append("## Regime Analysis: Positive vs Negative GEX\n")

    report_lines.append(f"**Total Days Analyzed:** {len(df)}\n")
    report_lines.append(f"**Days with Positive GEX:** {len(gex_pos)} ({len(gex_pos)/len(df)*100:.1f}%)\n")
    report_lines.append(f"**Days with Negative GEX:** {len(gex_neg)} ({len(gex_neg)/len(df)*100:.1f}%)\n")

    report_lines.append("\n### Forward 1-Day Returns (NASDAQ)\n")
    report_lines.append(f"- **Overall Avg:** {df['fwd_1d_ret'].mean()*100:.3f}% (Volatility: {df['fwd_1d_ret'].std()*100:.3f}%)\n")
    report_lines.append(f"- **Pos GEX Avg:** {gex_pos['fwd_1d_ret'].mean()*100:.3f}% (Volatility: {gex_pos['fwd_1d_ret'].std()*100:.3f}%)\n")
    report_lines.append(f"- **Neg GEX Avg:** {gex_neg['fwd_1d_ret'].mean()*100:.3f}% (Volatility: {gex_neg['fwd_1d_ret'].std()*100:.3f}%)\n")

    report_lines.append("\n### Forward 5-Day Returns (NASDAQ)\n")
    report_lines.append(f"- **Overall Avg:** {df['fwd_5d_ret'].mean()*100:.3f}% (Volatility: {df['fwd_5d_ret'].std()*100:.3f}%)\n")
    report_lines.append(f"- **Pos GEX Avg:** {gex_pos['fwd_5d_ret'].mean()*100:.3f}% (Volatility: {gex_pos['fwd_5d_ret'].std()*100:.3f}%)\n")
    report_lines.append(f"- **Neg GEX Avg:** {gex_neg['fwd_5d_ret'].mean()*100:.3f}% (Volatility: {gex_neg['fwd_5d_ret'].std()*100:.3f}%)\n")

    report_lines.append("\n### Forward 20-Day Returns (NASDAQ)\n")
    report_lines.append(f"- **Overall Avg:** {df['fwd_20d_ret'].mean()*100:.3f}% (Volatility: {df['fwd_20d_ret'].std()*100:.3f}%)\n")
    report_lines.append(f"- **Pos GEX Avg:** {gex_pos['fwd_20d_ret'].mean()*100:.3f}% (Volatility: {gex_pos['fwd_20d_ret'].std()*100:.3f}%)\n")
    report_lines.append(f"- **Neg GEX Avg:** {gex_neg['fwd_20d_ret'].mean()*100:.3f}% (Volatility: {gex_neg['fwd_20d_ret'].std()*100:.3f}%)\n")

    report_lines.append("\n## Correlation\n")
    corr = df[['gex', 'fwd_1d_ret', 'fwd_5d_ret', 'fwd_20d_ret']].corr()
    report_lines.append("Correlation between Raw GEX and Forward Returns:\n")
    report_lines.append(f"- 1-Day: {corr.loc['gex', 'fwd_1d_ret']:.3f}\n")
    report_lines.append(f"- 5-Day: {corr.loc['gex', 'fwd_5d_ret']:.3f}\n")
    report_lines.append(f"- 20-Day: {corr.loc['gex', 'fwd_20d_ret']:.3f}\n")

    report_lines.append("\n## Conclusion\n")
    report_lines.append("Generally, **Negative GEX** environments are associated with higher volatility (wider distribution of returns) and often higher but riskier short-term forward returns, as market makers trade with the trend. **Positive GEX** environments are usually associated with lower volatility, as market makers hedge by trading against the trend, suppressing movement.\n")
    report_lines.append("\n**Recommendation for the Decision Engine:**\n")
    report_lines.append("If integrated, GEX is primarily a **Volatility Filter / Regime Indicator** rather than a pure directional signal. In a Negative GEX regime, the risk engine might need to reduce position sizing or demand higher confidence from other models due to increased volatility.\n")

    with open("GEX_ANALYSIS.md", "w") as f:
        f.writelines(report_lines)

    print("GEX_ANALYSIS.md generated successfully.")

if __name__ == '__main__':
    analyze_gex()

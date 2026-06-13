import pandas as pd
import sys

# We need to forward fill missing dates to please zipline
for ticker in ['CL=F', 'QQQ']:
    df = pd.read_csv(f'csvdir/daily/{ticker}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Missing dates
    missing_dates = pd.to_datetime(['2016-10-10', '2016-11-11'])
    for d in missing_dates:
        if d not in df.index:
            # Duplicate the previous day's row
            prev_d = df.index[df.index < d][-1]
            df.loc[d] = df.loc[prev_d]
            # Zero volume to avoid fake spikes
            df.loc[d, 'volume'] = 0

    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    df.to_csv(f'csvdir/daily/{ticker}.csv', index=False)
print("Padded missing dates for US tickers")

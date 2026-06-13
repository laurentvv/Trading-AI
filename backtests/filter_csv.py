import pandas as pd
import sys

# We need to forward fill missing dates to please zipline
for ticker in ['CL=F', 'QQQ', 'CRUDP_PA', 'CRUDP.PA']:
    try:
        df = pd.read_csv(f'csvdir/daily/{ticker}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        df = df[df.index != pd.Timestamp('2019-12-25')]

        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
        df.to_csv(f'csvdir/daily/{ticker}.csv', index=False)
    except FileNotFoundError:
        pass

print("Filtered all")

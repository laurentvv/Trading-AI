import pandas as pd
from zipline.api import order, record, symbol
from zipline import run_algorithm


def initialize(context):
    context.asset = symbol("AAPL")


def handle_data(context, data):
    order(context.asset, 10)
    record(AAPL=data.current(context.asset, "price"))


# exchange-calendars 4.x wants naive timestamps!
start = pd.Timestamp("2020-01-02")
end = pd.Timestamp("2022-12-30")

result = run_algorithm(
    start=start, end=end, initialize=initialize, capital_base=10000, handle_data=handle_data, bundle="my-bundle-US"
)
print(result.head())

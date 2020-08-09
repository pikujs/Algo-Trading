## Imports
import zipline
from zipline.api import order_target_percent, record, symbol, set_benchmark, get_open_orders
from datetime import datetime
import pytz
import pyfolio as pf
import numpy as np
import pandas as pd


## Backtesting

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')
    set_benchmark(symbol('AAPL'))

def handle_data(context, data):
    # Skip first 200 days to get full windows
    context.i += 1
    if context.i < 200:
         return
    # Compute averages
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    short_mavg = data.history(context.asset, 'price', bar_count=50, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=200, frequency="1d").mean()

    # Trading logic
    open_orders = get_open_orders()
    
    if context.asset not in open_orders:
        if short_mavg > long_mavg:
            # order_target orders as many shares as needed to
            # achieve the desired number of shares.
            order_target_percent(context.asset, 1.0)
        elif short_mavg < long_mavg:
            order_target_percent(context.asset, 0.0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
           short_mavg=short_mavg,
           long_mavg=long_mavg)

start = datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc)

perf = zipline.run_algorithm(start=start,
                      end=end,
                      initialize=initialize,
                      capital_base=10000,
                      handle_data=handle_data)



# Extract algo returns and benchmark returns
returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)
benchmark_period_return = perf['benchmark_period_return']

# Convert benchmark returns to daily returns
#daily_returns = (1 + benchmark_period_return) / (1 + benchmark_period_return.shift()) - 1
daily_benchmark_returns = np.exp(np.log(benchmark_period_return + 1.0).diff()) - 1

# Create tear sheet
pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions, benchmark_rets=daily_benchmark_returns)

## Daily returns from cummulative returns


# We need to be able to calulate the daily returns from the cumulative returns
daily_returns = pd.Series([0.5, -0.5, 0.5, -0.5])
cumulative_returns = pd.Series([0.5, -0.25, 0.125, 0.5625])

# Two different formulas to calculate daily returns
print((1 + cumulative_returns) / (1 + cumulative_returns.shift()) -1)
print((np.exp(np.log(cumulative_returns + 1).diff()) - 1))

# Recreate daily returns manually for example purposes
print(daily_returns.head(1))
print((1 - 0.25) / (1.5) - 1)
print((1 + 0.125) / (1 - 0.25) - 1)
print((1 + 0.5625) / (1 + 0.125 ) - 1)
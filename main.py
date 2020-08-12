## Imports
from utils import data_fetch
import indicators

## Preprocessing

data = data_fetch.finnhub_hist("AAPL")
print(data)


print("Done")

## Models
# Compute the Bollinger Bands for NIFTY using the 50-day Moving average
n = 50
NIFTY_BBANDS = BBANDS(data, n)
print(NIFTY_BBANDS)



## Output

# Create the plot
pd.concat([NIFTY_BBANDS.Close,NIFTY_BBANDS.UpperBB,NIFTY_BBANDS.LowerBB],axis=1).plot(figsize=(9,5),grid=True)
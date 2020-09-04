## Imports
import pandas as pd
import numpy as np
import talib

from pyalgotrade import dataseries
from pyalgotrade import technical

## Indicators

# Compute the Bollinger Bands 
def bolinger_scratch(data, window=n, std_n=2, heiken_ashi=True):
    MA = data.Close.rolling(window=n).mean()
    SD = data.Close.rolling(window=n).std()
    new_data = pd.DataFrame()
    if heiken_ashi:
        heiken_ashi(data)
    else:
        new_data['UpperBB'] = MA + (std_n * SD) 
        new_data['LowerBB'] = MA - (std_n * SD)
    return new_data


def bolinger_talib(data, heiken_ashi=True):
    return talib.BBANDS(data["close"], matype=MA_Type.T3)

def heiken_ashi(data):
    return 



# An EventWindow is responsible for making calculations using a window of values.
class Accumulator(technical.EventWindow):
    def getValue(self):
        ret = None
        if self.windowFull():
            ret = self.getValues().sum()
        return ret

# # Build a sequence based DataSeries.
# seqDS = dataseries.SequenceDataSeries()
# # Wrap it with a filter that will get fed as new values get added to the underlying DataSeries.
# accum = technical.EventBasedFilter(seqDS, Accumulator(3))

# # Put in some values.
# for i in range(0, 50):
#     seqDS.append(i)

# # Get some values.
# print(accum[0])  # Not enough values yet.
# print(accum[1])  # Not enough values yet.
# print(accum[2])  # Ok, now we should have at least 3 values.
# print(accum[3])

# # Get the last value, which should be equal to 49 + 48 + 47.
# print(accum[-1])
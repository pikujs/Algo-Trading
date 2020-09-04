from datetime import datetime
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
#import indicators
from db import timscale_setup
from db import dbscrape
import ta
from backtesting import Backtest
#from backtesting.test import SMA
from backtesting import Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy


def prepareData(data, verbose=False):
    datetime_index = pd.DatetimeIndex(pd.to_datetime(data["datetime"]).values)
    data_backtesting = data.set_index(datetime_index)
    #data_backtesting.drop(["datetime", "internalname", "unknown", "expirydate", "exchange"], axis=1, inplace=True)
    data_backtesting.drop(["datetime", "internalname", "unknown", "expirydate", "exchange"], axis=1, inplace=True)
    data_backtesting.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace = True)
    if verbose:
        data_backtesting.head()
    return data_backtesting

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 13
    n2 = 32
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()

class SmaCrossSignalTrail(SignalStrategy,
               TrailingStrategy):
    n1 = 10
    n2 = 25
    
    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        
        # Precompute the two moving averages
        sma1 = self.I(SMA, self.data.Close, self.n1)
        sma2 = self.I(SMA, self.data.Close, self.n2)
        
        # Where sma1 crosses sma2 upwards. Diff gives us [-1,0, *1*]
        signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)  # Upwards/long only
        
        # Use 95% of available liquidity (at the time) on each order.
        # (Leaving a value of 1. would instead buy a single share.)
        entry_size = signal * .95
                
        # Set order entry sizes using the method provided by 
        # `SignalStrategy`. See the docs.
        self.set_signal(entry_size=entry_size)
        
        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
        self.set_trailing_sl(2)


class BollingerStrat(Strategy):
    
    n1 = 20
    n2 = 20
    n3 = 14
    n_dev = 2
    
    # def hband_last(self, data, n, n_dev):
    #     return ta.volatility.bollinger_hband_indicator(data, n, n_dev)[0]
    # def lband_last(self, data, n, n_dev):
    #     return ta.volatility.bollinger_lband(data, n, n_dev)[0]
    # def ema_last(self, data, n2):
    #     return ta.volatility.bollinger_mavg(data, n2)[0]
    # def vwap_Last(self, data, high, low, close, volume, n3):
    #     return ta.volume.volume_weighted_average_price(data, high, low, close, volume, n3)[0]

    def init(self):
        # Precompute the two moving averages
        self.bollinger_hband_indicator = self.I(ta.volatility.bollinger_hband_indicator, pd.Series(self.data.Close), self.n1, self.n_dev) #bollinger_hband
        self.bollinger_lband = self.I(ta.volatility.bollinger_lband, pd.Series(self.data.Close), self.n1, self.n_dev) #bollinger_lband
        self.ema = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), self.n2) # EMA
        self.vwap = self.I(ta.volume.volume_weighted_average_price, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), pd.Series(self.data.Volume), self.n3)

    
    # def stddev(self, data, n):
    #     SD = data.Close.rolling(window=n).std()
    
    # def bollinger_high(self, data, n=20, std_n=2):
    #     MA = data.rolling(window=n).mean()
    #     SD = data.rolling(window=n).std()
    #     return MA + (std_n * SD)
    # def bollinger_high(self, data, n=20, std_n=2):
    #     MA = data.rolling(window=n).mean()
    #     SD = data.rolling(window=n).std()
    #     return MA - (std_n * SD)

    def next(self):
        #if self.bollinger_lband > self.vwap:
        if crossover( self.data.Close, self.ema):
            print("Buying")
            self.buy()
        
        if self.bollinger_hband_indicator:
            print("Position Close")
            self.position.close()
        # if crossover(self.sma1, self.sma2):
        #     self.position.close()
        #     self.buy()

        # # Else, if sma1 crosses below sma2, close any existing
        # # long trades, and sell the asset
        # elif crossover(self.sma2, self.sma1):
        #     self.position.close()
        #     self.sell()

def run_backtest(data, strategy, cash=100000, commission=0.002, verbose=False):
    bt = Backtest(data, strategy, cash=cash, commission=commission)
    stats = bt.run()
    if verbose:
        print(stats)
    return bt, stats

def plot(bt, name="", results=None):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fname = "charts/" + bt._strategy.__name__ + "_" + name + "_" + timestr + ".html"
    bt.plot(filename=fname)
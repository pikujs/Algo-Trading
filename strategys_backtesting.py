import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
#import indicators
from db import timscale_setup
from db import dbscrape_old as dbscrape
# from db import minuteto
import ta
import datetime as dt
from backtesting import Backtest
#from backtesting.test import SMA
from backtesting import Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy


def prepareData(data, verbose=False):
    datetime_index = pd.DatetimeIndex(pd.to_datetime(data["datetime"]).values, name="datetime")
    data_backtesting = data.drop(["datetime", "internalname", "unknown", "expirydate", "exchange"], axis=1)
    data_backtesting.set_index(datetime_index, inplace=True)
    data_backtesting.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace = True)
    if verbose:
        print(data_backtesting.head())
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

class RSIStrat(Strategy):
    
    n1 = 20
    n2 = 20
    n3 = 14
    n4 = 20
    n_dev = 2

    def init(self):
        # Precompute the two moving averages
        self.bollinger_hband_indicator = self.I(ta.volatility.bollinger_hband_indicator, pd.Series(self.data.Close), self.n1, self.n_dev) #bollinger_hband
        self.bollinger_lband = self.I(ta.volatility.bollinger_lband, pd.Series(self.data.Close), self.n1, self.n_dev) #bollinger_lband
        self.bollinger_hband = self.I(ta.volatility.bollinger_hband, pd.Series(self.data.Close), self.n1, self.n_dev) #bollinger_lband
        self.ema = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), self.n2) # EMA
        self.vwap = self.I(ta.volume.volume_weighted_average_price, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), pd.Series(self.data.Volume), self.n3)
        self.rsi = self.I(ta.momentum.rsi, pd.Series(self.data.Close), self.n4)
        self.redline = False
        self.blueline = False

    def next(self):

        if self.data.Close < self.bollinger_lband and not self.position:
                self.redline = True

        if self.data.Close > self.bollinger_hband and self.position:
                self.blueline = True

        if self.data.Close > self.ema and not self.position and self.redline:        	
                # BUY, BUY, BUY!!! (with all possible default parameters)
                print('BUY CREATE, %.2f' % self.data.Close)
                # Keep track of the created order to avoid a 2nd order
                self.buy()

        if self.data.Close > self.bollinger_hband and not self.position:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                print('BUY CREATE, %.2f' % self.data.Close)
                # Keep track of the created order to avoid a 2nd order
                self.buy()

        if self.data.Close < self.ema and self.position and self.blueline:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                print('SELL CREATE, %.2f' % self.data.Close)
                self.blueline = False
                self.redline = False
                # Keep track of the created order to avoid a 2nd order
                self.position.close()
 
class SnRfollowup(Strategy):
    
    n1 = 20
    n2 = 10
    n3 = 14
    n4 = 20
    n_dev = 2
    entry_stopLoss = 30
    supportRange = 25
    resistanceRange = 25
    exitRange = 100
    support_down_counter = 0
    resistance_up_counter = 0
    support_down_counter = 0
    init_support_range = 100
    def init(self):
        # Precompute the two moving averages
        self.bollinger_mavg = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), self.n1) #bollinger_hband
        self.bollinger_lband = self.I(ta.volatility.bollinger_lband, pd.Series(self.data.Close), self.n1, self.n_dev) #bollinger_lband
        self.bollinger_hband = self.I(ta.volatility.bollinger_hband, pd.Series(self.data.Close), self.n1, self.n_dev) #bollinger_lband
        self.ema_10 = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), self.n2) # EMA
        self.vwap = self.I(ta.volume.volume_weighted_average_price, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), pd.Series(self.data.Volume), self.n3)
        self.rsi = self.I(ta.momentum.rsi, pd.Series(self.data.Close), self.n4)
        self.redLine = False
        self.blueLine = False
        self.supportLine = []
        self.resistanceLine = []
        self.fin_state = "init"
        self.stopLoss = 0
        # self.daily_data = minuteto.minutetoyear(self.data.df.index[0].month, self.data.df.index[-1].month, self.data.df.index[0].year, self.data.df.index[-1].year)

    def sellCheck(self):
        if not self.position:
            if self.data.High > self.bollinger_mavg:
                return True
        return False
    
    def buyCheck(self):
        if not self.position:
            if self.data.Low < self.bollinger_mavg:
                return True
        return False

    def plot_all(self):
        fig, ax = plt.subplots()

        candlestick_ohlc(ax,df.values,width=0.6, \
                        colorup='green', colordown='red', alpha=0.8)

        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        fig.tight_layout()

        for level in levels:
            plt.hlines(level[1],xmin=df['Date'][level[0]],\
                    xmax=max(df['Date']),colors='blue')
        fig.show()

    def getfractalSupRes(self, timeframe=750): ## timeframe=750 for 2 days
        levels = []
        if len(self.data) < timeframe:
            return levels
        candle_mean =  np.mean(self.data.High[-1*timeframe:] - self.data.Low[-1*timeframe:])
        for j in range(timeframe-2): ## 2 datapoint padding
            i = - j - 1 ## set the right iterator for direction (Currently backwards)
            if self.data.Low[i] < self.data.Low[i-1] \
                    and self.data.Low[i] < self.data.Low[i+1] \
                    and self.data.Low[i+1] < self.data.Low[i+2] \
                    and self.data.Low[i-1] < self.data.Low[i-2]:
                if np.sum([abs(self.data.Low[i]-x) < candle_mean  for x in levels]) == 0: ## Proximity Check
                    levels.append(self.data.Low[i]) ## Support Check
            if self.data.High[i] > self.data.High[i-1] \
                    and self.data.High[i] > self.data.High[i+1] \
                    and self.data.High[i+1] > self.data.High[i+2] \
                    and self.data.High[i-1] > self.data.High[i-2]:
                if np.sum([abs(self.data.High[i]-x) < candle_mean  for x in levels]) == 0: ## Proximity Check
                    levels.append((i,self.data.High[i])) ## Resisitance Check
        return levels

    def prevdayPrices(self):
        priceList = []
        current_date = self.data.index[-1].date()
        print(str(current_date))
        print(str(current_date - dt.timedelta(days=1)))
        day_mask = self.data.df.index == (current_date - dt.timedelta(days=1))
        prev_data = self.data.df.loc[day_mask]
        # prev_data = self.data.df.index.apply(lambda x: x.date() == (current_date - dt.timedelta(days=1)))
        if prev_data.size == 0:
            print("Previous day data not available: " + str(current_date))
            priceList.append(self.data.Open[-1] + self.init_support_range)
            priceList.append(self.data.Open[-1] - self.init_support_range)
        print(prev_data.head())
        return priceList

## SUP/RES Lines:
## OHLC of weekly data/previous day/expiry cycle
## recent data more weightage
## all prev same day of weeks data more weightage (Wed+Thurs)(Mon+Fri)

    def next(self):
        self.datetime = self.data.index
        ## POPULATE SUPPORT AND RESISTANCES
        self.supportLine = []
        self.resistanceLine = []
        if self.vwap > self.data.Close[-1]:
            self.resistanceLine.append(self.vwap)
        else:
            self.supportLine.append(self.vwap)

        prev_day_prices = self.prevdayPrices()
        for price in prev_day_prices:
            if price > self.data.Close[-1]:
                self.resistanceLine.append(price)
            else:
                self.supportLine.append(price)

        # last_day_close = self.daily_data[self.datetime.date - dt.timedelta(days=1)]["close"]

        # if last_day_close > self.data.Close[-1]:
        #     self.resistanceLine.append(last_day_close)
        # else:
        #     self.supportLine.append(last_day_close)

        ## CHECK EMPTY SUPPORT/RESISTANCE
        if not len(self.supportLine):
            self.supportLine.append(0)
        if not len(self.resistanceLine):
            self.resistanceLine.append(0)

        averagePrice = (self.data.Open + self.data.Close[-1])/2
        
        ## STOP LOSS
        if self.position.is_long:
            if self.data.Close[-1] < self.stopLoss:
                self.position.close()
        if self.position.is_short:
            if self.data.Close[-1] > self.stopLoss:
                self.position.close()

        ## EXIT STRATEGY
        if self.position:
            if self.position.pl > self.exitRange:
                self.position.close()

        ## RESISTANCE LINE STRATEGY
        if abs(self.data.High - self.resistanceLine[-1]) < self.resistanceRange:
            self.fin_state = "resistanceLine"
        if self.data.Close[-1] < self.data.Close[-2]:
            self.resistance_up_counter = 0

        if self.fin_state is "resistanceLine":
            if (self.ema_10 - self.ema_10[-1]) < 2: ## momentum based reversal
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
            if self.data.Close[-1] > self.data.Close[-2]: ## Count based breakout
                self.resistance_up_counter = self.resistance_up_counter + 1
            if self.resistance_up_counter >= 3:
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
            if self.data.High > (self.resistanceLine[-1] + self.resistanceRange): ## price based breakout
                self.supportLine.append(self.resistanceLine[-1])
                del self.resistanceLine[-1]
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                # todo: Change new support line

        ## SUPPORT LINE STRATEGY
        if abs(self.data.Low - self.supportLine[-1]) < self.supportRange:
            self.fin_state = "supportLine"
        if self.data.Close[-1] > self.data.Close[-2]:
            self.support_down_counter = 0

        if self.fin_state is "supportLine":
            if (self.ema_10 - self.ema_10[-1]) > -2: ## momentum based reversal
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
            if self.data.Close[-1] < self.data.Close[-2]: ## Count based breakout
                self.support_down_counter = self.support_down_counter + 1
            if self.support_down_counter >= 3:
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
            if self.data.Low < (self.supportLine[-1] - self.supportRange): ## price based breakout
                self.resistanceLine.append(self.supportLine[-1])
                del self.supportLine[-1]
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                # todo: Change new support line

class dl3timeframeStrat(Strategy):
    def init(self):
        pass
    def next(self):
        pass

def run_backtest(data, strategy, cash=1000000, commission=0.002, verbose=False):
    bt = Backtest(data, strategy, cash=cash, commission=commission)
    stats = bt.run()
    if verbose:
        print(stats)
    return bt, stats

def plot(bt, name="", results=None):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fname = "charts/" + bt._strategy.__name__ + "_" + name + "_" + timestr + ".html"
    bt.plot(filename=fname)

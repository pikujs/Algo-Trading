import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import indicators
from db import timscale_setup
from db import dbscrape
from tradeUtils import roughsr2
# from db import minuteto
import ta
import datetime as dt
import logging
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy
import tensorflow.keras as keras


def prepareData(data, heiken_ashi=False, verbose=False):
    datetime_index = pd.DatetimeIndex(pd.to_datetime(data["datetime"]).values, name="datetime")
    data_backtesting = data.drop(["datetime", "internalname", "unknown", "expirydate", "exchange"], axis=1)
    data_backtesting.set_index(datetime_index, inplace=True)
    data_backtesting.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace = True)
    # data_backtesting.drop(data_backtesting.index.loc[x.h]inplace=True)
    if heiken_ashi:
        data_backtesting = indicators.heiken_ashi(data_backtesting)
    if verbose:
        print(data_backtesting.head())
    return data_backtesting

class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 13
    n2 = 32
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), self.n1)
        self.sma2 = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), self.n2)
    
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
        sma1 = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), self.n1)
        sma2 = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), self.n2)
        
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
    exitRange = 80
    exit_rev = 60
    exitPrice = 0
    support_down_counter = 0
    resistance_up_counter = 0
    support_down_counter = 0
    init_support_range = 100
    rev_candleRange = 30
    levelPriceRange = 60
    bounce_candle_ratio = 0.5
    bounce_candle_len = 70
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
        self.fin_state = "searching"
        self.stopLoss = 0
        # self.daily_data = minuteto.minutetoyear(self.data.df.index[0].month, self.data.df.index[-1].month, self.data.df.index[0].year, self.data.df.index[-1].year)

    def sellCheck(self):
        if self.position.is_long:
            logging.info(str(self.data.index[-1]) + "- Closed long position at " + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl))
            self.position.close()
        if not self.position:
            # if self.data.High > self.bollinger_mavg:
            return True
        return False
    
    def buyCheck(self):
        if self.position.is_short:
            logging.info(str(self.data.index[-1]) + "- Closed short position at " + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl))
            self.position.close()
        if not self.position:
            # if self.data.Low < self.bollinger_mavg:
            return True
        return False

    def getfractalSupRes(self, timeframe=750): ## timeframe=750 for 2 days ##
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
                    levels.append((i, self.data.Low[i])) ## Support Check
            if self.data.High[i] > self.data.High[i-1] \
                    and self.data.High[i] > self.data.High[i+1] \
                    and self.data.High[i+1] > self.data.High[i+2] \
                    and self.data.High[i-1] > self.data.High[i-2]:
                if np.sum([abs(self.data.High[i]-x) < candle_mean  for x in levels]) == 0: ## Proximity Check
                    levels.append((i,self.data.High[i])) ## Resisitance Check
        return levels

    def createLevels(self, timeframe=250):
        levels = []
        if len(self.data) < timeframe:
            return levels
        levels.append((0, max(self.data.High[-timeframe:]))) # timeframe High
        levels.append((0, min(self.data.Low[-timeframe:]))) # timeframe Low

        df = pd.DataFrame()
        df_index = pd.DatetimeIndex(pd.to_datetime(self.data.index[-timeframe:]).values, name="datetime")
        df["Open"] = self.data.Open[-timeframe:]
        df["High"] = self.data.High[-timeframe:]
        df["Low"] = self.data.Low[-timeframe:]
        df["Close"] = self.data.Close[-timeframe:]
        df["Volume"] = self.data.Volume[-timeframe:]
        df.set_index(df_index, inplace=True)

        # df.resample('5m',  how='ohlc', axis=0, fill_method='bfill')
        candle_len=20
        candle_mean = np.mean(df["High"] - df["Low"])
        for i in range(timeframe-candle_len):
            thisMax = max(self.data.High[-i-candle_len-1:-i-1])
            thisMin = min(self.data.Low[-i-candle_len-1:-i-1])
            if thisMax - thisMin < self.levelPriceRange:
                thisPrice = (thisMax+thisMin)/2
                if np.sum([abs(thisPrice-x) < candle_mean  for x in levels]) == 0:
                    levels.append((-i-5, thisPrice))
        # # Plot levels
        # indexes, lvls = zip(*levels)
        # mpf.plot(df, hlines=lvls, type="candle")

        return levels
        # highs in 30 point range level
        # lows in 30 point range level

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
            logging.info("Previous day data not available: " + str(current_date))
            priceList.append(self.data.Open[-1] + self.init_support_range)
            priceList.append(self.data.Open[-1] - self.init_support_range)
        print(prev_data.head())
        return priceList

    def next(self):
        if len(self.data.Close) < 251:
            logging.info("close len is " + str(len(self.data.Close)))
            # print("close len is " + str(len(self.data.Close)))
            return
        ## DO NOT TRADE first and last 30 mins of day
        tradeStartTime = dt.time(hour=10, minute=0)
        tradeEndTime = dt.time(hour=15, minute=0)
        if not (tradeStartTime < self.data.index[-1].time() < tradeEndTime):
            logging.info("Outside Trading hours, Time=" + str(self.data.index[-1]))
            if self.position:
                self.position.close()
                logging.info(str(self.data.index[-1]) + "- Closed Position after trading hours at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
            return
        ## POPULATE SUPPORT AND RESISTANCES
        inRange = False
        # for l in (self.supportLine + self.resistanceLine):
        #     if (self.data.Close[-1] - l) < self.supportRange:
        #         inRange = True
        # if not inRange:
        #     self.fin_state = "searching"
        if self.fin_state == "searching":
            self.supportLine = []
            self.resistanceLine = []
            levels = self.createLevels(timeframe=250)
            # print(levels)
            for level in levels:
                pos, l  = level
                if l > self.data.Close[-1]:
                    self.resistanceLine.append(l)
                else:
                    self.supportLine.append(l)


        # if self.vwap > self.data.Close[-1]:
        #     self.resistanceLine.append(self.vwap)
        # else:
        #     self.supportLine.append(self.vwap)        
        # prev_day_prices = self.prevdayPrices()
        # for price in prev_day_prices:
        #     if price > self.data.Close[-1]:
        #         self.resistanceLine.append(price)
        #     else:
        #         self.supportLine.append(price)
        
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

        averagePrice = (self.data.Open[-1] + self.data.Close[-1])/2
        
        ## trailing Constant Stop Loss
        if self.position.is_long:
            if self.data.Close[-1] < self.stopLoss:
                logging.info(str(self.data.index[-1]) + "- Closed long position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                # print(str(self.data.index[-1]) + "- Closed long position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                self.position.close()
            else:
                const_stoploss = self.data.Close[-1] - self.entry_stopLoss
                if const_stoploss > self.stopLoss:
                    self.stopLoss = const_stoploss
        if self.position.is_short:
            if self.data.Close[-1] > self.stopLoss:
                logging.info(str(self.data.index[-1]) + "- Closed short position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                # print(str(self.data.index[-1]) + "- Closed short position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                self.position.close()
            else:
                const_stoploss = self.data.Close[-1] + self.entry_stopLoss
                if const_stoploss < self.stopLoss:
                    self.stopLoss = const_stoploss

        """  ## STOP LOSS
        if self.position.is_long:
            if self.data.Close[-1] < self.stopLoss:
                logging.info(str(self.data.index[-1]) + " Closed Position at Price=" + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl) + ", StopLoss=" + str(self.stopLoss))
                self.position.close()
        if self.position.is_short:
            if self.data.Close[-1] > self.stopLoss:
                logging.info(str(self.data.index[-1]) + " Closed Position at Price=" + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl) + ", StopLoss=" + str(self.stopLoss))
                self.position.close() 
 """
        ## EXIT STRATEGY
        if self.position.is_long:
            if self.data.Close[-1] > self.exitPrice:
                logging.info(str(self.data.index[-1]) + " Exited Position at Price=" + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl))
                self.position.close()
        if self.position.is_short:
            if self.data.Close[-1] < self.exitPrice:
                logging.info(str(self.data.index[-1]) + " Exited Position at Price=" + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl))
                self.position.close()

        ## RESISTANCE LINE STRATEGY
        if abs(self.data.Close[-1] - self.resistanceLine[-1]) < self.resistanceRange:
            self.fin_state = "resistanceLine"
        elif self.fin_state is not "supportLine":
            self.fin_state = "searching"
        if self.data.Close[-1] < self.data.Close[-2]:
            self.resistance_up_counter = 0

        if self.fin_state is "resistanceLine":
            logging.info(str(self.data.index[-1]) + "- Price at resisitance line=" + str(self.resistanceLine[-1]))
            logging.info(str(self.resistanceLine) + str(self.supportLine))
            if ((self.data.Open[-1] - self.data.Close[-1]) > self.rev_candleRange):
            # if (self.ema_10 - self.ema_10[-1]) < 2: ## momentum based reversal #todo Change this method
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] - self.exit_rev
                    logging.info("Short order(Big Rev Candle) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            if (self.data.Open[-1] < self.data.Close[-1]) and \
                ((self.data.High[-1]-self.data.Close[-1])/(self.data.High[-1]-self.data.Open[-1])) > self.bounce_candle_ratio and \
                (self.data.High[-1]-self.data.Low[-1]) > self.bounce_candle_len:
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] - self.exit_rev
                    logging.info("Short order(Bounce Reversal) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            """ if self.data.Close[-1] > self.data.Close[-2]: ## Count based breakout
                self.resistance_up_counter = self.resistance_up_counter + 1
                logging.info("res_up_count="+str(self.resistance_up_counter))
            if self.resistance_up_counter >= 3:
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] + self.exitRange
                    logging.info("Long order(Count Breakout) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return """
            if self.data.Close[-1] > (self.resistanceLine[-1] + self.resistanceRange): ## price based breakout
                self.supportLine.append(self.resistanceLine[-1])
                del self.resistanceLine[-1]
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] + self.exitRange
                    logging.info("Long order(Price Breakout) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return

        ## SUPPORT LINE STRATEGY
        if abs(self.data.Close[-1] - self.supportLine[-1]) < self.supportRange:
            self.fin_state = "supportLine"
        elif self.fin_state is not "resistanceLine":
            self.fin_state = "searching"
        if self.data.Close[-1] > self.data.Close[-2]:
            self.support_down_counter = 0
        

        if self.fin_state is "supportLine": ## Candle range reversal
            logging.info(str(self.data.index[-1]) + "- Price at support line=" + str(self.supportLine[-1]))
            logging.info(str(self.resistanceLine) + str(self.supportLine))
            if ((self.data.Close[-1] - self.data.Open[-1]) > self.rev_candleRange): # Big Rev Candle
                    # or (self.ema_10 - self.ema_10[-1]) > -2: ## momentum based reversal
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] + self.exit_rev
                    logging.info("Long order(Big Rev Candle) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            if (self.data.Close[-1] < self.data.Open[-1]) and \
                ((self.data.Close[-1]-self.data.Low[-1])/(self.data.Open[-1]-self.data.Low[-1])) > self.bounce_candle_ratio and \
                (self.data.High[-1]-self.data.Low[-1]) > self.bounce_candle_len:
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] + self.exit_rev
                    logging.info("Long order(Bounce Reversal) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            """ if self.data.Close[-1] < self.data.Close[-2]: ## Count based breakout
                self.support_down_counter = self.support_down_counter + 1
                logging.info("sup_down_count="+str(self.support_down_counter))
            if self.support_down_counter >= 3:
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] - self.exitRange
                    logging.info("Short order(Count Breakout) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return """
            if self.data.Close[-1] < (self.supportLine[-1] - self.supportRange): ## price based breakout
                self.resistanceLine.append(self.supportLine[-1])
                del self.supportLine[-1]
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] - self.exitRange
                    logging.info("Short order(Price Breakout) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return

## SUP/RES Lines:
## OHLC of weekly data/previous day/expiry cycle
## recent data more weightage
## all prev same day of weeks data more weightage (Wed+Thurs)(Mon+Fri)
class multiTimeframeSnR(Strategy):
    n1 = 20
    n2 = 10
    n3 = 14
    n4 = 20
    n_dev = 2
    entry_stopLoss = 30
    supportRange = 25
    resistanceRange = 25
    exitRange = 80
    exit_rev = 50
    exitPrice = 0
    support_down_counter = 0
    resistance_up_counter = 0
    support_down_counter = 0
    init_support_range = 100
    rev_candleRange = 30
    levelPriceRange = 60
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
        self.fin_state = "searching"
        self.stopLoss = 0
        # self.daily_data = minuteto.minutetoyear(self.data.df.index[0].month, self.data.df.index[-1].month, self.data.df.index[0].year, self.data.df.index[-1].year)

    def sellCheck(self):
        if self.position.is_long:
            logging.info(str(self.data.index[-1]) + "- Closed long position at " + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl))
            self.position.close()
        if not self.position:
            if self.data.High > self.bollinger_mavg:
                return True
        return False
    
    def buyCheck(self):
        if self.position.is_short:
            logging.info(str(self.data.index[-1]) + "- Closed short position at " + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl))
            self.position.close()
        if not self.position:
            if self.data.Low < self.bollinger_mavg:
                return True
        return False

    def getfractalSupRes(self, timeframe=750): ## timeframe=750 for 2 days ##
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
                    levels.append((i, self.data.Low[i])) ## Support Check
            if self.data.High[i] > self.data.High[i-1] \
                    and self.data.High[i] > self.data.High[i+1] \
                    and self.data.High[i+1] > self.data.High[i+2] \
                    and self.data.High[i-1] > self.data.High[i-2]:
                if np.sum([abs(self.data.High[i]-x) < candle_mean  for x in levels]) == 0: ## Proximity Check
                    levels.append((i,self.data.High[i])) ## Resisitance Check
        return levels

    def createLevels(self, timeframe=800):
        levels = []
        if len(self.data) < timeframe:
            return levels
        levels.append((0, max(self.data.High[-timeframe:]))) # timeframe High
        levels.append((0, min(self.data.Low[-timeframe:]))) # timeframe Low

        df = pd.DataFrame()
        df_index = pd.DatetimeIndex(pd.to_datetime(self.data.index[-timeframe:]).values, name="datetime")
        df["Open"] = self.data.Open[-timeframe:]
        df["High"] = self.data.High[-timeframe:]
        df["Low"] = self.data.Low[-timeframe:]
        df["Close"] = self.data.Close[-timeframe:]
        df["Volume"] = self.data.Volume[-timeframe:]
        df.set_index(df_index, inplace=True)

        # df.resample('5m',  how='ohlc', axis=0, fill_method='bfill')
        candle_len=60
        candle_mean = np.mean(df["High"] - df["Low"])
        for i in range(timeframe-candle_len):
            thisMax = max(self.data.High[-i-candle_len-1:-i-1])
            thisMin = min(self.data.Low[-i-candle_len-1:-i-1])
            if thisMax - thisMin < self.levelPriceRange:
                thisPrice = (thisMax+thisMin)/2
                if np.sum([abs(thisPrice-x) < candle_mean  for x in levels]) == 0:
                    levels.append((-i-5, thisPrice))
        # # Plot levels
        # indexes, lvls = zip(*levels)
        # mpf.plot(df, hlines=lvls, type="candle")

        return levels
        # highs in 30 point range level
        # lows in 30 point range level

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
            logging.info("Previous day data not available: " + str(current_date))
            priceList.append(self.data.Open[-1] + self.init_support_range)
            priceList.append(self.data.Open[-1] - self.init_support_range)
        print(prev_data.head())
        return priceList

    def next(self):
        if len(self.data.Close) < 751:
            logging.info("close len is " + str(len(self.data.Close)))
            # print("close len is " + str(len(self.data.Close)))
            return
        ## POPULATE SUPPORT AND RESISTANCES
        inRange = False
        # for l in (self.supportLine + self.resistanceLine):
        #     if (self.data.Close[-1] - l) < self.supportRange:
        #         inRange = True
        # if not inRange:
        #     self.fin_state = "searching"
        if self.fin_state == "searching":
            self.supportLine = []
            self.resistanceLine = []
            levels = self.createLevels()
            # print(levels)
            for level in levels:
                pos, l  = level
                if l > self.data.Close[-1]:
                    self.resistanceLine.append(l)
                else:
                    self.supportLine.append(l)


        # if self.vwap > self.data.Close[-1]:
        #     self.resistanceLine.append(self.vwap)
        # else:
        #     self.supportLine.append(self.vwap)        
        # prev_day_prices = self.prevdayPrices()
        # for price in prev_day_prices:
        #     if price > self.data.Close[-1]:
        #         self.resistanceLine.append(price)
        #     else:
        #         self.supportLine.append(price)
        
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

        averagePrice = (self.data.Open[-1] + self.data.Close[-1])/2
        
        ## trailing Constant Stop Loss
        if self.position.is_long:
            if self.data.Close[-1] < self.stopLoss:
                logging.info(str(self.data.index[-1]) + "- Closed long position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                # print(str(self.data.index[-1]) + "- Closed long position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                self.position.close()
            else:
                const_stoploss = self.data.Close[-1] - self.entry_stopLoss
                if const_stoploss > self.stopLoss:
                    self.stopLoss = const_stoploss
        if self.position.is_short:
            if self.data.Close[-1] > self.stopLoss:
                logging.info(str(self.data.index[-1]) + "- Closed short position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                # print(str(self.data.index[-1]) + "- Closed short position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                self.position.close()
            else:
                const_stoploss = self.data.Close[-1] + self.entry_stopLoss
                if const_stoploss < self.stopLoss:
                    self.stopLoss = const_stoploss

        """  ## STOP LOSS
        if self.position.is_long:
            if self.data.Close[-1] < self.stopLoss:
                logging.info(str(self.data.index[-1]) + " Closed Position at Price=" + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl) + ", StopLoss=" + str(self.stopLoss))
                self.position.close()
        if self.position.is_short:
            if self.data.Close[-1] > self.stopLoss:
                logging.info(str(self.data.index[-1]) + " Closed Position at Price=" + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl) + ", StopLoss=" + str(self.stopLoss))
                self.position.close() 
 """
        ## EXIT STRATEGY
        if self.position.is_long:
            if self.data.Close[-1] > self.exitPrice:
                logging.info(str(self.data.index[-1]) + " Exited Position at Price=" + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl))
                self.position.close()
        if self.position.is_short:
            if self.data.Close[-1] < self.exitPrice:
                logging.info(str(self.data.index[-1]) + " Exited Position at Price=" + str(self.data.Close[-1]) + ", Profit=" + str(self.position.pl))
                self.position.close()

        ## RESISTANCE LINE STRATEGY
        if abs(self.data.High - self.resistanceLine[-1]) < self.resistanceRange:
            self.fin_state = "resistanceLine"
        elif self.fin_state is not "supportLine":
            self.fin_state = "searching"
        if self.data.Close[-1] < self.data.Close[-2]:
            self.resistance_up_counter = 0

        if self.fin_state is "resistanceLine":
            logging.info(str(self.data.index[-1]) + "- Price at resisitance line=" + str(self.resistanceLine[-1]))
            logging.info(str(self.resistanceLine) + str(self.supportLine))
            if ((self.data.Open[-1] - self.data.Close[-1]) > self.rev_candleRange):
            # if (self.ema_10 - self.ema_10[-1]) < 2: ## momentum based reversal #todo Change this method
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] - self.exit_rev
                    logging.info("Short order(Big Rev Candle) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            if (self.data.Open[-1] < self.data.Close[-1]) and \
                ((self.data.High[-1]-self.data.Close[-1])/(self.data.High[-1]-self.data.Open[-1])) > 0.4:
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] - self.exit_rev
                    logging.info("Short order(Bounce Reversal) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            if self.data.Close[-1] > self.data.Close[-2]: ## Count based breakout
                self.resistance_up_counter = self.resistance_up_counter + 1
                logging.info("res_up_count="+str(self.resistance_up_counter))
            if self.resistance_up_counter >= 3:
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] + self.exitRange
                    logging.info("Long order(Count Breakout) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            if self.data.High > (self.resistanceLine[-1] + self.resistanceRange): ## price based breakout
                self.supportLine.append(self.resistanceLine[-1])
                del self.resistanceLine[-1]
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] + self.exitRange
                    logging.info("Long order(Price Breakout) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return

        ## SUPPORT LINE STRATEGY
        if abs(self.data.Low - self.supportLine[-1]) < self.supportRange:
            self.fin_state = "supportLine"
        elif self.fin_state is not "resistanceLine":
            self.fin_state = "searching"
        if self.data.Close[-1] > self.data.Close[-2]:
            self.support_down_counter = 0
        

        if self.fin_state is "supportLine": ## Candle range reversal
            logging.info(str(self.data.index[-1]) + "- Price at support line=" + str(self.supportLine[-1]))
            logging.info(str(self.resistanceLine) + str(self.supportLine))
            if ((self.data.Close[-1] - self.data.Open[-1]) > self.rev_candleRange): # Big Rev Candle
                    # or (self.ema_10 - self.ema_10[-1]) > -2: ## momentum based reversal
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] + self.exit_rev
                    logging.info("Long order(Big Rev Candle) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            if (self.data.Close[-1] < self.data.Open[-1]) and \
                ((self.data.Close[-1]-self.data.Low[-1])/(self.data.Open[-1]-self.data.Low[-1])) > 0.4:
                if self.buyCheck():
                    self.buy()
                    self.stopLoss = self.data.Close[-1] - self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] + self.exit_rev
                    logging.info("Long order(Bounce Reversal) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            if self.data.Close[-1] < self.data.Close[-2]: ## Count based breakout
                self.support_down_counter = self.support_down_counter + 1
                logging.info("sup_down_count="+str(self.support_down_counter))
            if self.support_down_counter >= 3:
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] - self.exitRange
                    logging.info("Short order(Count Breakout) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return
            if self.data.Low < (self.supportLine[-1] - self.supportRange): ## price based breakout
                self.resistanceLine.append(self.supportLine[-1])
                del self.supportLine[-1]
                if self.sellCheck():
                    self.sell()
                    self.stopLoss = self.data.Close[-1] + self.entry_stopLoss
                    self.exitPrice = self.data.Close[-1] - self.exitRange
                    logging.info("Short order(Price Breakout) at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                    return


class dl2timeframeStrat(Strategy):
    dlFeatures = ["open", "high", "low", "close", "trend/sma/5", "trend/sma/20", 
                    "trend/ema/3", "trend/ema/9", "trend/ema/27", "volatility/mband", "volatility/hband", 
                    "volatility/lband", "volume/vwap", "momentum/kama", "volume"]
    lookback = 60
    init_null = 27
    stopLoss = 0
    atr_mult = 1.5
    exitRange = 100
    exitPrice = 0
    orderPrice = 0
    def init(self):
        self.vwap = self.I(ta.volume.volume_weighted_average_price, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), pd.Series(self.data.Volume), 14)
        # self.rsi = self.I(ta.momentum.rsi, pd.Series(self.data.Close), 20)
        self.sma5 = self.I(ta.trend.sma_indicator, pd.Series(self.data.Close), 5)
        self.sma20 = self.I(ta.trend.sma_indicator, pd.Series(self.data.Close), 20)
        self.ema3 = self.I(ta.trend.ema_indicator, pd.Series(self.data.Close), 3)
        self.ema9 = self.I(ta.trend.ema_indicator, pd.Series(self.data.Close), 9)
        self.ema27 = self.I(ta.trend.ema_indicator, pd.Series(self.data.Close), 27)
        self.mband = self.I(ta.volatility.bollinger_mavg, pd.Series(self.data.Close), 20)
        self.hband = self.I(ta.volatility.bollinger_hband, pd.Series(self.data.Close), 20, 2)
        self.lband = self.I(ta.volatility.bollinger_lband, pd.Series(self.data.Close), 20, 2)
        self.kama = self.I(ta.momentum.kama, pd.Series(self.data.Close), 10, 2, 30)
        self.atr = self.I(ta.volatility.average_true_range, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), 14)
        self.dl5_model = keras.models.load_model("models/alexNetClassifier_priceFeatures/dlc5min_f15_5_2020_7.h5")
        self.dl15_model = keras.models.load_model("models/alexNetClassifier_priceFeatures/dlc15min_f15_5_2020_7.h5")
        logging.info("Init Complete")

    def prepare_dlData(self, lookback=60):
        if len(self.data.Close) < lookback + self.init_null:
            return []
        priceData = []
        priceData.append(self.data.Open[-lookback:])
        priceData.append(self.data.High[-lookback:])
        priceData.append(self.data.Low[-lookback:])
        priceData.append(self.data.Close[-lookback:])
        priceData.append(self.sma5[-lookback:])
        priceData.append(self.sma20[-lookback:])
        priceData.append(self.ema3[-lookback:])
        priceData.append(self.ema9[-lookback:])
        priceData.append(self.ema27[-lookback:])
        priceData.append(self.mband[-lookback:])
        priceData.append(self.hband[-lookback:])
        priceData.append(self.lband[-lookback:])
        priceData.append(self.vwap[-lookback:])
        priceData.append(self.kama[-lookback:])
        pricerange_max, pricerange_min = np.amax(priceData), np.amin(priceData)
        priceData = np.vectorize(lambda x: (x-pricerange_min)/(pricerange_max-pricerange_min))(priceData)
        volData = self.data.Volume[-lookback:]
        volrange_max, volrange_min = np.amax(volData), np.amin(volData)
        volData = np.vectorize(lambda x: (x-volrange_min)/(volrange_max-volrange_min))(volData)
        return np.reshape(np.append(priceData, np.reshape(volData, (1, lookback)), axis=0), (1, len(self.dlFeatures), lookback, 1))

    def next(self):
        ## trailing ATR Stop Loss
        if self.position.is_long:
            if self.data.Close[-1] < self.stopLoss:
                logging.info(str(self.data.index[-1]) + "- Closed long position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                # print(str(self.data.index[-1]) + "- Closed long position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                self.position.close()
            else:
                atr_stoploss = self.data.Close[-1] - self.atr_mult*self.atr[-1]
                if atr_stoploss > self.stopLoss:
                    self.stopLoss = atr_stoploss
                atr_exit = max(self.data.Close[-1], self.trades[-1].entry_price) + self.atr[-1]
                if atr_exit < self.exitPrice:
                    self.exitPrice = atr_exit
        if self.position.is_short:
            if self.data.Close[-1] > self.stopLoss:
                logging.info(str(self.data.index[-1]) + "- Closed short position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                # print(str(self.data.index[-1]) + "- Closed short position at " + ", Stop Loss=".join([str(self.data.Close[-1]), str(self.stopLoss)]) + ", Profit=" + str(self.position.pl))
                self.position.close()
            else:
                atr_stoploss = self.data.Close[-1] + self.atr_mult*self.atr[-1]
                if atr_stoploss < self.stopLoss:
                    self.stopLoss = atr_stoploss
                atr_exit = max(self.data.Close[-1], self.trades[-1].entry_price) - self.atr[-1]
                if atr_exit > self.exitPrice:
                    self.exitPrice = atr_exit

        if self.position: ## Trailing ATR EXIT STRATEGY
            if self.position.pl > self.exitRange:
                logging.info(str(self.data.index[-1]) + "- Exited position at " + ", Profit=".join([str(self.data.Close[-1]), str(self.position.pl)]))
                # print(str(self.data.index[-1]) + "- Exited position at " + ", Profit=".join([str(self.data.Close[-1]), str(self.position.pl)]))
                self.position.close()

        dl_input = self.prepare_dlData() ## Prepare DL Model Input data
        if not len(dl_input):
            logging.info("Model Input not ready - steps=" + str(len(self.data.Close)))
            # print("Model Input not ready")
            return
        pred_5 = self.dl5_model.predict(dl_input)[0]
        pred_15 = self.dl15_model.predict(dl_input)[0]
        win_5 = pred_5[0]/(pred_5[0] + pred_5[1]) ## Prediction of Probablility for price to go up in 5 mins
        win_15 = pred_15[0]/(pred_15[0] + pred_15[1]) ## Prediction of Probablility for price to go up in 15 mins

        if not self.position:
            if win_5 > 0.7 and win_15 > 0.5:
                logging.info(str(self.data.index[-1]) + " - win5=" + str(win_5) + ", win15=" + str(win_15))
                self.buy()
                self.stopLoss = self.data.Close[-1] - self.atr_mult*self.atr[-1]
                logging.info("Long order at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                # print("Long at:(" + ", ".join([str(self.data.index[-1]), str(self.data.Close[-1])]) + ") Stop Loss=" + str(self.stopLoss))
            if win_5 < 0.3  and win_15 < 0.5:
                logging.info(str(self.data.index[-1]) + " - win5=" + str(win_5) + ", win15=" + str(win_15))
                self.sell()
                self.stopLoss = self.data.Close[-1] + self.atr_mult*self.atr[-1]
                logging.info("Short order at " + str(self.data.index[-1]) + ", Price=" + str(self.data.Close[-1]) + ", Stop Loss=" + str(self.stopLoss))
                # print("Short at:(" + ", ".join([str(self.data.index[-1]), str(self.data.Close[-1])]) + ") Stop Loss=" + str(self.stopLoss))
            

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

""" QQQ Signal trading
from statistics import mean

# QQQ data 1-1-2000 thru 10-19-2019
q3 = { ...PASTE STOCK DATA HERE... }

# total % increase in value
(q3[-1]['adjClose'] - q3[0]['adjClose'])/q3[0]['adjClose']*100

# list of close prices
closePrice = [p['adjClose'] for p in q3]

# add 5day SMA to all values
for day in range(6,len(q3)+1):
    q3[day-1]['5daySMA'] = round(mean(closePrice[day-5:day]),2)

# set the initial EMA (exponential moving average) sample
q3[5]['5dayEMA'] = q3[5]['5daySMA']
# add 5day EMA to all values
for day in range(7,len(q3)+1):
    q3[day-1]['5dayEMA'] = round(0.33*(q3[day-1]['adjClose']-q3[day-2]['5dayEMA'])+q3[day-2]['5dayEMA'],2)

# add Bollinger Band to all values
bollingerDays = 20
for day in range(bollingerDays,len(q3)+1):
    q3[day-1]['20dayBB'] = round(mean(closePrice[day-bollingerDays:day]),2)

# identify EMA and BB crossings, w/label 'Long' or 'Short'
for day in range(20,len(q3)):
    if q3[day-1]['20dayBB']<q3[day-1]['5dayEMA'] and q3[day]['20dayBB']>q3[day]['5dayEMA']:
        q3[day]['signal'] = 'Short'
    elif q3[day-1]['20dayBB']>q3[day-1]['5dayEMA'] and q3[day]['20dayBB']<q3[day]['5dayEMA']:
        q3[day]['signal'] = 'Long'

for year in range(2000,2020):
    print('Year:', year)
    # isolate yearly data
    q3_series = [q for q in q3 if str(year) in q['date']]
    
    priorSignalClose = None
    priotSignalType = None
    gains = []
    for q in q3_series:
        if 'signal' in q:
            # if this is the 1st signal encountered then continue
            if not priorSignalClose:
                priorSignalClose = q['adjClose']
                priorSignalType = q['signal']
                print("%s new position:%s " % (q['date'][0:10], priorSignalType))
                continue

            if priorSignalType == 'Long':
                gain = round((q['adjClose'] - priorSignalClose)/priorSignalClose*100,2)
            else:
                gain = round((priorSignalClose - q['adjClose'])/priorSignalClose*100,2)

            gains.append(gain)
            priorSignalClose = q['adjClose']
            priorSignalType = q['signal']
            print("%s gain:%s%% new position:%s" % (q['date'][0:10], gain, priorSignalType))

    print("%s%%" % round(sum(gains),2))
"""
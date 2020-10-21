## Imports
import logging
# logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# logging.basicConfig(
#     level=logging.DEBUG,
#     # format="%(asctime)s - %(name)s - [ %(message)s ]",
#     # datefmt='%d-%b-%y %H:%M:%S',
#     force=True,
#     handlers=[
#         logging.FileHandler("logs/backtesting_strategies.log"),
#         logging.StreamHandler()
# ])
logging.basicConfig(filename='logs/backtesting_strategies.log', level=logging.DEBUG)
#from utils import data_fetch
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import indicators

from db import timscale_setup
from db import dbscrape
import strategys_backtesting
# import strategy_optimizer
import charts
import mplfinance as mpf

from backtesting import Backtest
#from backtesting.test import SMA
from backtesting import Strategy
#from gooey import Gooey, GooeyParser
# import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx

## Define Variables
banknifty_table = "BANKNIFTY_F1"

## Models
class backtestModel:
    def __init__(self, tablename="BANKNIFTY_F1"):
        self.data = pd.DataFrame()
        self.table_name = tablename
    def fetchData(self, month=2, year=2020, day="all", heiken_ashi=False, startdate="2020-02-02 09:30:00", enddate="2020-02-29 15:30:00", verbose=False):
        #data = dbscrape.gettablerange(*(timscale_setup.get_config()), banknifty_table, startdate, enddate)
        rawdata = dbscrape.expirymonth(*(timscale_setup.get_config()), self.table_name, month, year)
        if verbose:
            print(rawdata.head())
            print(rawdata.tail())
        self.instrument_name = rawdata['internalname'][0]
        self.expiry_date = rawdata["expirydate"][0]
        processed_data = strategys_backtesting.prepareData(rawdata, heiken_ashi=heiken_ashi, verbose=verbose)
        if day is not "all":
            processed_data = processed_data.index.apply(lambda x: x.strftime("%A") == day)
        if verbose:
            print(processed_data.head())
        return processed_data
    def setData(self, processed_data, verbose=False):
        self.data = processed_data
    def prepareBacktest(self, strategy, cash=100000, commission=0.002):
        self.bt = Backtest(self.data, strategy, cash=cash, commission=commission)
        print("backtest prepared")
    def runBacktest(self):
        self.stats = self.bt.run()
        print(self.stats)
    def getReturnsError(self, x, verbose=False):
        stats = self.bt.run(n1=int(x[0]), n2=int(x[1]))
        error = 1/math.exp(stats["Return [%]"]/100)
        if verbose:
            print("Return [%] = " + str(stats["Return [%]"]) + "\nError = " + str(error))
        return error
    def getCumReturnsError(self, xs, verbose=False):
        errors = []
        for x in xs:
            errors.append(self.getReturnsError(x, verbose=verbose))
        return np.array(errors)
    def plot(self, title="test"):
        strategys_backtesting.plot(self.bt, title)
    def finchart(self):
        self.fplt = charts.FinPlotter(self.data, self.instrument_name, format="bt", verbose=True)
        plotname = " ".join([self.table_name, self.instrument_name])
        # self.fplt.fullPlot(name=plotname)
        self.fplt.simplePlot(name=plotname)
        # self.fplt.indiPlot(name=plotname)
    def run(self, model_func=strategys_backtesting.SnRfollowup):
        self.prepareBacktest(model_func, cash=100000, commission=0.0)
        self.runBacktest()

def setupLogging(fname="testLog"):
    # logging.basicConfig(filename=fname,level=logging.DEBUG)
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format("logs", fname))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

def generatelevels(data, timeframe=750): ## run on 5 minute candles/ proximity > 1%
    levels = []
    if len(data) < timeframe:
        return levels
    candle_mean =  np.mean(data.High[-1*timeframe:] - data.Low[-1*timeframe:])
    for j in range(timeframe-2): ## 2 datapoint padding
        i = - j - 1 ## set the right iterator for direction (Currently backwards)
        if data.Low[i] < data.Low[i-1] \
                and data.Low[i] < data.Low[i+1] \
                and data.Low[i+1] < data.Low[i+2] \
                and data.Low[i-1] < data.Low[i-2]:
            if np.sum([abs(data.Low[i]-x) < candle_mean  for x in levels]) == 0: ## Proximity Check
                levels.append((i, data.Low[i])) ## Support Check
        if data.High[i] > data.High[i-1] \
                and data.High[i] > data.High[i+1] \
                and data.High[i+1] > data.High[i+2] \
                and data.High[i-1] > data.High[i-2]:
            if np.sum([abs(data.High[i]-x) < candle_mean  for x in levels]) == 0: ## Proximity Check
                levels.append((i,data.High[i])) ## Resisitance Check
    return levels

## dl2timeframe_take1
## SnRfollowup_take1
def main():
    model = backtestModel()
    onemin_data = model.fetchData(month=2, year=2020)
    threemin_data = indicators.chart_resample(onemin_data, target_sr=3)
    threemin_data = indicators.heiken_ashi(threemin_data)
    model.setData(threemin_data)
    # logging.info(str(threemin_data.head()))
    print(threemin_data.head())
    print(model.data.head())
    # indicators.SRLevels(model.data[0:800])
    # lowC, highC = indicators.getCenters(model.data["Low"][0:750], model.data["High"][0:750])
    # charts.plot_stock_data_centers(model.data.iloc[0:750], highC, lowC)

    # # Plot levels
    # indexes, lvls = zip(*levels)
    # mpf.plot(df, hlines=lvls, type="candle")

    # RUN Model
    model.run(model_func=strategys_backtesting.SnRfollowup)
    model.plot(title="SRFollowup_hAshi_3min_feb_test8")
    # model.finchart()
    #pyswarmstrat = strategy_optimizer.PyswarmOptimizer(model_func=backtestModel)
    #pyswarmstrat.optimize()
    # pso = strategy_optimizer.PSO(dims=2, numOfBoids=30, numOfEpochs=500)
    # pso.optimize()

if __name__ == "__main__":
    setupLogging(fname="SRFollowup_hAshi_3min_feb_test8")
    logging.info("changed breakout detection to Close price")
    main()
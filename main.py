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
    def fetchData(self, month=1, year=2020, startdate="2020-02-02 09:30:00", enddate="2020-02-29 15:30:00"):
        #data = dbscrape.gettablerange(*(timscale_setup.get_config()), banknifty_table, startdate, enddate)
        rawdata = dbscrape.expirymonth(*(timscale_setup.get_config()), self.table_name, month, year)
        return rawdata
    def setData(self, rawdata, day="all", verbose=False):
        self.instrument_name = rawdata['internalname'][0]
        self.expiry_date = rawdata["expirydate"][0]
        self.data = strategys_backtesting.prepareData(rawdata, heiken_ashi=False, verbose=verbose)
        if day is not "all":
            self.data = self.data.index.apply(lambda x: x.strftime("%A") == day)
        if verbose:
            print(self.data.head())
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
        self.setData(self.fetchData(), verbose=True)
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
    model.setData(model.fetchData(month=1, year=2020))
    indicators.SRLevels(model.data[0:800])
    lowC, highC = indicators.getCenters(model.data["Low"][0:750], model.data["High"][0:750])
    charts.plot_stock_data_centers(model.data[0:750], highC, lowC)
    # model.run(model_func=strategys_backtesting.SnRfollowup)
    # model.plot(title="supres_test1")
    # model.finchart()
    #pyswarmstrat = strategy_optimizer.PyswarmOptimizer(model_func=backtestModel)
    #pyswarmstrat.optimize()
    # pso = strategy_optimizer.PSO(dims=2, numOfBoids=30, numOfEpochs=500)
    # pso.optimize()

if __name__ == "__main__":

    # setupLogging(fname="dl2timeframeStrat_take2")
    main()
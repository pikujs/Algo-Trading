## Imports
#from utils import data_fetch
from datetime import datetime
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#import indicators
from db import timscale_setup
from db import dbscrape_old as dbscrape
import strategys_backtesting
import strategy_optimizer
import charts

from backtesting import Backtest
#from backtesting.test import SMA
from backtesting import Strategy

#from gooey import Gooey, GooeyParser
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


## Define Variables

banknifty_table = "BANKNIFTY_F1"

#data = data_fetch.finnhub_hist("AAPL")

## Preprocessing


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
        self.data = strategys_backtesting.prepareData(rawdata, verbose=verbose)
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
    def run(self):
        self.setData(self.fetchData(), verbose=True)
        self.prepareBacktest(strategys_backtesting.SnRfollowup, cash=100000, commission=0.0)
        self.runBacktest()




## Optimization


## Pyswarms



## Output
# Create the plot
""" 
xs = []
for i in range(10):
    xs.append([5*i, 10*i])
 """
def main():
    model = backtestModel()
    model.setData(model.fetchData(month=1, year=2020))
    model.run()
    model.plot()
    # model.finchart()
    #pyswarmstrat = strategy_optimizer.PyswarmOptimizer()
    #pyswarmstrat.optimize()
    # pso = strategy_optimizer.PSO(dims=2, numOfBoids=30, numOfEpochs=500)
    # pso.optimize()

if __name__ == "__main__":
    main()
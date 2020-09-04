## Imports
#from utils import data_fetch
from datetime import datetime
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#import indicators
from db import timscale_setup
from db import dbscrape
import strategys_backtesting
import strategy_optimizer

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
    def __init__(self):
        self.data = pd.DataFrame()
        self.table_name = "BANKNIFTY_F1"
    def fetchData(self, month=4, year=2020, startdate="2020-02-02 09:30:00", enddate="2020-02-29 15:30:00"):
        #data = dbscrape.gettablerange(*(timscale_setup.get_config()), banknifty_table, startdate, enddate)
        rawdata = dbscrape.expirymonth(*(timscale_setup.get_config()), self.table_name, month, year)
        return rawdata
    def setData(self, rawdata, verbose=False):
        self.data = strategys_backtesting.prepareData(rawdata, verbose=verbose)
    def prepareBacktest(self, strategy, cash=100000, commission=0.002):
        self.bt = Backtest(self.data, strategy, cash=cash, commission=commission)
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
    def run(self):
        self.setData(self.fetchData(), verbose=True)
        self.prepareBacktest(strategys_backtesting.BollingerStrat)
        self.runBacktest()




## Optimization


## Pyswarms



## Output
# Create the plot

xs = []
for i in range(10):
    xs.append([5*i, 10*i])

def main():
    model = backtestModel()
    model.run()
    model.plot()
    #pyswarmstrat = strategy_optimizer.PyswarmOptimizer()
    #pyswarmstrat.optimize()
    # pso = strategy_optimizer.PSO(dims=2, numOfBoids=30, numOfEpochs=500)
    # pso.optimize()

if __name__ == "__main__":
    main()
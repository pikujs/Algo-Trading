import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])


# Import the backtrader platform
import backtrader as bt


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('printlog', False),
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)


class BTstrategy:
    def __init__(self):
        # Create a cerebro entity
        self.cerebro = bt.Cerebro()
        # Datas are in a subfolder of the samples. Need to find where the script is
        # because it could have been called from anywhere
        self.modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.datapath = os.path.join(modpath, '../../datas/orcl-1995-2014.txt')


    def addStrategy(self, strategy=TestStrategy):
        # Add a strategy
        self.strats = self.cerebro.optstrategy(
            strategy,
            maperiod=range(10, 31))

    def getData(self):
        # Create a Data Feed
        self.data = bt.feeds.YahooFinanceCSVData(
            dataname=datapath,
            # Do not pass values before this date
            fromdate=datetime.datetime(2000, 1, 1),
            # Do not pass values before this date
            todate=datetime.datetime(2000, 12, 31),
            # Do not pass values after this date
            reverse=False)

        # Add the Data Feed to Cerebro
        self.cerebro.adddata(self.data)

    def prepareBacktest(self):
        # Set our desired cash start
        self.cerebro.broker.setcash(1000.0)

        # Add a FixedSize sizer according to the stake
        self.cerebro.addsizer(bt.sizers.FixedSize, stake=10)

        # Set the commission
        self.cerebro.broker.setcommission(commission=0.0)

    def run(self):
        # Run over everything
        self.cerebro.run(maxcpus=1)

if __name__ == '__main__':
    pass


    ## TradeView Momentum Strat:
##strategy("QuantCat Mom Finder Strateg (1H)", overlay=true)

#Series to sum the amount of crosses in EMA for sideways trend/noise filtering
#can change EMA lengths, can change to SMA's/WMA's e.t.c

lookback_value = 60
minMA = 20
midMA = 40
maxMA = 60

ema25_crossover = (crossover(close, ema(close, minMA)))
ema25_crossover_sum = sum(ema25_crossover, lookback_value) #potentially change lookback value to alter results

ema50_crossover = (crossover(close, ema(close, midMA)))
ema50_crossover_sum = sum(ema50_crossover, lookback_value) #potentially change lookback value to alter results

ema75_crossover = (crossover(close, ema(close, maxMA)))
ema75_crossover_sum = sum(ema75_crossover, lookback_value) #potentially change lookback value to alter results

ema25_crossunder = (crossunder(close, ema(close, minMA)))
ema25_crossunder_sum = sum(ema25_crossunder, lookback_value) #potentially change lookback value to alter results

ema50_crossunder = (crossunder(close, ema(close, midMA)))
ema50_crossunder_sum = sum(ema50_crossunder, lookback_value) #potentially change lookback value to alter results

ema75_crossunder = (crossunder(close, ema(close, maxMA)))
ema75_crossunder_sum = sum(ema75_crossunder, lookback_value) #potentially change lookback value to alter results4


# Boolean series declaration
# can change amount of times crossed over the EMA verification to stop sideways trend filtering (3)

maxNoCross=2

macdmidlinebull=-0.5
macdmidlinebear=0.5
macdLine, signalLine, histLine = macd(close, 12, 26, 9)

# Series Creation

bullishMacd = (macdLine > signalLine) and (macdLine > macdmidlinebull)

bearishMacd = (macdLine < signalLine) and (macdLine < macdmidlinebear)

bullRsiMin = 50 #53 initial values
bullRsiMax = 60 #61
bearRsiMin = 40 #39
bearRsiMax = 50 #47

basicBullCross25bool = ((ema25_crossover_sum < ema50_crossover_sum)
     and (ema25_crossover_sum < ema75_crossover_sum)
     and (ema25_crossover_sum < maxNoCross)
     and crossover(close, ema(close, minMA)) and (rsi(close, 14) > bullRsiMin)
     and (rsi(close, 14) < bullRsiMax) and bullishMacd)
 
basicBullCross50bool = ((ema50_crossover_sum < ema25_crossover_sum)
     and (ema50_crossover_sum < ema75_crossover_sum)
     and (ema50_crossover_sum < maxNoCross)
     and crossover(close, ema(close, midMA)) and (rsi(close, 14) > bullRsiMin)
     and (not basicBullCross25bool)
     and (rsi(close, 14) < bullRsiMax) and bullishMacd)
 
basicBullCross75bool = ((ema75_crossover_sum < ema25_crossover_sum)
     and (ema75_crossover_sum < ema50_crossover_sum)
     and (ema75_crossover_sum < maxNoCross)
     and crossover(close, ema(close, maxMA)) and (rsi(close, 14) > bullRsiMin)
     and (not basicBullCross25bool) and (not basicBullCross50bool)
     and (rsi(close, 14) < bullRsiMax) and bullishMacd)
     
basicBearCross25bool = ((ema25_crossunder_sum < ema50_crossunder_sum)
     and (ema25_crossunder_sum < ema75_crossunder_sum)
     and (ema25_crossunder_sum < maxNoCross)
     and crossunder(close, ema(close, minMA)) and (rsi(close, 14) <bearRsiMax)
     and (rsi(close, 14) > bearRsiMin) and bearishMacd)
 
basicBearCross50bool = ((ema50_crossunder_sum < ema25_crossunder_sum)
     and (ema50_crossunder_sum < ema75_crossover_sum)
     and (ema50_crossunder_sum < maxNoCross)
     and crossunder(close, ema(close, midMA)) and (rsi(close, 14) < bearRsiMax)
     and (not basicBearCross25bool)
     and (rsi(close, 14) > bearRsiMin) and bearishMacd)
 
basicBearCross75bool = ((ema75_crossunder_sum < ema25_crossunder_sum)
     and (ema75_crossunder_sum < ema50_crossunder_sum)
     and (ema75_crossunder_sum < maxNoCross)
     and crossunder(close, ema(close, maxMA)) and (rsi(close, 14) < bearRsiMax)
     and (not basicBearCross25bool) and (not basicBearCross50bool)
     and (rsi(close, 14) > bearRsiMin) and bearishMacd)

# STRATEGY
# can change lookback input on ATR

atrLkb = input(14, minval=1, title='ATR Stop Period')
atrRes = input("D", type=resolution, title='ATR Resolution')
atr = security(tickerid, atrRes, atr(atrLkb))


longCondition = (basicBullCross25bool or basicBullCross50bool or basicBullCross75bool)
if (longCondition):
    strategy.entry("Long", strategy.long)

shortCondition = (basicBearCross25bool or basicBearCross50bool or basicBearCross75bool)
if (shortCondition):
    strategy.entry("Short", strategy.short)
   
   
# Calc ATR Stops
# can change atr multiplier to affect stop distance/tp distance, and change "close" to ema values- could try ema 50

stopMult = 0.6 # 0.6 is optimal

longStop = None
longStop =  (close - (atr * stopMult) if longCondition and (strategy.position_size <=0) else longStop[1]) if not shortCondition else None 
shortStop = None
shortStop = (close + (atr * stopMult) if shortCondition and (strategy.position_size >=0) else shortStop[1]) if not longCondition else None 

# Calc ATR Target

targetMult = 2.2 # 2.2 is optimal for crypto x/btc pairs

longTarget = None
longTarget =  (close + (atr*targetMult) if (longCondition and (strategy.position_size <=0)) else longTarget[1]) if not shortCondition else None 
shortTarget = None
shortTarget =  (close - (atr*targetMult) if shortCondition and (strategy.position_size >=0) else shortTarget[1]) if not longCondition else None 

# Place the exits

strategy.exit("Long ATR Stop", "Long", stop=longStop, limit=longTarget)
strategy.exit("Short ATR Stop", "Short", stop=shortStop, limit=shortTarget)

# Bar color series

longColour = lime if longCondition else None
shortColour = red if shortCondition else None
   
# Plot the stoplosses and targets

plot(longStop, style=linebr, color=red, linewidth=2,     title='Long ATR Stop')
plot(shortStop, style=linebr, color=red, linewidth=2,  title='Short ATR Stop')
plot(longTarget, style=linebr, linewidth=2, color=lime,  title='Long ATR Target')
plot(shortTarget, linewidth=2, style=linebr, color=lime,  title='Long ATR Target')

barcolor(color=longColour)
barcolor(color=shortColour)

alertcondition((basicBullCross25bool or basicBullCross50bool or basicBullCross75bool), title='Long Entry', message='Bullish Momentum Change!')
alertcondition((basicBearCross25bool or basicBearCross50bool or basicBearCross75bool), title='Short Entry', message='Bearish Momentum Change!')
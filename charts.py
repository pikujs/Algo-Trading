#Import
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import finplot as fplt
import requests
from io import StringIO
from time import time
import strategys_backtesting

#import quantmod

## Definitions

class FinPlotter:
    def __init__(self, data, instrumentname, format=None, verbose=False):
        if not format:
            self.data = data
        else:
            self.data = self.formatData(data, format)
        self.indicators = pd.DataFrame()
        self.instrument_name = instrumentname
        if verbose:
            print(self.data.head())
    def formatData(self, data, format=None):
        if not format:
            return data
        if format is "db":
            data = strategys_backtesting.prepareData(data)
            format = "bt"
        if format is "bt":
            data["dt"] = data.index.astype("int64")
            # data.set_index(range(len(data.index)))
            return data
        # self.data.index = self.data.index.astype('int64')
    def addI(self, func, name="indi", *args, **kwargs):
        if self.indicators[name]:
            name = name + "_"
        self.indicators[name] = func(self.data, *args, **kwargs)
    def linePlot(self, line=None, name="Line 1"):
        if not line:
            line = (self.data["datetime"][100], self.data["Close"][100]), (self.data["datetime"][-100], self.data["Close"][-100])
        fplt.plot(self.data["datetime"], self.data["Close"], width=3)
        line = fplt.add_line(line, color='#9900ff', interactive=True)
        ## fplt.remove_line(line)
        text = fplt.add_text((self.data["datetime"][len(self.data["datetime"])/2], self.data["Close"][len(self.data["datetime"])/2]), name, color='#bb7700')
        ## fplt.remove_text(text)
        self.fplt = fplt
        fplt.timer_callback(self.savepng, 0.5, single_shot=True) # wait some until we're rendered
        fplt.show()
    def savepng(self, name=None):
        self.fplt.screenshot(open("screenshot_" + str(name) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".png", 'wb'))
    def fullPlot(self, name="Full Plot"):
        ax1,ax2,ax3,ax4,ax5 = fplt.create_plot(name, rows=5, maximize=False)

        fplt.plot(self.data.Close, color='#000', legend='Price', ax=ax1)
        self.indicators['ma200'] = self.data.Close.rolling(200).mean()
        self.indicators['ma50'] = self.data.Close.rolling(50).mean()
        fplt.plot(self.indicators.ma200, legend='MA200', ax=ax1)
        fplt.plot(self.indicators.ma50, legend='MA50', ax=ax1)
        self.indicators['one'] = 1
        fplt.volume_ocv(self.indicators[['ma200','ma50','one']], candle_width=1, ax=ax1.overlay(scale=0.02))

        daily_ret = self.data.Close.pct_change()*100
        fplt.plot(daily_ret, width=3, color='#000', legend='Daily returns %', ax=ax2)

        fplt.add_legend('Daily % returns histogram', ax=ax3)
        fplt.hist(daily_ret, bins=60, ax=ax3)

        fplt.add_legend('Yearly returns in %', ax=ax4)
        """ # fplt.bar(self.data.Close.resample('Y').last().pct_change().dropna()*100, ax=ax4)

        # calculate monthly returns, display as a 4x3 heatmap
        months = self.data['Close'].resample('M').last().pct_change().dropna().to_frame() * 100
        months.index = mnames = months.index.month_name().to_list()
        mnames = mnames[mnames.index('January'):][:12]
        mrets = [months.loc[mname].mean()[0] for mname in mnames]
        hmap = pd.DataFrame(columns=[2,1,0], data=np.array(mrets).reshape((3,4)).T)
        hmap = hmap.reset_index() # use the range index as X-coordinates (if no DateTimeIndex is found, the first column is used as X)
        fplt.heatmap(hmap, rect_size=1, colcurve=lambda x: x, ax=ax5)
        for j,mrow in enumerate(np.array(mnames).reshape((3,4))):
            for i,month in enumerate(mrow):
                s = month+' %+.2f%%'%hmap.loc[i,2-j]
                fplt.add_text((i, 2.5-j), s, anchor=(0.5,0.5), ax=ax5)
        ax5.set_visible(crosshair=False, xaxis=False, yaxis=False) # hide junk for a more pleasing look """
        fplt.show()
    def simplePlot(self, name="Price Plot"):
        ax,ax2 = fplt.create_plot(name, rows=2)
        # plot macd with standard colors first
        macd = self.data.Close.ewm(span=12).mean() - self.data.Close.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        self.data['macd_diff'] = macd - signal
        #fplt.volume_ocv(self.data[['datetime','Open','Close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
        fplt.plot(macd, ax=ax2, legend='MACD')
        fplt.plot(signal, ax=ax2, legend='Signal')

        # change to b/w coloring templates for next plots
        fplt.candle_bull_color = fplt.candle_bear_color = '#000'
        fplt.volume_bull_color = fplt.volume_bear_color = '#333'
        fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'

        # plot price and volume
        fplt.candlestick_ochl(self.data[["dt", "Open", "Close", "High", "Low"]], ax=ax)
        hover_label = fplt.add_legend('', ax=ax)
        axo = ax.overlay()
        #fplt.volume_ocv(self.data[['datetime','Open','Close','Volume']], ax=axo)
        fplt.plot(self.data.Volume.ewm(span=24).mean(), ax=axo, color=1)

        #######################################################
        ## update crosshair and legend when moving the mouse ##

        def update_legend_text(x, y):
            print("x="+ str(x))
            print("y=" + str(y))
            row = self.data.loc[self.data["dt"] == x]
            print(row)
            # row = self.data.loc[self.data.index==pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S")]
            # print(pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
            # format html with the candle and set legend
            fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open<row.Close).all() else 'a00')
            rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
            hover_label.setText(rawtxt % (self.instrument_name, "1 min", row.Open, row.Close, row.High, row.Low))

        def update_crosshair_text(x, y, xtext, ytext):
            ytext = '%s (Close%+.2f)' % (ytext, (y - self.data.iloc[x].Close))
            return xtext, ytext

        # fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
        fplt.set_time_inspector(update_legend_text, ax=ax)
        fplt.add_crosshair_info(update_crosshair_text, ax=ax)

        fplt.show()

    def indiPlot(self, name="Indi Plot", icators=None):
        def local2timestamp(s):
            return int(dateutil.parser.parse(s).timestamp())

        def plot_accumulation_distribution(df, ax):
            ad = (2*df.Close-df.High-df.Low) * df.Volume / (df.High - df.Low)
            df['acc_dist'] = ad.cumsum().ffill()
            fplt.plot(df.datetime, df.acc_dist, ax=ax, legend='Accum/Dist', color='#f00000')


        def plot_bollinger_bands(df, ax):
            mean = df.Close.rolling(20).mean()
            stddev = df.Close.rolling(20).std()
            df['boll_hi'] = mean + 2.5*stddev
            df['boll_lo'] = mean - 2.5*stddev
            p0 = fplt.plot(df.datetime, df.boll_hi, ax=ax, color='#808080', legend='BB')
            p1 = fplt.plot(df.datetime, df.boll_lo, ax=ax, color='#808080')
            fplt.fill_between(p0, p1, color='#bbb')


        def plot_ema(df, ax):
            fplt.plot(df.datetime, df.Close.ewm(span=9).mean(), ax=ax, legend='EMA')


        def plot_heikin_ashi(df, ax):
            df['h_close'] = (df.Open+df.Close+df.High+df.Low) * 0.25
            df['h_open'] = (df.Open.shift()+df.Close.shift()) * 0.5
            df['h_high'] = df[['High','h_open','h_close']].max(axis=1)
            df['h_low'] = df[['Low','h_open','h_close']].min(axis=1)
            candles = df['datetime h_open h_close h_high h_low'.split()]
            fplt.candlestick_ochl(candles, ax=ax)


        def plot_heikin_ashi_volume(df, ax):
            volume = df['datetime h_open h_close Volume'.split()]
            fplt.volume_ocv(volume, ax=ax)


        def plot_on_balance_volume(df, ax):
            obv = df.Volume.copy()
            obv[df.Close < df.Close.shift()] = -obv
            obv[df.Close==df.Close.shift()] = 0
            df['obv'] = obv.cumsum()
            fplt.plot(df.datetime, df.obv, ax=ax, legend='OBV', color='#008800')


        def plot_rsi(df, ax):
            diff = df.Close.diff().values
            gains = diff
            losses = -diff
            with np.errstate(invalid='ignore'):
                gains[(gains<0)|np.isnan(gains)] = 0.0
                losses[(losses<=0)|np.isnan(losses)] = 1e-10 # we don't want divide by zero/NaN
            n = 14
            m = (n-1) / n
            ni = 1 / n
            g = gains[n] = np.nanmean(gains[:n])
            l = losses[n] = np.nanmean(losses[:n])
            gains[:n] = losses[:n] = np.nan
            for i,v in enumerate(gains[n:],n):
                g = gains[i] = ni*v + m*g
            for i,v in enumerate(losses[n:],n):
                l = losses[i] = ni*v + m*l
            rs = gains / losses
            df['rsi'] = pd.Series(100 - (100/(1+rs)))
            fplt.plot(df.datetime, df.rsi, ax=ax, legend='RSI')
            fplt.set_y_range(0, 100, ax=ax)
            fplt.add_band(30, 70, ax=ax)


        def plot_vma(df, ax):
            fplt.plot(df.datetime, df.Volume.rolling(20).mean(), ax=ax, color='#c0c030')


        ax,axv,ax2,ax3,ax4 = fplt.create_plot(name, rows=5)

        # price chart
        plot_heikin_ashi(self.data, ax)
        plot_bollinger_bands(self.data, ax)
        plot_ema(self.data, ax)

        # volume chart
        plot_heikin_ashi_volume(self.data, axv)
        plot_vma(self.data, ax=axv)

        # some more charts
        plot_accumulation_distribution(self.data, ax2)
        plot_on_balance_volume(self.data, ax3)
        plot_rsi(self.data, ax4)

        # restore view (X-position and zoom) when we run this example again
        fplt.autoviewrestore()

        fplt.show()

## bad and slow
def plotly_candlestick(data, instrumentName):
    fig = go.Figure(data=[go.Candlestick(x=data['datetime'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'])])
    fig.update_layout(
        title= {
            'text': instrumentName,
        'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(
            family="Courier New, monospace",
            size=20,
            color="#7f7f7f"
        )
        )
    fig.show()
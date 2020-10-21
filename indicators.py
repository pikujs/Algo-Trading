## Imports
import pandas as pd
import numpy as np
import datetime
import pytz
import ta
import matplotlib.pyplot as plt
import matplotlib.ticker as mpticker
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

## Indicators

# Compute the Bollinger Bands 
def bolinger_scratch(data, window=20, std_n=2, heiken_ashi=True):
    MA = data.Close.rolling(window=window).mean()
    SD = data.Close.rolling(window=window).std()
    new_data = pd.DataFrame()
    if heiken_ashi:
        heiken_ashi(data)
    else:
        new_data['UpperBB'] = MA + (std_n * SD) 
        new_data['LowerBB'] = MA - (std_n * SD)
    return new_data

def chart_resample(df, target_sr=5, current_sr=1):
    df_res = pd.DataFrame()
    datetimes = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    vwaps = []
    sr_ratio = int(target_sr/current_sr)
    i=0
    while i < df.index.size:
        thisTime = df.index[i]
        max_len = min(sr_ratio, df.index.size-i)
        next_i=0
        # print(str((df.index[i+max_len-1] - df.index[i])))
        # print(type((df.index[i+max_len-1] - df.index[i])))
        if (df.index[i+max_len-1] - df.index[i]) == datetime.timedelta(minutes=target_sr-1):
            next_i = i + max_len
        else:
            for j in range(max_len):
                if (df.index[i+j] - thisTime) != datetime.timedelta(minutes=current_sr*(j+1)):
                    next_i = i+j+1
                    break
        if not next_i:
            next_i=i+1
        datetimes.append(thisTime)
        opens.append(df["Open"][i])
        closes.append(df["Close"][next_i-1])
        highs.append(df["High"][i:next_i].max())
        lows.append(df["Low"][i:next_i].min())
        volumes.append(df["Volume"][i:next_i].sum())
        thisVwap = 0
        for k in range(max_len):
            thisVwap = thisVwap + (df["Close"][i+k]+df["Open"][i+k]+df["Low"][i+k])*df["Volume"][i+k]/3
        thisVwap = thisVwap/volumes[-1]
        vwaps.append(thisVwap)
        i = next_i
    if len(opens) == len(closes) == len(highs) == len(lows) == len(volumes) == len(datetimes) == len(vwaps):
        dt_index = pd.DatetimeIndex(datetimes, name="datetime")
        df_res["Open"] = pd.Series(opens)
        df_res["High"] = pd.Series(highs)
        df_res["Low"] = pd.Series(lows)
        df_res["Close"] = pd.Series(closes)
        df_res["Volume"] = pd.Series(volumes)
        df_res["VWAP"] = pd.Series(vwaps)
        df_res.set_index(dt_index, inplace=True)
    
    return df_res

def heiken_ashi(df):
    df['HA_Close']=(df['Open']+ df['High']+ df['Low']+df['Close'])/4
    df["HA_Open"] = (df["Open"] + df["Close"]) / 2
    df["HA_Open"] = df["HA_Open"].shift(periods=1)
    df["HA_Open"][0] = df["HA_Open"][1]
    idx = df.index.name
    df.reset_index(inplace=True)
    if idx:
        df.set_index(idx, inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','High']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','Low']].min(axis=1)
    df.rename(columns = {'Open':'Open_old', 'High':'High_old', 'Low':'Low_old', 'Close':'Close_old', \
                        'HA_Open':'Open', 'HA_High':'High', 'HA_Low':'Low', 'HA_Close':'Close'}, inplace = True)
    return df

def donchian_channel(df):
    highs = [df["High"][0]]
    lows = [df["High"][0]]
    for d in range(1,df.index.size):
        if highs[-1] < df["High"][d]:
            highs.append(df["High"][d])
        else:
            highs.append(highs[-1])
        if lows[-1] < df["Low"][d]:
            lows.append(df["Low"][d])
        else:
            lows.append(lows[-1])
    df["DC_High"] = pd.Series(highs)
    df["DC_Low"] = pd.Series(lows)
    return df

def SRLevels(df):
    # Calculate VERY simple waves
    # mx = df.High_15T.rolling( 100 ).max().rename('waves')
    # mn = df.Low_15T.rolling( 100 ).min().rename('waves')
    mx = df["High"].rolling( 100 ).max().rename('waves')
    mn = df["Low"].rolling( 100 ).min().rename('waves')

    mx_waves = pd.concat([mx,pd.Series(np.zeros(len(mx))+1)],axis = 1)
    mn_waves = pd.concat([mn,pd.Series(np.zeros(len(mn))+-1)],axis = 1)    

    mx_waves.drop_duplicates('waves',inplace = True)
    mn_waves.drop_duplicates('waves',inplace = True)

    W = mx_waves.append(mn_waves).sort_index()
    W = W[ W[0] != W[0].shift() ].dropna()

    # Find Support/Resistance with clustering

    # Create [x,y] array where y is always 1
    X = np.concatenate((W.waves.values.reshape(-1,1),
                        (np.zeros(len(W))+1).reshape(-1,1)), axis = 1 )

    # Pick n_clusters, I chose the sqrt of the df + 2
    n = round(len(W)**(1/2)) + 2
    cluster = AgglomerativeClustering(n_clusters=n,
            affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)
    W['clusters'] = cluster.labels_

    # I chose to get the index of the max wave for each cluster
    W2 = W.loc[W.groupby('clusters')['waves'].idxmax()]

    # Plotit
    fig, axis = plt.subplots()
    for row in W2.itertuples():

        axis.axhline( y = row.waves, 
                color = 'green', ls = 'dashed' )

    axis.plot( W.index.values, W.waves.values )
    plt.show()

def get_optimum_clusters(df, saturation_point=0.05):
    '''

    :param df: dataframe
    :param saturation_point: The amount of difference we are willing to detect
    :return: clusters with optimum K centers

    This method uses elbow method to find the optimum number of K clusters
    We initialize different K-means with 1..10 centers and compare the inertias
    If the difference is no more than saturation_point, we choose that as K and move on
    '''

    wcss = []
    k_models = []

    size = min(11, len(df.index))
    for i in range(1, size):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)

    # Compare differences in inertias until it's no more than saturation_point
    optimum_k = len(wcss)-1
    for i in range(0, len(wcss)-1):
        diff = abs(wcss[i+1] - wcss[i])
        if diff < saturation_point:
            optimum_k = i
            break

    print("Optimum K is " + str(optimum_k + 1))
    optimum_clusters = k_models[optimum_k]

    return optimum_clusters

def getCenters(lows, highs):
    low_clusters = get_optimum_clusters(lows)
    low_centers = low_clusters.cluster_centers_
    low_centers = np.sort(low_centers, axis=0)

    high_clusters = get_optimum_clusters(highs)
    high_centers = high_clusters.cluster_centers_
    high_centers = np.sort(high_centers, axis=0)

    return low_centers, high_centers

def getfractalSupRes(df, timeframe=750): ## timeframe=750 for 2 days ##
    levels = []
    if len(df) < timeframe:
        return levels
    candle_mean =  np.mean(df.High[-1*timeframe:] - df.Low[-1*timeframe:])
    for j in range(timeframe-2): ## 2 datapoint padding
        i = - j - 1 ## set the right iterator for direction (Currently backwards)
        if df.Low[i] < df.Low[i-1] \
                and df.Low[i] < df.Low[i+1] \
                and df.Low[i+1] < df.Low[i+2] \
                and df.Low[i-1] < df.Low[i-2]:
            if np.sum([abs(df.Low[i]-x) < candle_mean  for x in levels]) == 0: ## Proximity Check
                levels.append((i, df.Low[i])) ## Support Check
        if df.High[i] > df.High[i-1] \
                and df.High[i] > df.High[i+1] \
                and df.High[i+1] > df.High[i+2] \
                and df.High[i-1] > df.High[i-2]:
            if np.sum([abs(df.High[i]-x) < candle_mean  for x in levels]) == 0: ## Proximity Check
                levels.append((i,df.High[i])) ## Resisitance Check
    return levels

def getMaxMins(df, n_ema=3):
    ema_high = ta.volatility.bollinger_mavg((df["Close"] + df["High"])/2, n_ema)
    d_ema_high = ema_high.diff()
    ema_low = ta.volatility.bollinger_mavg((df["Close"] + df["Low"])/2, n_ema)
    d_ema_low = ema_low.diff()
    minimas = []
    maximas = []
    for d in range(n_ema+1, min(d_ema_high.size, d_ema_low.size)):
        if d_ema_low[d] > 0 and d_ema_low[d-1] < 0:
            minimas.append([ema_low.size-d+1, ema_low[d-1]])
        if d_ema_high[d] < 0 and d_ema_high[d-1] > 0:
            maximas.append([ema_high.size-d+1, ema_high[d-1]])
    return maximas, minimas


def SRclusters(df, timeframe=800, saturation_point=0.05):
    '''

    :param df: dataframe
    :param saturation_point: The amount of difference we are willing to detect
    :return: clusters with optimum K centers

    This method uses elbow method to find the optimum number of K clusters
    We initialize different K-means with 1..10 centers and compare the inertias
    If the difference is no more than saturation_point, we choose that as K and move on
    '''

    maxs, mins = getMaxMins(df)
    size = min(11, len(maxs), len(mins))
    ## For Resistances
    wcss = []
    k_models = []
    for i in range(1, size):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(maxs)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)

    # Compare differences in inertias until it's no more than saturation_point
    optimum_k = len(wcss)-1
    for i in range(0, len(wcss)-1):
        diff = abs(wcss[i+1] - wcss[i])
        if diff < saturation_point:
            optimum_k = i
            break
    ## For Supports
    wcss = []
    k_models = []
    for i in range(1, size):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(maxs)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)

    # Compare differences in inertias until it's no more than saturation_point
    optimum_k = len(wcss)-1
    for i in range(0, len(wcss)-1):
        diff = abs(wcss[i+1] - wcss[i])
        if diff < saturation_point:
            optimum_k = i
            break

    print("Optimum K is " + str(optimum_k + 1))
    optimum_clusters = k_models[optimum_k]


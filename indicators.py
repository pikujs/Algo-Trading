## Imports
import pandas as pd
import numpy as np
import datetime
import pytz
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


def heiken_ashi(df):
    df['HA_Close']=(df['Open']+ df['High']+ df['Low']+df['Close'])/4

    idx = df.index.name
    df.reset_index(inplace=True)

    for i in range(0, len(df)):
        if i == 0:
            df.set_value(i, 'HA_Open', ((df.get_value(i, 'Open') + df.get_value(i, 'Close')) / 2))
        else:
            df.set_value(i, 'HA_Open', ((df.get_value(i - 1, 'HA_Open') + df.get_value(i - 1, 'HA_Close')) / 2))

    if idx:
        df.set_index(idx, inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','High']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','Low']].min(axis=1)
    df.rename(columns = {'Open':'Open_old', 'High':'High_old', 'Low':'Low_old', 'Close':'Close_old', \
                        'HA_Open':'Open', 'HA_High':'High', 'HA_Low':'Low', 'HA_Close':'Close'}, inplace = True)
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


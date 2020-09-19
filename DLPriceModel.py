import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import csv
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from sklearn import metrics
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
# from sklearn.model_selection import StratifiedKFold
# from sklearn.utils import shuffle
# from keras.layers.core import Dense, Activation, Dropout
import pickle
import time
from progress.bar import Bar
# import indicators
import ta
# import strategys_backtesting
from db import timscale_setup, dbscrape
import datetime as dt
print(tf.executing_eagerly())
## Generating tfrecords

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
  """Returns a float_list from a float / double series."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
  """Returns an int64_list from a bool / enum / int / uint Series."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def prepareData(data, verbose=False):
    exp_date = data["expirydate"][0]
    internalname = data["internalname"][0]
    data_backtesting = data.drop(["internalname", "unknown", "expirydate", "exchange"], axis=1)
    data_backtesting.rename(columns = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace = True)
    if verbose:
        print(data_backtesting.head())
    return data_backtesting, exp_date, internalname
def getData(table_name="BANKNIFTY_F1", month=1, year=2020, verbose=False):
    rawdata = dbscrape.expirymonth(*(timscale_setup.get_config()), table_name, month, year)
    return prepareData(rawdata, verbose=verbose)

def create_expiry_series(data, expiry_date, start, end):
    timeseries = []
    for index, value in data[start:end].items():
        timeseries.append((expiry_date-value).total_seconds())
    return timeseries
def get_offset(data):
    # return data["27ema"].isnull().sum()
    return max(([key, value.isnull().sum()] for key, value in data.iteritems()), key=lambda x: x[1])
def set_indicators(data):
    data["5sma"] = ta.trend.sma_indicator(data["Close"], n=5)
    data["20sma"] = ta.trend.sma_indicator(data["Close"], n=20)
    data["3ema"] = ta.trend.ema_indicator(data["Close"], n=3)
    data["9ema"] = ta.trend.ema_indicator(data["Close"], n=9)
    data["27ema"] = ta.trend.ema_indicator(data["Close"], n=27)
    data["macd"] = ta.trend.macd(data["Close"], n_slow=26, n_fast=12)
    data["mband"] = ta.volatility.bollinger_mavg(data["Close"], n=20)
    data["hband"] = ta.volatility.bollinger_hband(data["Close"], n=20, ndev=2)
    data["lband"] = ta.volatility.bollinger_lband(data["Close"], n=20, ndev=2)
    data["bwidth"] = ta.volatility.bollinger_wband(data["Close"], n=20, ndev=2)
    data["atr"] = ta.volatility.average_true_range(data["High"], data["Low"], data["Close"], n=14)
    data["vwap"] = ta.volume.volume_weighted_average_price(data["High"], data["Low"], data["Close"], data["Volume"], n=14)
    data["adi"] = ta.volume.acc_dist_index(data["High"], data["Low"], data["Close"], data["Volume"])
    data["kama"] = ta.momentum.kama(data["Close"], n=10, pow1=2, pow2=30)
    data["rsi"] = ta.momentum.rsi(data["Close"], n=14)
    data["ultiosc"] = ta.momentum.uo(data["High"], data["Low"], data["Close"], s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0)
    return get_offset(data)

def generate_obs(data, pred_x, expiry_date, timeframe=5, lookback=60, max_offset=0, 
                day_start=dt.time(9,16), day_end=dt.time(15, 30)):
    obs = pd.DataFrame()
    curr_time = data["datetime"][pred_x].time()
    start_timedelta = dt.timedelta(minutes=int(lookback + timeframe + max_offset))
    if (curr_time <= (dt.datetime.combine(dt.date.today(), day_start) + start_timedelta).time()) or (curr_time >= day_end) or (pred_x == len(data)):
        return obs
    init_x = pred_x-timeframe-lookback
    final_x = pred_x-timeframe

    # print("init=" + str(init_x) + " final=" + str(final_x))
    obs["open"] = data["Open"][init_x:final_x].values
    obs["high"] = data["High"][init_x:final_x].values
    obs["low"] = data["Low"][init_x:final_x].values
    obs["close"] = data["Close"][init_x:final_x].values
    obs["volume"] = data["Volume"][init_x:final_x].values
    obs["timetoexpiry"] = create_expiry_series(data["datetime"], expiry_date, init_x, final_x)
    obs["trend/sma/5"] = data["5sma"][init_x:final_x].values
    obs["trend/sma/20"] = data["20sma"][init_x:final_x].values
    obs["trend/ema/3"] = data["3ema"][init_x:final_x].values
    obs["trend/ema/9"] = data["9ema"][init_x:final_x].values
    obs["trend/ema/27"] = data["27ema"][init_x:final_x].values
    obs["trend/macd"] = data["macd"][init_x:final_x].values
    obs["volatility/mband"] = data["mband"][init_x:final_x].values
    obs["volatility/hband"] = data["hband"][init_x:final_x].values
    obs["volatility/lband"] = data["lband"][init_x:final_x].values
    obs["volatility/bwidth"] = data["bwidth"][init_x:final_x].values
    obs["volatility/atr"] = data["atr"][init_x:final_x].values
    obs["volume/vwap"] = data["vwap"][init_x:final_x].values
    obs["volume/adi"] = data["adi"][init_x:final_x].values
    obs["momentum/kama"] = data["kama"][init_x:final_x].values
    obs["momentum/rsi"] = data["rsi"][init_x:final_x].values
    obs["momentum/uo"] = data["ultiosc"][init_x:final_x].values
    obs["pred"] = pd.Series([data["Open"][pred_x], data["High"][pred_x], data["Low"][pred_x], data["Close"][pred_x]])
    return obs

features_format = ["open", "high", "low", "close", "volume", "timetoexpiry", 
            "trend/sma/5", "trend/sma/20", "trend/ema/3", "trend/ema/9", "trend/ema/27", "trend/macd",
            "volatility/mband", "volatility/hband", "volatility/lband", "volatility/bwidth", "volatility/atr",
            "volume/vwap", "volume/adi", "momentum/kama", "momentum/rsi", "momentum/uo"]

priceFeatures_format = ["open", "high", "low", "close", "trend/sma/5", "trend/sma/20", 
                    "trend/ema/3", "trend/ema/9", "trend/ema/27", "volatility/mband", "volatility/hband", 
                    "volatility/lband", "volume/vwap", "momentum/kama"]

lowFeatures_format = ["volatility/bwidth", "trend/macd", "volatility/atr", "momentum/uo", "momentum/rsi"]

otherFeatures_format = ["volume", "volume/adi", "timetoexpiry"]

def obs_preprocessing_df(obs, features_list=features_format):
    new_data = pd.DataFrame()
    for feature in features_list:
        new_data[feature] = pd.Series(obs[feature])
    new_data["pred"] = pd.Series(obs["pred"])
    return new_data

def obs_minmax_scaler(obs, alarm_outofrange=False, features_list=features_format):
    obs_scaled = pd.DataFrame()
    price_columns = [item for item in features_list if item in ["open", "high", "low", "close", "trend/sma/5", "trend/sma/20", 
                    "trend/ema/3", "trend/ema/9", "trend/ema/27", "volatility/mband", "volatility/hband", 
                    "volatility/lband", "volume/vwap", "momentum/kama"]]
    lowrange_columns = [item for item in features_list if item in ["trend/macd", "volatility/atr", "momentum/uo", "momentum/rsi"]]

    if "volume/adi" in features_list:
        obs_scaled["volume/adi"] = obs["volume/adi"].diff()
        obs_scaled["volume/adi"][0] = obs_scaled["volume/adi"][1] - ((obs_scaled["volume/adi"][obs_scaled["volume/adi"].size-1] - obs_scaled["volume/adi"][1])/(obs_scaled["volume/adi"].size-1))
        # print(obs["volume/adi"].to_string())
        # print(obs_scaled["volume/adi"].to_string())

    pricerange_max, pricerange_min = obs[price_columns].max().max(), obs[price_columns].min().min()
    lowrange_max, lowrange_min = obs[lowrange_columns].max().max(), obs[lowrange_columns].min().min()
    volrange_max, volrange_min = obs["volume"].max(), obs["volume"].min()
    adirange_max, adirange_min = obs_scaled["volume/adi"].max(), obs_scaled["volume/adi"].min()
    # print(" / ".join([str(pricerange_max), str(pricerange_min),str(volrange_max),str(volrange_min),str(lowrange_max),str(lowrange_min),]))
    obs_scaled["pred"] = obs["pred"].apply(lambda x: (x-pricerange_min)/(pricerange_max-pricerange_min))
    if alarm_outofrange:
        max_out, min_out = 0, 0
        for i in range(4):
            if (obs_scaled["pred"][i] > 1):
                # print(str(i) + " Index Pred > max = " + str(obs_scaled["pred"][i]))
                max_out = max_out + 1
            if (obs_scaled["pred"][i] < 0):
                # print(str(i) + " Index Pred < min = " + str(obs_scaled["pred"][i]))
                min_out = min_out + 1
        print("Total pred values > max = " + str(max_out))
        print("Total pred values < min = " + str(min_out))
    for feature in price_columns:
        obs_scaled[feature] = obs[feature].apply(lambda x: (x-pricerange_min)/(pricerange_max-pricerange_min))
    for feature in lowrange_columns:
        obs_scaled[feature] = obs[feature].apply(lambda x: (x-lowrange_min)/(lowrange_max-lowrange_min))
    obs_scaled["volume"] = obs["volume"].apply(lambda x: (x-volrange_min)/(volrange_max-volrange_min))
    obs_scaled["volume/adi"] = obs["volume/adi"].apply(lambda x: (x-adirange_min)/(adirange_max-adirange_min))
    obs_scaled["timetoexpiry"] = obs["timetoexpiry"]/(60*60*24*60)
    obs_scaled["volatility/bwidth"] = obs["volatility/bwidth"]
    return obs_scaled

def obs_tfexample(obs, pred_len=4):
    obs_df = obs_minmax_scaler(obs_preprocessing_df(obs))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'data': _float_list_feature(obs_df[features_format].to_numpy().values),
        'pred': _float_list_feature(obs_df["pred"][:pred_len].to_numpy().values),
    }))
    return tf_example

def create_tfexample(obs):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'open': _float_list_feature(obs["open"].values),
        'high': _float_list_feature(obs["high"].values),
        'low': _float_list_feature(obs["low"].values),
        'close': _float_list_feature(obs["close"].values),
        'volume': _float_list_feature(obs["volume"].values),
        'timetoexpiry': _float_list_feature(obs["timetoexpiry"].values),
        'trend/sma/5': _float_list_feature(obs["trend/sma/5"].values),
        'trend/sma/20': _float_list_feature(obs["trend/sma/20"].values),
        'trend/ema/3': _float_list_feature(obs["trend/ema/3"].values),
        'trend/ema/9': _float_list_feature(obs["trend/ema/9"].values),
        'trend/ema/27': _float_list_feature(obs["trend/ema/27"].values),
        'trend/macd': _float_list_feature(obs["trend/macd"].values),
        'volatility/mband': _float_list_feature(obs["volatility/mband"].values),
        'volatility/hband': _float_list_feature(obs["volatility/hband"].values),
        'volatility/lband': _float_list_feature(obs["volatility/lband"].values),
        'volatility/bwidth': _float_list_feature(obs["volatility/bwidth"].values),
        'volatility/atr': _float_list_feature(obs["volatility/atr"].values),
        'volume/vwap': _float_list_feature(obs["volume/vwap"].values),
        'volume/adi': _float_list_feature(obs["volume/adi"].values),
        'momentum/kama': _float_list_feature(obs["momentum/kama"].values),
        'momentum/rsi': _float_list_feature(obs["momentum/rsi"].values),
        'momentum/uo': _float_list_feature(obs["momentum/uo"].values),
        'pred': _float_list_feature(obs["pred"][:4].values),
    }))
    return tf_example

def generate_tfexamples(tablename="BANKNIFTY_F1", month=1, year=2020, timeframe=5, max_i=None, verbose=False):
    data, expiry_date, internalname = getData(table_name=tablename, month=month, year=year, verbose=verbose)
    max_offset = set_indicators(data)
    tf_exs = []
    if verbose:
        print("Got Data for (m, y)= " + str(month) + ", " + str(year))
        print("Max Offset=" + str(max_offset) + " | TimeFrame = " + str(timeframe))
        bar = Bar((internalname + " (" + str(month) + ", " + str(year) + ")"), max=(max_i if max_i else len(data)))
    for index, row in data.iterrows():
        obs = generate_obs(data, pred_x=index, expiry_date=expiry_date, timeframe=timeframe, max_offset=max_offset[1])
        if obs.empty:
            if verbose:
                bar.next()
            #     print("Out if time range/Indicators loading: index=" + str(index) + " row=" + str(row["datetime"]))
            continue
        tf_example = create_tfexample(obs)
        tf_exs.append(tf_example)
        if verbose:
            bar.next()
        if max_i and (index > max_i):
            break
    if verbose:
        bar.finish()
        # print("Generated TFRecord for " + internalname)
    return tf_exs

def save_tfrecords(tf_examples, outputfilePath="data/tfrecords/test.tfrecords", verbose=False):
    with tf.io.TFRecordWriter(outputfilePath) as writer:
        for tf_example in tf_examples:
            writer.write(tf_example.SerializeToString())
    if verbose:
        print("TfRecord written to - " + outputfilePath)
    return outputfilePath

def read_tfrecords(filepath="data/tfrecords/test.tfrecords"):
    raw_obs_dataset = tf.data.TFRecordDataset(filepath)
    lookback = 60
    pred_len = 4
    # Create a dictionary describing the features.
    obs_feature_description = {
        'open': tf.io.FixedLenFeature([lookback], tf.float32),
        'high': tf.io.FixedLenFeature([lookback], tf.float32),
        'low': tf.io.FixedLenFeature([lookback], tf.float32),
        'close': tf.io.FixedLenFeature([lookback], tf.float32),
        'volume': tf.io.FixedLenFeature([lookback], tf.float32),
        'timetoexpiry': tf.io.FixedLenFeature([lookback], tf.float32),
        'trend/sma/5': tf.io.FixedLenFeature([lookback], tf.float32),
        'trend/sma/20': tf.io.FixedLenFeature([lookback], tf.float32),
        'trend/ema/3': tf.io.FixedLenFeature([lookback], tf.float32),
        'trend/ema/9': tf.io.FixedLenFeature([lookback], tf.float32),
        'trend/ema/27': tf.io.FixedLenFeature([lookback], tf.float32),
        'trend/macd': tf.io.FixedLenFeature([lookback], tf.float32),
        'volatility/mband': tf.io.FixedLenFeature([lookback], tf.float32),
        'volatility/hband': tf.io.FixedLenFeature([lookback], tf.float32),
        'volatility/lband': tf.io.FixedLenFeature([lookback], tf.float32),
        'volatility/bwidth': tf.io.FixedLenFeature([lookback], tf.float32),
        'volatility/atr': tf.io.FixedLenFeature([lookback], tf.float32),
        'volume/vwap': tf.io.FixedLenFeature([lookback], tf.float32),
        'volume/adi': tf.io.FixedLenFeature([lookback], tf.float32),
        'momentum/kama': tf.io.FixedLenFeature([lookback], tf.float32),
        'momentum/rsi': tf.io.FixedLenFeature([lookback], tf.float32),
        'momentum/uo': tf.io.FixedLenFeature([lookback], tf.float32),
        'pred': tf.io.FixedLenFeature([pred_len], tf.float32),
    }

    def _parse_obs_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, obs_feature_description)

    parsed_obs_dataset = raw_obs_dataset.map(_parse_obs_function)
    # for pod in parsed_obs_dataset:
    #     print(pod)
    return parsed_obs_dataset
    # parsed_obs_dataset = parsed_obs_dataset.repeat()
    # # Set the number of datapoints you want to load and shuffle 
    # parsed_obs_dataset = parsed_obs_dataset.shuffle(10000)
    # # Set the batchsize
    # parsed_obs_dataset = parsed_obs_dataset.batch(128)
    # iterator = parsed_obs_dataset.make_one_shot_iterator()
    # next = iterator.get_next()
    # print(next) # next is a dict with key=columns names and value=column data
    # inputs = next['text'] 
    # labels = next["pred"]
    # # Create a one hot array for your labels
    # label = tf.one_hot(label, NUM_CLASSES)
    
    # return image, label

def cnnRegressor(lookback=60, num_features=len(features_format), pred_len=4):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(num_features, lookback, 1)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(pred_len, activation='linear'))
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy', 'mae', 'mse'])
    return model

def cnnClassifier(lookback=60, num_features=len(features_format)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(num_features, lookback, 1)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy', 'mae'])
    return model

def alexNetClassifier(lookback=60, num_features=len(features_format)):
    model = Sequential()
    # model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(num_features, lookback, 1)))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(num_features, lookback, 1), padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy', 'mae'])
    return model

def simpleLSTM(lookback=60, num_features=len(features_format), pred_len=4):
    # create the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
    model = Sequential()
    model.add(LSTM(100, input_shape=(num_features, lookback)))
    model.add(Dropout(0.1))
    model.add(Dense(pred_len, activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    return model

def create_dataset(filepath="data/tfrecords/tfex_test.tfrecords", features_list=features_format, 
                    lookback=60, pred_len=4, scaleMinMax=False, make_classifier=True, ret_tfexs=False):
    dataset = read_tfrecords(filepath)
    X = []
    Y = []
    tfexs = []
    num_features = len(features_list)
    if make_classifier:
        pred_len = 2
    bar = Bar('Preprocessing Dataset', max=len(list(dataset.as_numpy_iterator())))
    for d in dataset:
        obs_processed = obs_preprocessing_df(d, features_list=features_list)
        if scaleMinMax:
            obs_processed = obs_minmax_scaler(obs_processed)
        if ret_tfexs:
            tf_ex = create_tfexample(obs_processed)
            tfexs.append(tf_ex)
        obs_list = []
        for feature in features_list:
            obs_list.append(obs_processed[feature].to_numpy())
        X.append(np.array(obs_list))
        if make_classifier:
            if obs_processed["pred"][3] > obs_processed["close"].iloc[-1]:
                Y.append(np.array([1.0, 0.0]))
            else:
                Y.append(np.array([0.0, 1.0]))
        else:
            Y.append(obs_processed["pred"][:pred_len].to_numpy())
        bar.next()
    bar.finish()
    # X = X/np.linalg.norm(X) # Features only Normalisation
    # return np.reshape(X, (len(X), num_features, lookback)), np.reshape(Y, (len(Y), pred_len)) ## for simpleLSTM
    return np.reshape(X, (len(X), num_features, lookback, 1)), np.reshape(Y, (len(Y), pred_len)), tfexs ## For cnnRegressor/Classifier and ret tf_exs
    # return np.reshape(X, (len(X), num_features, lookback, 1)), np.reshape(Y, (len(Y), pred_len)) ## For cnnRegressor/Classifier

mList = [(1, 2020), (2, 2020), (3, 2020), (4, 2020), (5, 2020), (6, 2020), (7, 2020)]
timeframe_list = [5, 15, 30]
test_mList = [(1, 2020)]
test_timeframe_list = [5]
# testfilename = "data/tfrecords/tfex_test.tfrecords"
    
def batch_tfexamples(tablename="BANKNIFTY_F1", month_list=[], timeframe_list=[], 
                    savefile="data/tfrecords/filelist.csv", verbose=False):
    record_files = []
    for tFrame in timeframe_list:
        for month, year in month_list:
            fPath = "data/tfrecords/dl" + str(tFrame) + "_" + tablename + "_" + str(year) + "_" + str(month) + ".tfrecords"
            tf_exs = generate_tfexamples(tablename=tablename, month=month, year=year, timeframe=tFrame, verbose=verbose)
            record_files.append(save_tfrecords(tf_exs, outputfilePath=fPath, verbose=verbose))
    if verbose:
        print(record_files)
    with open(savefile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(record_files)
    return record_files

def train_model(model_func=cnnClassifier, record_files=[], features_list=features_format, 
                record_filelistPath="data/tfrecords/filelist.csv", save_folder="models/", 
                make_classifier=True, save_pickle=False, saveRecords=False, verbose=False):
    if not len(record_files):
        with open(record_filelistPath, "w") as f:
            reader = csv.reader(f)
            for row in reader:
                recordfiles = row
    if not len(record_files):
        print("Record Files not Found at filelistPath = " + record_filelistPath)
        return
    models = {}
    model_counts = {}
    model_filenames = []
    rec_filenames = []
    training_history = {}
    for rFile in record_files:
        if not isinstance(rFile, str):
            print("Record filePath is not a string : " + str(rFile))
            continue
        params = rFile.split("_")
        month = int(params[-1].split(".")[0])
        year = int(params[-2])
        scaleMinMax = True
        if params[1] == "scaled":
            timeframe = int(params[2])
            scaleMinMax = False
        else:
            timeframe = int(params[0].split("dl")[-1])
        dataset_tuple = create_dataset(rFile, scaleMinMax=scaleMinMax, features_list=features_list, ret_tfexs=saveRecords)
        X = dataset_tuple[0]
        Y = dataset_tuple[1]
        if saveRecords:
            tf_exs = dataset_tuple[2]
        if verbose:
            print("Got Dataset for: (" + str(month) + ", " + str(year) + ")")
            print("Timeframe = " + str(timeframe))
            print("Data Shape = " + str(X.shape))
            print("Label Shape = " + str(Y.shape))
            print("Zero Count=" + str(np.count_nonzero(X==0)))
        if make_classifier:
            model_name = "dlc" + str(timeframe) + "min_f" + str(len(features_list))
        else:
            model_name = "dlr" + str(timeframe) + "min_f" + str(len(features_list))
        if saveRecords:
            fPath = "data/tfrecords/dl_scaled_" + str(timeframe) + "_BANKNIFTY_F1_" + str(year) + "_" + str(month) + ".tfrecords"
            rec_filenames.append(save_tfrecords(tf_exs, outputfilePath=fPath, verbose=verbose))
        if model_name in models.keys():
            model_counts[model_name] = model_counts[model_name] + 1
            if verbose:
                print("Using Existing Model: " + model_name + "\nIteration = " + str(model_counts[model_name]))
        else:
            models[model_name] = model_func(num_features=len(features_list))
            model_counts[model_name] = 0
            if verbose:
                print("Created new Model: " + str(model_name))
                print(model.summary())
        savemodelname = "_".join([model_name, str(model_counts[model_name]), str(year), str(month)])
        training_history[savemodelname] = models[model_name].fit(X, Y, batch_size=32, epochs=30, verbose=1,validation_data=None)
        saveFilename = "".join([save_folder, savemodelname, ".h5"])
        models[model_name].save(saveFilename)
        model_filenames.append(saveFilename)
        if verbose:
            print("Model Saved at File: " + saveFilename)
    if verbose:
        print("Total Models Saved: " + str(model_filenames))
        if saveRecords:
            print("Tf records saved:")
            print(rec_filenames)
    if save_pickle:
        # with open("models/models_dl_5_15_30.pkl", "wb") as fHandler:
        #     pickle.dump(models, fHandler)
        if saveRecords:
            with open("data/tfrecords/scaled_recordsFilenames.pkl", "wb") as fHandler:
                pickle.dump(rec_filenames, fHandler)
        with open(save_folder + "trainingHistory_dlc_5_15_30.pkl", "wb") as fHandler:
            pickle.dump(model_filenames, fHandler)
        with open(save_folder + "modelfilenames_dlc_5_15_30.pkl", "wb") as fHandler:
            pickle.dump(model_filenames, fHandler)
        # with open(save_folder + "model_counts_dlc_5_15_30.pkl", "wb") as fHandler:
        #     pickle.dump(model_counts, fHandler)
    return models, model_filenames, model_counts

""" 
## LSTM RNN Model Keras
def LSTMRNN():
    # file is downloaded from finance.yahoo.com, 1.1.1997-1.1.2017
    # training data = 1.1.1997 - 1.1.2007
    # test data = 1.1.2007 - 1.1.2017
    input_file="repos\LSTM_RNN_Tutorials_with_Demo-master\StockPricesPredictionProject\DIS.csv"

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # fix random seed for reproducibility
    np.random.seed(5)

    # load the dataset
    df = read_csv(input_file, header=None, index_col=None, delimiter=',')
    # print(df.head())
    # exit()
    # take close price column[5]
    all_y = df[5].values
    dataset=all_y.reshape(-1, 1)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets, 50% test data, 50% training data
    train_size = int(len(dataset) * 0.5)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1, timestep 240
    look_back = 240
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
    model = Sequential()
    model.add(LSTM(25, input_shape=(1, look_back)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=1000, batch_size=240, verbose=1)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    print('testPrices:')
    testPrices=scaler.inverse_transform(dataset[test_size+look_back:])

    print('testPredictions:')
    print(testPredict)

    # export prediction and actual prices
    df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(testPrices.reshape(-1)), decimals=2)})
    df.to_csv("lstm_result.csv", sep=';', index=None)

    # plot the actual price, prediction in test data=red line, actual price=blue line
    plt.plot(testPredictPlot)
    plt.show()

 """

class dlModel():
    features_format = ["open", "high", "low", "close", "volume", "timetoexpiry", 
                "trend/sma/5", "trend/sma/20", "trend/ema/3", "trend/ema/9", "trend/ema/27", "trend/macd",
                "volatility/mband", "volatility/hband", "volatility/lband", "volatility/bwidth", "volatility/atr",
                "volume/vwap", "volume/adi", "momentum/kama", "momentum/rsi", "momentum/uo"]
    def __init__(self, filePath):
        self.modelFilepath = filePath
        self.model = load_model(filepath=self.modelFilepath)
    def create_obs(self, data):
        obs = pd.DataFrame()
        for feature in self.features_format[:5]:
            obs[feature] = data[feature]
        return obs
    def data_process(self, obs):
        return obs
    def predict_obs(self, obs):
        obs_array = self.data_process(obs)
        self.lastPrediction = self.model.predict(obs_array)
        return self.lastPrediction


tfrecords_filelist = ['data/tfrecords/dl5_BANKNIFTY_F1_2020_1.tfrecords', 
                    'data/tfrecords/dl5_BANKNIFTY_F1_2020_2.tfrecords', 
                    # 'data/tfrecords/dl5_BANKNIFTY_F1_2020_3.tfrecords', 
                    'data/tfrecords/dl5_BANKNIFTY_F1_2020_4.tfrecords', 
                    'data/tfrecords/dl5_BANKNIFTY_F1_2020_5.tfrecords', 
                    'data/tfrecords/dl5_BANKNIFTY_F1_2020_6.tfrecords', 
                    'data/tfrecords/dl5_BANKNIFTY_F1_2020_7.tfrecords', 
                    'data/tfrecords/dl15_BANKNIFTY_F1_2020_1.tfrecords', 
                    'data/tfrecords/dl15_BANKNIFTY_F1_2020_2.tfrecords', 
                    # 'data/tfrecords/dl15_BANKNIFTY_F1_2020_3.tfrecords', 
                    'data/tfrecords/dl15_BANKNIFTY_F1_2020_4.tfrecords', 
                    'data/tfrecords/dl15_BANKNIFTY_F1_2020_5.tfrecords', 
                    'data/tfrecords/dl15_BANKNIFTY_F1_2020_6.tfrecords', 
                    'data/tfrecords/dl15_BANKNIFTY_F1_2020_7.tfrecords', 
                    'data/tfrecords/dl30_BANKNIFTY_F1_2020_1.tfrecords', 
                    'data/tfrecords/dl30_BANKNIFTY_F1_2020_2.tfrecords', 
                    # 'data/tfrecords/dl30_BANKNIFTY_F1_2020_3.tfrecords', 
                    'data/tfrecords/dl30_BANKNIFTY_F1_2020_4.tfrecords', 
                    'data/tfrecords/dl30_BANKNIFTY_F1_2020_5.tfrecords', 
                    'data/tfrecords/dl30_BANKNIFTY_F1_2020_6.tfrecords', 
                    'data/tfrecords/dl30_BANKNIFTY_F1_2020_7.tfrecords']

tf_scaled_filelist = ['data/tfrecords/dl_scaled_5_BANKNIFTY_F1_2020_1.tfrecords', 
                    'data/tfrecords/dl_scaled_5_BANKNIFTY_F1_2020_2.tfrecords', 
                    'data/tfrecords/dl_scaled_5_BANKNIFTY_F1_2020_4.tfrecords', 
                    'data/tfrecords/dl_scaled_5_BANKNIFTY_F1_2020_5.tfrecords', 
                    'data/tfrecords/dl_scaled_5_BANKNIFTY_F1_2020_6.tfrecords', 
                    'data/tfrecords/dl_scaled_5_BANKNIFTY_F1_2020_7.tfrecords', 
                    'data/tfrecords/dl_scaled_15_BANKNIFTY_F1_2020_1.tfrecords', 
                    'data/tfrecords/dl_scaled_15_BANKNIFTY_F1_2020_2.tfrecords', 
                    'data/tfrecords/dl_scaled_15_BANKNIFTY_F1_2020_4.tfrecords', 
                    'data/tfrecords/dl_scaled_15_BANKNIFTY_F1_2020_5.tfrecords', 
                    'data/tfrecords/dl_scaled_15_BANKNIFTY_F1_2020_6.tfrecords', 
                    'data/tfrecords/dl_scaled_15_BANKNIFTY_F1_2020_7.tfrecords', 
                    'data/tfrecords/dl_scaled_30_BANKNIFTY_F1_2020_1.tfrecords', 
                    'data/tfrecords/dl_scaled_30_BANKNIFTY_F1_2020_2.tfrecords', 
                    'data/tfrecords/dl_scaled_30_BANKNIFTY_F1_2020_4.tfrecords', 
                    'data/tfrecords/dl_scaled_30_BANKNIFTY_F1_2020_5.tfrecords', 
                    'data/tfrecords/dl_scaled_30_BANKNIFTY_F1_2020_6.tfrecords', 
                    'data/tfrecords/dl_scaled_30_BANKNIFTY_F1_2020_7.tfrecords']

ohlcv_features = ["open", "high", "low", "close", "volume"]

if __name__ == "__main__":
    ## Train Model
    models, model_filenames, model_counts = train_model(model_func=alexNetClassifier, record_files=tf_scaled_filelist, 
                                                    features_list=priceFeatures_format+["volume"],
                                                    save_folder="models/alexNetClassifier_priceFeatures/", save_pickle=True, verbose=True)
    # alexetModel = alexNetClassifier(num_features=15)
    # print(alexetModel.summary())
    
    ## Test Data Processing
    # dataset = read_tfrecords(filepath="data/tfrecords/dl5_BANKNIFTY_F1_2020_1.tfrecords")
    # for d in dataset:
    #     print(d.head())
    #     break
    # X, Y = create_dataset(filepath="data/tfrecords/dl5_BANKNIFTY_F1_2020_1.tfrecords")
    # for x in X[0]:
    #     print(x[:3])
    ## Generate TF Records
    # tf_exs = generate_tfexamples(max_i=500)
    # save_tfrecords(tf_exs)
    # record_files = batch_tfexamples(month_list=mList, timeframe_list=timeframe_list, verbose=True)
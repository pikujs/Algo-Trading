import pandas as pd
import requests
import urllib.parse as par
from datetime import datetime
import time
import os
import zipfile

DB_URL = "http://ec2-15-207-107-183.ap-south-1.compute.amazonaws.com:9000/"
bnifty_table = "BANKNIFTY_F1"
DATA_FOLDER = "data/oneminutedata/"

file_path_list = ['data/oneminutedata/2020/JAN/IntradayData_JAN2020/BANKNIFTY_F1.txt', 
    'data/oneminutedata/2020/FEB/IntradayData_FEB2020/BANKNIFTY_F1.txt',
    'data/oneminutedata/2020/MAR/IntradayData_MAR2020/BANKNIFTY_F1.txt',
    'data/oneminutedata/2020/APR/IntradayData_APR2020/BANKNIFTY_F1.txt',
    'data/oneminutedata/2020/MAY/IntradayData_MAY2020/BANKNIFTY_F1.txt',
    'data/oneminutedata/2020/JUN/IntradayData_JUN2020/BANKNIFTY_F1.txt',
    'data/oneminutedata/2020/JUL/IntradayData_JUL2020/BANKNIFTY_F1.txt']


def query(query, verbose=False):
    if verbose:
        print(query)
    return requests.get(DB_URL + "exec?query=" + query)


def create_table(name):
    q = 'create table ' + name + ' '\
    '(timestamp timestamp, '\
    'OPEN double, '\
    'HIGH double, '\
    'LOW double, '\
    'CLOSE double, '\
    'VOLUME long, '\
    'UNKNOWN long) '\
    'timestamp(timestamp)'
    r = query(q)
    return True if r.status_code == 200 else False

def insert_table(data, table_name, verbose=False):
    tstamp = "to_timestamp('" + data["date"] + data["time"] + "', 'yyyyMMddHH:mm')"
    q = "insert into " + table_name + " values("\
        + tstamp + ", "\
        + data["open"] + ", " \
        + data["high"] + ", " \
        + data["low"] + ", " \
        + data["close"] + ", " \
        + data["volume"] + ", " \
        + data["unknown"] + ")"
    r = query(q, verbose)
    return True if r.status_code == 200 else False

def insertAll_table(data, table_name, verbose=False):
    data = data.applymap(str)
    successCount = 0
    failCount = 0
    for index, d in data.iterrows():
        if insert_table(d, table_name, verbose):
            successCount = successCount + 1
        else:
            failCount = failCount + 1
    return successCount, failCount

random_data = {"date": "20200101",
    "time": "09:08",
    "open": "1024.24",
    "high": "999.46",
    "low": "3390.6839",
    "close": "54.305",
    "volume": "2338",
    "unknown": "298576"}


dt = query("drop table " + bnifty_table)
ct = create_table(bnifty_table)

#res = insert_table(random_data, bnifty_table)
print(str(dt) + str(ct))
time.sleep(5)


def getDataFromCSV(path):
    d = pd.read_csv(path, header=None)
    d.rename(columns = {0: "instrument-name", 
            1: "date", 
            2: "time", 
            3: "open", 
            4: "high", 
            5: "low", 
            6: "close", 
            7: "volume", 
            8: "unknown"}, inplace=True)
    return d

def getDatafromFileList(file_list, verbose=False):
    master_data = pd.DataFrame(columns=["instrument-name", "date", "time" ,"open" ,"high" ,"low" ,"close", "volume", "unknown"])
    for file in file_list:
        data = getDataFromCSV(file)
        if verbose:
            print("Got data from " + file)
        master_data = master_data.append(data, ignore_index=True)
    return master_data

def getInstumentDatafromFolder(dir_path, name, verbose=False):
    master_data = pd.DataFrame(columns=["instrument-name", "date", "time" ,"open" ,"high" ,"low" ,"close", "volume", "unknown"])
    #print(master_data.head())
    for path, dir_list, file_list in os.walk(dir_path):
        for file_name in file_list:
            if file_name.endswith(name + ".txt"):
                abs_file_path = os.path.join(path, file_name)
                data = getDataFromCSV(abs_file_path)
                if verbose:
                    print("Got data from " + abs_file_path)
                master_data = master_data.append(data, ignore_index=True)
    return master_data

#data = getInstumentDatafromFolder(DATA_FOLDER + "2020/", bnifty_table)
data = getDatafromFileList(file_path_list)
print(data.head())
sc, fc = insertAll_table(data, bnifty_table, verbose=True)
print("\nSuccess count = " + str(sc) + "\nFail Count = " + str(fc))



"""
def unzipAll(dir_path):
    for path, dir_list, file_list in os.walk(dir_path):
        for file_name in file_list:
            if file_name.endswith(".zip"):
                abs_file_path = os.path.join(path, file_name)

                # The following three lines of code are only useful if 
                # a. the zip file is to unzipped in it's parent folder and 
                # b. inside the folder of the same name as the file

                parent_path = os.path.split(abs_file_path)[0]
                output_folder_name = os.path.splitext(abs_file_path)[0]
                output_path = os.path.join(parent_path, output_folder_name)

                zip_obj = zipfile.ZipFile(abs_file_path, 'r')
                zip_obj.extractall(output_path)
                zip_obj.close()

unzipAll("../../data/oneminutedata/2020/")
"""
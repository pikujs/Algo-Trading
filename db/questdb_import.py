import pandas as pd
import requests
import urllib.parse as par
from datetime import datetime
import time
import os
import csv
import zipfile

DB_URL = "http://ec2-15-207-107-183.ap-south-1.compute.amazonaws.com:9000/"

bnifty_table = "BANKNIFTY_F1"
instrumentNames = [bnifty_table, "BANKNIFTY", "NIFTY_F1", "NIFTY"]

DATA_FOLDER = "data/oneminutedata/"
month_map = {"JAN": 1, 
    "FEB": 2, 
    "MAR": 3, 
    "APR": 4, 
    "MAY": 5, 
    "JUN": 6, 
    "JUL": 7, 
    "AUG": 8, 
    "SEP": 9, 
    "OCT": 10, 
    "NOV": 11, 
    "DEC": 12}

random_data = {"date": "20200101",
    "time": "09:08",
    "open": "1024.24",
    "high": "999.46",
    "low": "3390.6839",
    "close": "54.305",
    "volume": "2338",
    "unknown": "298576"}

def getIntradayFilePathList(name, year, verbose=False): ## Get all data file paths from DATA_FOLDER with instrument-name and year
    path_list = []
    for month in os.listdir(DATA_FOLDER + str(year) + "/"):
        if os.path.isdir(DATA_FOLDER + str(year) + "/" + month):
            thisPath = DATA_FOLDER + str(year) + "/" + month + "/IntradayData_" + month + str(year) + "/" + name + ".txt"
            if os.path.isfile(thisPath):
                path_list.append((month, thisPath))
            else:
                print(thisPath + " Does not Exist!")
    if path_list:
        path_list = list(zip(*sorted(path_list, key=lambda x: month_map[x[0]])))
    if verbose:
        print(path_list)
    return list(path_list[1])

def timestampFormat(date, time): ## format of timestamp in questdb
    return "to_timestamp('" + str(date) + str(time) + "', 'yyyyMMddHH:mm')"

def store_db_format(data, fileName):  ## Stores the Data in a formatted csv file for uploading to db
    data["timestamp"] = data.apply(lambda x: timestampFormat(x["date"], x["time"]), axis=1)
    #data.drop(["instrument-name", "date", "time"], axis=1)
    print(data.head())
    data[["timestamp", "open", "high", "low", "close", "volume", "unknown"]].to_csv(fileName, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

def query(query, verbose=False): ## Simple REST API Query
    if verbose:
        print(query)
    return requests.get(DB_URL + "exec?query=" + query)


def create_table(name): ## Simple REST API Create Table by instrument-name with schema
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

def insert_table(data, table_name, verbose=False): # Insert single Data Value in existing table using Rest API Query
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

def insert_data_table(): ## Insert Data in table using /imp REST API
    pass

q_base_size = len((DB_URL + "exec?query=").encode('ascii', 'ignore'))

def insertMax_table(data, table_name, verbose=False): ## Inser Max Values in 1800 get query length in one GET Request Query
    data = data.applymap(str)
    q = "".join(["insert into ", table_name, " values "])
    max_i = 5
    sc = fc = 0
    firstVal = True
    for index, d in data.iterrows():
        tstamp = "to_timestamp('" + d["date"] + d["time"] + "', 'yyyyMMddHH:mm')"
        val = "(" \
            + tstamp + ", "\
            + d["open"] + ", " \
            + d["high"] + ", " \
            + d["low"] + ", " \
            + d["close"] + ", " \
            + d["volume"] + ", " \
            + d["unknown"] + ")"
        if firstVal:
            firstVal = False
        else:
            val = ", " + val
        q = q + val
        query_size = q_base_size + len(q.encode('ascii', 'replace'))
        if query_size > 1800:
            r = query(q, verbose)
            q = "".join(["insert into ", table_name, " values "])
            firstVal = True
            if r.status_code == 200:
                sc = sc + 1
            else:
                fc = fc + 1
    return sc, fc

def insertAll_table(data, table_name, verbose=False): ## Insert all data into table using single Query for each
    data = data.applymap(str)
    successCount = 0
    failCount = 0
    for index, d in data.iterrows():
        if insert_table(d, table_name, verbose):
            successCount = successCount + 1
        else:
            failCount = failCount + 1
    return successCount, failCount

def recreate_table(name, verbose=False): ## Drop Table and Recreate with schema
    dt = query("drop table " + bnifty_table)
    ct = create_table(bnifty_table)
    time.sleep(3)
    if verbose:
        if dt.status_code == 200:
            print(name + " Table droppeed")
        if ct:
            print(name + " Table Created")
    return ct and (dt.status_code == 200)

def getDataFromCSV(path): ## import Data from raw csv
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

def getDatafromFileList(file_list, verbose=False): ## get DataFrame from file List
    master_data = pd.DataFrame(columns=["instrument-name", "date", "time" ,"open" ,"high" ,"low" ,"close", "volume", "unknown"])
    for file in file_list:
        data = getDataFromCSV(file)
        if verbose:
            print("Got data from " + file)
        master_data = master_data.append(data, ignore_index=True)
    if verbose:
        print(master_data.head())
    return master_data

def getDatafromFolder(dir_path, name, verbose=False): ## Get DataFrame from folder
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


def create_full_table(name, year, verbose=False): ## Create full table and insert values from imported csv data
    if verbose:
        print("Creating Full Table " + name)
    recreate_table(name, verbose=verbose)
    file_path_list = getIntradayFilePathList(name, year, verbose=verbose)
    data = getDatafromFileList(file_path_list, verbose=verbose)
    sc, fc = insertMax_table(data, bnifty_table, verbose=verbose)
    if verbose:
        print("\nSuccess count = " + str(sc) + "\nFail Count = " + str(fc))

#for iName in instrumentNames[0:1]:
#file_path_list = getIntradayFilePathList(bnifty_table, "2020", True)
#data = getDatafromFileList(file_path_list, True)
create_full_table(bnifty_table, "2020", True)
    #store_db_format(data, DATA_FOLDER + "2020/" + iName + "_small.csv")

#data = getDatafromFolder(DATA_FOLDER + "2020/", bnifty_table)
#data.to_csv(DATA_FOLDER + "2020/BNIFTY_F1_2020.csv", index=False)
#maxinsert = insertMax_table(data, bnifty_table, verbose=True)


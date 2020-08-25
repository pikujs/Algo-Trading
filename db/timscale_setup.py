import os
import requests
import json
import csv
from datetime import datetime
import calendar
import pandas as pd
import numpy as np
import psycopg2
from pgcopy import CopyManager

bnifty_table = "BANKNIFTY_F1"
instrumentNames = ["BANKNIFTY", "NIFTY"]
futuresNames = [bnifty_table, "BANKNIFTY_F2", "NIFTY_F1", "NIFTY_F2"]
schema = [("datetime", "TIMESTAMP NOT NULL"), 
    ("InternalName", "VARCHAR(25)"), 
    ("open", "DOUBLE PRECISION"), 
    ("high", "DOUBLE PRECISION"), 
    ("low", "DOUBLE PRECISION"), 
    ("close", "DOUBLE PRECISION"), 
    ("volume", "INT"), 
    ("unknown", "INT"), 
    ("ExpiryDate", "TIMESTAMP"), 
    ("exchange", "VARCHAR(8)")]

username = "postgres"
password = "timepa$$"
hostname = "db.pikujs.com"
port = 5432
database_name = "ohlcvdata"

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

def counter_map(month_count):
    for code, count in month_map.items():
        if count == month_count:
            return code

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

def get_data(name, year, verbose=False):
    file_path_list = getIntradayFilePathList(name, year, verbose=verbose)
    return getDatafromFileList(file_path_list, verbose=verbose)

def insert_query(table_name):
    return "INSERT INTO " + table_name + " VALUES (%(datetime)s, %(internalName)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(unknown)s, %(expiryDate)s, %(exchange)s);"

def connection_str(uname, pword, host, p, dbname):
    return "".join(["postgres://", uname, ":", pword, "@", host, ":", str(p), "/", dbname])

def query(query_str, data, verbose=False):
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    cur = conn.cursor()
    cur.execute(query_str, data)
    if verbose:
        print(query_str)
        print(cur.statusmessage)
    conn.commit()
    cur.close()
    if cur.description:
        return cur.fetchall()


def create_new_table(name, schema, verbose=False):
    q = "DROP TABLE IF EXISTS " + name + "; CREATE TABLE " + name + "( "
    sch = ", ".join([" ".join([col, dtype]) for (col, dtype) in schema])
    q = q + sch  + ");"
    q_hyper = "SELECT create_hypertable('" + name + "', '" + schema[0][0] + "');"
    if verbose:
        print(q)
        print(q_hyper)
    #conn = psycopg2.connect(connection_str(username, password, hostname, password, database_name))
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    cur = conn.cursor()
    try:
        cur.execute(q)
        cur.execute(q_hyper)
    except (Exception, psycopg2.Error) as error:
        print(error.pgerror)
    conn.commit()
    cur.close()

def insert_row(table_name, data, verbose=False):
    q = insert_query(table_name)
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    cur = conn.cursor()
    cur.execute(q, data)
    if verbose:
        print(q)
        print(cur.statusmessage)
    conn.commit()
    cur.close()

def insert_many(table_name, data, verbose=False):
    q = insert_query(table_name)
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    cur = conn.cursor()
    if verbose:
        print("Connected to db.\nTrying to many insert " + str(len(data)) + "rows in Table " + table_name)
    cur.executemany(q, data)
    if verbose:
        print(cur.statusmessage)
    conn.commit()
    cur.close()

def insert_batch(table_name, data, verbose=False):
    q = insert_query(table_name)
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    cur = conn.cursor()
    if verbose:
        print("Connected to db.\nTrying to batch insert " + str(len(data)) + "rows in Table " + table_name)
    psycopg2.extras.execute_batch(cur, q, data)
    if verbose:
        print(cur.statusmessage)
    conn.commit()
    cur.close()


def fast_insert(table_name, data, schema, verbose=False):
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    mgr = CopyManager(conn, table_name, list(zip(*schema))[0])
    try:
        mgr.copy(data)
    except Exception as e:
        print(e)

def get_full_table(table_name):
    sql_query = "SELECT * FROM " + table_name
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    return pd.read_sql_query(sql_query, conn)

def last_thursday(month, year):
    return max(week[-4] for week in calendar.monthcalendar(year, month))

def prepare_futures_Data(data, verbose=False):
    pData = []
    if data["instrument-name"][1][-2] != 'F':
        print("Warning: Data might not be futures data.")
    suffix_count = int(data["instrument-name"][1][-1])
    if verbose:
        print("Processing Futures Data for " + data["instrument-name"][1])
    for index, d in data.iterrows():
        exp_month = int(d["date"]/100)%100
        exp_year = int(d["date"]/10000)
        exp_date = last_thursday(exp_month, exp_year)
        month_offset = 1 if (d["date"]%100) <= exp_date else 0
        exp_month = exp_month + suffix_count - month_offset
        if exp_month > 12:
            exp_month = exp_month - 12
            exp_year = exp_year + 1
        exp_date = last_thursday(exp_month, exp_year)
        exp_datetime = " ".join(["-".join([str(exp_year), str(exp_month), str(exp_date)]), "23:59"])
        parsed_datetime = " ".join(["-".join([str(d["date"])[:4], str(d["date"])[4:6], str(d["date"])[6:8]]), str(d["time"])])
        intName = " ".join([d["instrument-name"][:-3], counter_map(exp_month), "FUT"])
        pData.append({
            'datetime': parsed_datetime, 
            'internalName': intName,
            'open': d["open"], 
            'high': d["high"], 
            'low': d["low"], 
            'close': d["close"], 
            'volume': d["volume"], 
            'unknown': d["unknown"],
            'expiryDate': exp_datetime,
            'exchange': "NSE"})
    if verbose:
        print(pData[:3])
    return pData


for iName in [futuresNames[1], futuresNames[3]]:
    create_new_table(iName, schema, verbose=True)
    rawdata = get_data(iName, "2020", False)
    insert_batch(iName, prepare_futures_Data(rawdata), True)
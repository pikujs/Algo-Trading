
import requests
import json
import csv
from datetime import datetime
import questdb_import
import pandas as pd
import numpy as np
import psycopg2
from pgcopy import CopyManager

bnifty_table = "BANKNIFTY_F1"
instrumentNames = [bnifty_table, "BANKNIFTY", "NIFTY_F1", "NIFTY"]
schema = [("datetime", "TIMESTAMP WITH TIME ZONE NOT NULL"), 
    ("open", "DOUBLE PRECISION"), 
    ("high", "DOUBLE PRECISION"), 
    ("low", "DOUBLE PRECISION"), 
    ("close", "DOUBLE PRECISION"), 
    ("volume", "INT"), 
    ("unknown", "INT"), 
    ("exchange", "VARCHAR(8)")]

username = "postgres"
password = "timepa$$"
hostname = "db.pikujs.com"
port = 5432
database_name = "ohlcvdata"

insert_query = "INSERT INTO BANKNIFTY_F1 VALUES ('%(datetime)s', %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(unknown)s, '%(exchange)s');"

def connection_str(uname, pword, host, p, dbname):
    return "".join(["postgres://", uname, ":", pword, "@", host, ":", str(p), "/", dbname])

def query(query_str, verbose=False):
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
    q = "INSERT INTO " + table_name + " VALUES (%(datetime)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(unknown)s, %(exchange)s);"
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    cur = conn.cursor()
    cur.execute(q, data)
    if verbose:
        print(q)
        print(cur.statusmessage)
    conn.commit()
    cur.close()

def insert_many(table_name, data, verbose=False):
    q = "INSERT INTO " + table_name + " VALUES (%(datetime)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(unknown)s, %(exchange)s);"
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    cur = conn.cursor()
    cur.executemany(q, data)
    if verbose:
        print(q)
        print(cur.statusmessage)
    conn.commit()
    cur.close()

def insert_batch(table_name, data, verbose=False):
    q = "INSERT INTO " + table_name + " VALUES (%(datetime)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(unknown)s, %(exchange)s);"
    conn = psycopg2.connect(host=hostname, port=port, user=username, password=password, database=database_name)
    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, q, data)
    if verbose:
        print(q)
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

def prepareData(data):
    pData = []
    for index, d in data.iterrows():
        parsed_datetime = " ".join(["-".join([str(d["date"])[:4], str(d["date"])[4:6], str(d["date"])[6:8]]), str(d["time"])])
        pData.append({
            'datetime': parsed_datetime, 
            'open': d["open"], 
            'high': d["high"], 
            'low': d["low"], 
            'close': d["close"], 
            'volume': d["volume"], 
            'unknown': d["unknown"],
            'exchange': "NSE"})
    return pData


for iName in instrumentNames[1:]:
    create_new_table(iName, schema, verbose=True)
    rawdata = questdb_import.get_data(iName, "2020", True)
    insert_batch(iName, prepareData(rawdata), True)
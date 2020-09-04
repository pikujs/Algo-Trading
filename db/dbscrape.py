# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:53:07 2020

@author: Akil
"""
import psycopg2
import pandas as pd
import sys


def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError):
        sys.exit(1) 
    return conn



def postgresql_to_dataframe(conn, select_query, column_names):

    cursor = conn.cursor()
    try:
        cursor.execute(select_query)
    except (Exception, psycopg2.DatabaseError):
        cursor.close()
        return 1
    
    tupples = cursor.fetchall()
    cursor.close()
    
    df = pd.DataFrame(tupples, columns=column_names)
    df.sort_values(by = ['datetime'],inplace = True)
    return df


def gettablerange(host,database,username,password,tablename,startdt,enddt):

    param_dic = {
        "host"      : host,
        "database"  : database,
        "user"      : username,
        "password"  : password
    }
    startdt = '\'' + str(startdt) + '\''
    enddt = '\'' + str(enddt) + '\''

    conn = connect(param_dic)
    
    column_names = ["datetime","internaltime","open","high","low","close","volume","unknown","expirydate","exchange"]
    df = postgresql_to_dataframe(conn, "SELECT * FROM public."+tablename+" WHERE datetime between "+startdt+" AND "+enddt, column_names)
    return df

def gettable(host,database,username,password,tablename):
    
    param_dic = {
    "host"      : host,
    "database"  : database,
    "user"      : username,
    "password"  : password
    }
    
    conn = connect(param_dic)
    
    column_names = ["datetime","internaltime","open","high","low","close","volume","unknown","expirydate","exchange"]
    df = postgresql_to_dataframe(conn, "SELECT * FROM public."+tablename, column_names)
    return df
    
def expirymonth(host,database,username,password,tablename,month,year):
    
    param_dic = {
        "host"      : host,
        "database"  : database,
        "user"      : username,
        "password"  : password
    }
    
    m_str = str(month)
    next_m = str(month+1)
    next_yr = str(year)
    if month<9:
        m_str = "0" + str(month)
        next_m = "0" + str(month+1)
    elif month == 9:
        m_str = "09"
        next_m = "10"
    elif month == 12:
        m_str = "12"
        next_m = "01"
        next_yr = str(year+1)
    

    startdt = '\''+str(str(year) +'-'+m_str+'-01'+'  09:14:00')+'\''
    enddt = '\''+str(next_yr +'-'+next_m+'-01'+'  09:14:00')+'\''
    # enddt = '\''+str(str(year) +'-'+m_str+'-31'+'  23:59:00')+'\''
    conn = connect(param_dic)
    
    column_names = ["datetime","internalname","open","high","low","close","volume","unknown","expirydate","exchange"]
    df = postgresql_to_dataframe(conn, "SELECT * FROM public."+tablename+" WHERE expirydate between "+startdt+" AND "+enddt, column_names)
    return df

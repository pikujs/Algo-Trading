# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 09:29:44 2020

@author: Akil
"""
import pandas as pd
from dbscrape import *

def minutetoyear(startmonth,endmonth,startyear,endyear):
    
    dffinal = pd.DataFrame(columns = ['date','open','high','low','close','volume'])
    
    year = startyear
    
    for month in range(startmonth,endmonth+1):
        df = getmonth("db.pikujs.com","ohlcvdata","postgres","timepa$$","banknifty_f1",month,year)
        pd.to_datetime(df['datetime'])
        df = df.set_index('datetime') 
        df = df[['open','high','low','close','volume']]
        
        dffinal = dffinal.append(minutetomonth(month,year,df),ignore_index = True)

    dffinal['date'] = pd.to_datetime(dffinal['date'], format="%d-%m-%Y")
    dffinal = dffinal.set_index('date')

    return dffinal



def minutetomonth(month,year,df):
    
    dft = pd.DataFrame(columns = ['date','open','high','low','close','volume'])
    i = 0
    
    startday = df.index[0].day
    endday = df.index[-1].day

    for day in range(startday,endday):
        
        high = 0
        low =  9999999999999
        volume = 0
        open1 = 0
        close1 = 0
        count = 0
        
        while((df.index[i].day)==day):
            
            if count==0:
                open1 = df['open'][i]
                count+=1
                
    
            high = max(high,df['high'][i])
            low = min(low,df['low'][i])
            volume+=df['volume'][i]
            
            i+=1
                    
        close1 = df['close'][i-1]
            
        if(open1!=0):
            dft = dft.append({'date': str(day)+'-'+str(month)+'-'+str(year), 'open' : open1,'high':high,'low':low,'close':close1,'volume':volume} , ignore_index=True)
                
    return dft

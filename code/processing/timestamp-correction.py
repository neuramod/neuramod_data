# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 

@author: Qureshi
"""
import csv
import pandas as pd

eeg = pd.read_csv('EEG.csv')
# Step 1:  Remove jitter from start time value
eeg["estimated-time-zero"] = eeg["timestamp"]-(eeg["sequence"]+1)*31250

#Step 2:  Correct for time of flight
median= eeg.median()
ctf = eeg.median()[36]-40000 #subtract time of flight recorded by bitbrain viewer

#Step 3:  Calculate corrected timestamp for all blocks
eeg["adjusted-block-time"] = ctf+(eeg["sequence"]*31250)

newdf = eeg[['timestamp', 'estimated-time-zero','adjusted-block-time' ]]

#Step 4:  Calculate the timestamp for each sample
data=[]
constant = 3906.25

for i in range(0,len(df),8):
    
    a=df.iloc[i] 
    data = np.hstack((data,a))
    b=a+constant  
    data = np.hstack((data,b))
    c=b+constant  
    data = np.hstack((data,c))    
    d=c+constant  
    data = np.hstack((data,d))
    e=d+constant   
    data = np.hstack((data,e))
    f=e+constant  
    data = np.hstack((data,f))
    g=f+constant   
    data = np.hstack((data,g))
    h=g+constant  
    data = np.hstack((data,h))

eeg["adjusted-block-time-sample"] = data.astype('int64')

newdf = eeg[['timestamp', 'estimated-time-zero','adjusted-block-time', 'adjusted-block-time-sample' ]]

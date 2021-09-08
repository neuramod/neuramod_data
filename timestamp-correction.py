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
ctf = eeg.median()[36]-40000

#Step 3:  Calculate corrected timestamp for all blocks
eeg["adjusted-block-time"] = ctf+(eeg["sequence"]*31250)

newdf = eeg[['timestamp', 'estimated-time-zero','adjusted-block-time' ]]

#Step 4:  Calculate the timestamp for each sample


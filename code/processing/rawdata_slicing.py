# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:31:30 2021

@author: Qureshi
"""
import numpy as np
import pandas as pd

df = pd.read_csv('Photodiode2.csv')

# Slicing 
df['f'] = df.Photodiode.astype(str).str[:2].astype(int)
# Extracting paradigm starting value 
a = df[df.f >= 31]
a.reset_index(level=0, inplace=True)
value = a.iloc[0, 0]
initial_corrected_data = df.iloc[value:,0:5]

#%%

df = pd.read_csv('EEG.csv')
result_EEG = df.iloc[value:,:]

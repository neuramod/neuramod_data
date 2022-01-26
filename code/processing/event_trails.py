# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:06:02 2022

@author: neuramod
"""

import pandas as pd
#%%

#Adding trails in an event file

df = pd.read_csv("0092_S000_T001_events.csv")
df["trail"]=""
ntimes=200
list=[1,2,3,4]
df['trail']= [i for i in list for _ in range(ntimes)]

# saving event csv file
df.to_csv('0092_S000_T001_events.csv', encoding='utf-8')
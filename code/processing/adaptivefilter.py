# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 20:07:12 2022

@author: neuramod
"""
import numpy as np
import padasip as pa

input1=np.zeros((32,106880))
ref=np.zeros((106880))
for i in range(32):
    input0=filt._data[[i],:].transpose()
        for j in range(32):
        if j!=i:
            r=filt._data[j,:]
            ref=ref+r
    ref=ref/31

    f = pa.filters.FilterLMS(n=1, mu=0.01, w="random")
    y, e, w = f.run(ref,input0)
    output0[i,:]=y


filt._data=output0
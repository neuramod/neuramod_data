# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 15:41:28 2022

@author: neuramod
"""
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne import create_info, concatenate_raws
from mne.io import RawArray
import numpy as np

#%%
the_stream = []
df = pd.read_csv("0092_S000_T001.csv")
df_IMU =pd.read_csv("IMU.csv")

#%%
# IMU's reading, normalizing and adding to data frame
df_IMU.rename({'IMU_B-ch1': 'LA X', 'IMU_B-ch2': 'LA Y', 'IMU_B-ch3':'LA Z', 'IMU_B-ch4': 'AA X', 'IMU_B-ch5':'AA Y', 'IMU_B-ch6': 'AA Z'}, axis=1, inplace=True)
df_IMU['Sum AA'] = df_IMU['AA X'] + df_IMU['AA Y'] + df_IMU['AA Z']
Z= df_IMU['Sum AA'].repeat(8)
N = 3
Z.drop(index=Z.index[:N], 
        axis=0, 
        inplace=True)
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

Z = NormalizeData(Z)

df1=pd.DataFrame(Z)
df1= df1.reset_index()
df1=df1.drop('index', axis=1)
ext = df1["Sum AA"]

df=df.join(ext)
df_mne = df.drop(['timestamp', 'sequence', 'battery', 'flags'], axis=1)
data = df_mne.to_numpy().transpose()
data1 = np.vstack((data[:-1]/1000000, data[-1]))

#%%
#creating RawArray

ch_names= ['Fp1', 'Fpz', 'Fp2', 'AF3','AF4', 'FCz', 'F3', 'Fz', 'F4', 'CPz', 'PO3', 'FC1', 'FC2','PO4', 'PO7', 'P3', 'Cz', 'C4', 'PO8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'C3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2','SAA']
ch_types = ['eeg'] * 32 + ['bio']
sfreq=256
ch_name=['SAA']
ch_type= ['bio']

the_info = create_info(ch_names=ch_names, ch_types=ch_types,sfreq=sfreq)
the_stream.append(RawArray(data=data1, info=the_info))
#%%
# concatenating
raws = concatenate_raws(the_stream)

#%%

## ploting raw data
raw_data = raws._data

print("Number of channels: ", str(len(raw_data)))
print("Number of samples: ", str(len(raw_data[0])))

plt.plot(raw_data[0,:106880])
plt.title("Raw EEG, Fp1, samples 0-4999")
plt.show()

#%%

## Monataging and applying filter
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
raw = raws.copy().set_montage(ten_twenty_montage)
plot_fig = raw.plot(scalings= 'auto')
plot_psd_fig = raw.plot_psd(average=True)
plot_sensors_fig = raw.plot_sensors(kind = '3d', show_names=True)
plot_sensors_topo_fig = raw.plot_sensors(kind='topomap', show_names=True)
pEEG = raw.plot(start=0., duration=100., scalings=0.5e-2, remove_dc=True, n_channels=33)


filt=raw.filter(0.1, 30, method='iir', iir_params=dict(order=8, ftype='butter', output='sos', picks='eeg'))

pEEG = filt.plot(start=0., duration=50., scalings=0.5e-2, remove_dc=True, n_channels=32)
plot_psd_fig = filt.plot_psd(average=False)

#%%

# Pre and Post filter visualization

channel_names = ['AF3', 'O2']
two_eeg_chans = filt[channel_names, 0:512]
y_offset = np.array([5e-5, 0])  # just enough to separate the channel traces
x = two_eeg_chans[1]
y = two_eeg_chans[0].T + y_offset
lines = plt.plot(x, y)
plt.legend(lines, channel_names)



channel_names = ['AF3', 'O2']
two_eeg_chans = raw[channel_names, 0:512]
y_offset = np.array([5e-5, 0])  # just enough to separate the channel traces
x = two_eeg_chans[1]
y = two_eeg_chans[0].T + y_offset
lines = plt.plot(x, y)
plt.legend(lines, channel_names)





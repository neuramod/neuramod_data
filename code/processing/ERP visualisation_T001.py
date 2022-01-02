# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:21:39 2021

@author: neuramod
"""

import mne
import pybv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab, seaborn as sns
from scipy.stats import ttest_rel, sem
from mne.preprocessing import  Xdawn, ICA
from mne.decoding import CSP
from mne import compute_raw_covariance
import scipy
from scipy.signal import butter, lfilter, filtfilt
from mne.viz import plot_epochs_image

#%% 
df = pd.read_csv("00605_S000_T001.csv")
df_mne = df.drop(['timestamp', 'sequence', 'battery', 'flags'], axis=1)
data = df_mne.to_numpy().transpose()
data = data/1000000
channels = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC1', 'FCz', 'FC2', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CPz', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
unit= ['µV'] * 32

#%%
# Events marking
event = pd.read_csv("events.csv")
events = np.array(event)
#%%
#Converting to BrainVision Format
pybv.write_brainvision(data=data, sfreq=256, ch_names=channels,
                  folder_out='./analysis',
                  fname_base='00563_S000_T001', events=events,
                  unit=unit)

#%%
#Reading from brain vision
exp = mne.io.read_raw_brainvision('00605_S000_T001.vhdr', preload=True)

#%%
#montage
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
raw_1020 = exp.copy().set_montage(ten_twenty_montage)
plot_fig = raw_1020.plot(scalings= 'auto')
plot_psd_fig = raw_1020.plot_psd(average=True)
plot_sensors_fig = raw_1020.plot_sensors(kind = '3d', show_names=True)
plot_sensors_topo_fig = raw_1020.plot_sensors(kind='topomap', show_names=True)

pEEG = raw_1020.plot(start=0., duration=420., scalings=0.5e-2, remove_dc=True, n_channels=32)
#%%
### High/lowpass filter
'''
lowcut=0.1
highcut=30
fs=256
n_channels=32
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
order=5
b, a = butter(order, [low, high], 'bandpass')
y= filtfilt(b, a, raw_1020, axis=-1)
'''
filt=raw_1020.filter(0.1, 30, method='iir')


#fmin, fmax = 0.1, 30
#filt = raw_1020.filter(l_freq=fmin, h_freq=fmax, )
timeEEG = filt.plot(start=0., duration=420., scalings=0.5e-4, remove_dc=True, n_channels=32)
freqEEG = filt.plot_psd(fmax = 30)

#%%
### Applying notch filter
pl_freq=50.
ny_freq=128.

nth = filt.notch_filter(np.arange(pl_freq, ny_freq, pl_freq), fir_design='firwin')
timeEEG1 = nth.plot(start=0., duration=420., scalings=0.5e-4, remove_dc=True, n_channels=32)
freqEEG1 = nth.plot_psd(fmax=30)
#%%
### Referencing

ref_data= nth.set_eeg_reference(ref_channels='average')
timeEEG2 = ref_data.plot(start=0., duration=420., scalings=0.5e-4, remove_dc=True, n_channels=32)
freqEEG2 = ref_data.plot_psd(fmax=30)
#%%
#Events
events,event_ids = mne.events_from_annotations(ref_data)
print(events,event_ids)

#%%
## Mappping epochs
mapping = {4: '4', 1:'1', 0: '0', 8: '8', 2: '2',
           90:'90', 37:'37', 40:'40', 7:'7', 9:'9', 5:'5', 
           32:'32', 6:'6', 38:'38'}
annot_from_events = mne.annotations_from_events(
    events=events, event_desc=mapping, sfreq=exp.info['sfreq'],
    orig_time=ref_data.info['meas_date'])
ref_data.set_annotations(annot_from_events)
ref_data.plot(start=110, duration=20, n_channels=32)
fig = mne.viz.plot_events(events, sfreq=ref_data.info['sfreq'])
event_id={'0': 0, '1': 1, '2': 2, '4': 4, '5': 5, 
 '6': 6, '7': 7, '8': 8, '9': 9, '32': 32, 
 '37': 37, '38': 38, '40': 40, '90': 90}
epochs = mne.Epochs(ref_data, events, event_id=event_id, baseline=None, preload=True)
fig = epochs.plot(events=events, n_channels=32)
print(epochs)
#verbose=False
epochs.average().plot()

### Average ERP of the three experimental condition at given channel 
target = epochs['0']
nontarget = epochs['90']
dist = epochs['1','2','32','37','38','4','40','5','6','7','8','9']
ch = 0  ## Channel for visualization
conditions = ['target', 'nontarget', 'dist']

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Time Instances')
ax.set_ylabel('Volt')

ax.plot(target.average().data[ch, :], color='blue', label='target')
ax.plot(nontarget.average().data[ch, :], color='red', label='nontarget')
ax.plot(dist.average().data[ch, :], color='green', label='dist')

legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
plt.title('ERP of different conditions')
plt.show()

#%%
### Mean Event-Related Potential on given channel
def plot_conditions(data, times, plot_title):

    sns.set(style="white")
    ColorsL = np.array(([228,26,28], [55,126,184], [77,175,74], [152,78,163], [255,127,0]))/256
    col_axes = np.array((82, 82, 82))/256

    al = 0.2
    fig = plt.figure(num=None, figsize=(4, 2), dpi=150)

    
    epochs_mean = np.mean(data, axis = 0)
    epochs_std = sem(data, axis = 0)/2

    plt.plot(times, epochs_mean, color = ColorsL[0], linewidth = 2)
    plt.fill_between(times, epochs_mean, epochs_mean + epochs_std, color = ColorsL[0], interpolate=True, alpha = al)
    plt.fill_between(times, epochs_mean, epochs_mean - epochs_std, color = ColorsL[0], interpolate=True, alpha = al)
    plt.ylabel('Mean ERP')
    plt.xlabel('Times')
    plt.title(plot_title)

plot_conditions(epochs.get_data()[:,31,:], epochs.times, 'Channel O2')

#%% 
## Averaging Epochs

target = epochs['0'].average()
fig1 = target.plot()
fig2 = target.plot(spatial_colors=True)
target.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)
target.plot_joint()
target.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))
#non-target
nontarget = epochs['90'].average()
fig3= nontarget.plot()
fig = nontarget.plot(spatial_colors=True)
nontarget.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)
nontarget.plot_joint()
nontarget.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))

dist = epochs['1','2','32','37','38','4','40','5','6','7','8','9'].average()
dist.plot_joint()
dist.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)
dist.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))

all_evoked = [target, nontarget, dist]
print (all_evoked)
 
all_evoked = [epochs[cond].average() for cond in sorted(event_id.keys())]
print(all_evoked)

grand_average = mne.grand_average(all_evoked)
print(grand_average)

evokeds = dict(target= target, nontarget=nontarget, dist=dist)
picks = [n for i,n in enumerate(channels)]
mne.viz.plot_compare_evokeds(evokeds, picks=picks, show_sensors=True)


evokeds = dict(target=list(epochs['0'].iter_evoked()),
               nontarget=list(epochs['90'].iter_evoked()), 
                  dist=list(epochs['1','2','32','37','38','4','40','5','6','7','8','9'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, picks=picks)


#%%
#CSP Filter
epochs_data = epochs.get_data()
n_channels = epochs_data.shape[1]
n_components = 4
csp = CSP(n_components=n_components, norm_trace=False)
csp.fit(epochs_data, epochs.events[:, -1])
y = epochs.events[:, -1]
X = csp.fit_transform(epochs_data, y)
csp.plot_filters(epochs.info, ch_type='eeg', size=1.5)

csp.filters_ = csp.filters_.T
csp.plot_filters(epochs.info, ch_type='eeg', size=1.5)

#%%
#XDAwn-filter

picks = mne.pick_types(ref_data.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
signal_cov = compute_raw_covariance(ref_data, picks=picks)

# Xdawn instance
xd = Xdawn(n_components=4, signal_cov=signal_cov)
xd.fit(epochs)
epochs_denoised= xd.apply(epochs)
plot_epochs_image(epochs_denoised['0'], vmin=-500, vmax =500)

#%% 
### Apply ICA
ica = ICA(n_components=30, max_iter= 'auto', random_state=97)
ica.fit(ref_data)

ica.plot_sources(ref_data, show_scrollbars=False)
ica.plot_components()

#ica.exclude=[0,1,2, 3, 4, 6,7,  8, 10,11,12,15,17,20, 27, 23, 21, 25, 27, 28]
#ica.exclude=[0,1,2, 3, 4, 6,7,11, 13, 15,19,17,20,21,25,28]
ica.exclude=[0,1,2, 3, 4, 6,7,8, 9, 10, 11, 13, 15,16, 18,17,19,20, 21,22,23,24,27, 25, 26]
reconst_raw = ref_data.copy()
data= ica.apply(reconst_raw)
ica.plot_overlay(data)
timeEEG1 = data.plot(start=0., duration=420., scalings=0.5e-4, remove_dc=True, n_channels=32)
freqEEG1 = data.plot_psd(fmax=30)

#%% Averaging epochs post ICA

epochs = mne.Epochs(data, events, event_id=event_id, baseline=None)
fig = epochs.plot(events=events, n_channels=32)
print(epochs)

reject_criteria = dict(eeg=100e-6)
_ = epochs.drop_bad(reject=reject_criteria)
epochs.plot_drop_log()
print(epochs)


target = epochs['0'].average()
fig1 = target.plot()
fig2 = target.plot(spatial_colors=True)
target.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)
target.plot_joint()
target.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))
#non-target
nontarget = epochs['90'].average()
fig3= nontarget.plot()
fig = nontarget.plot(spatial_colors=True)
nontarget.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)
nontarget.plot_joint()
nontarget.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))

dist = epochs['1','2','32','37','38','4','40','5','6','7','8','9'].average()
dist.plot_joint()
dist.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)
dist.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))

all_evoked = [target, nontarget, dist]
print (all_evoked)
 
all_evoked = [epochs[cond].average() for cond in sorted(event_id.keys())]
print(all_evoked)

grand_average = mne.grand_average(all_evoked)
print(grand_average)

evokeds = dict(target= target, nontarget=nontarget, dist=dist)
picks = [n for i,n in enumerate(channels)]
mne.viz.plot_compare_evokeds(evokeds, picks=picks, show_sensors=True)


evokeds = dict(target=list(epochs['0'].iter_evoked()),
               nontarget=list(epochs['90'].iter_evoked()), 
                  dist=list(epochs['1','2','32','37','38','4','40','5','6','7','8','9'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, picks=picks)
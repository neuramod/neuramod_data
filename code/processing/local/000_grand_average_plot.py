# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:48:57 2022

@author: neuramod
"""

import mne
mne.set_log_level('error')  # reduce extraneous MNE output
import matplotlib.pyplot as plt
import numpy as np
import glob

#%%
conditions = ['target', 'ntarget']
#%%
data_dir = 'evoked'
data_files = glob.glob(data_dir + '/0*_S*-ave.fif' )
data_files

#%%
evokeds = {}

for idx, c in enumerate(conditions):
    evokeds[c] = [mne.read_evokeds(d)[idx] for d in data_files]
evokeds
#%%
channels =['Fp1','Fpz','Fp2','AF3','AF4','FC1','FCz','FC2','F3','Fz','F4','CP1','CP2','CPz','CP5','CP6',
           'C3','Cz','C4','P3','P4','Pz','P7','P8','PO3','PO4','POz','PO7','PO8','O1','Oz','O2']
for i in channels:
    mne.viz.plot_compare_evokeds(evokeds, combine= 'mean', picks=i);
#%%
# Define plot parameters
roi = ['O1','Oz','O2']

color_dict = {'target':'blue', 'ntarget':'red'}
linestyle_dict = {'target':'-', 'ntarget':'-'}


aa=mne.viz.plot_compare_evokeds(evokeds,
                             combine='mean',
                             legend='lower right',
                             picks=roi, show_sensors='upper right',
                             colors=color_dict,
                             linestyles=linestyle_dict,
                             title='target vs. ntarget')
plt.show()
#%%
### --- Difference wave 'Target-NonTarget' --- ###

diff_waves = []
for i in range(len(data_files)):
    diff_waves.append(mne.combine_evoked([evokeds['target'][i], evokeds['ntarget'][i]], weights=[1, -1]) )
    
evokeds.append(diff_waves)
diff_waves

contrast = 'Difference wave: target-ntarget'
bb=mne.viz.plot_compare_evokeds({contrast:diff_waves}, combine='mean',
                            legend=None,
                            picks=roi, show_sensors='upper right',
                            title=contrast
                            )
plt.show()

diff_evok =[]
for i in diff_waves:
    x = (i._data)
    diff_evok.append(x)


evk = evokeds.append('diff_waves')



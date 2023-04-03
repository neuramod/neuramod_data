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
data_dir = 'evoked'
data_files = glob.glob(data_dir + '/0*_S*-ave.fif' )
evokeds = {}
for idx, c in enumerate(conditions):
    evokeds[c] = [mne.read_evokeds(d)[idx] for d in data_files]
evokeds
channel = evokeds['target'][0]
channel = channel.ch_names
for i in channel:
    mne.viz.plot_compare_evokeds(evokeds, combine= 'mean', picks=i);
#%%
# Plot grand average w.r.t roi
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



# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:47:45 2022

@author: neuramod
"""

from meegkit.asr import ASR
#%%
## Artifact Subspace Reconstruction (ASR)
ref_data_uncleaned = ref_data.copy()

asr = ASR(sfreq=ref_data.info["sfreq"], cutoff=5)
asr.fit(ref_data._data)
ref_data._data = asr.transform(ref_data._data)

scalings= {"eeg":2e-5}

print("Uncleaned data")
ref_data_uncleaned.plot(start=10, scalings=scalings);

print("Cleaned data")
ref_data.plot(start=10, scalings=scalings);

timeEEG1 = ref_data_uncleaned.plot(start = 0., duration = 30., scalings=0.5e-4, remove_dc=True, n_channels=32)
timeEEG2 = ref_data.plot(start = 0., duration = 30., scalings=0.5e-4, remove_dc=True, n_channels=32)

psds = ref_data.psd(fmax=30);
ref_data.plot_psd(fmax=30);
ref_data_uncleaned.plot_psd(fmax=30);
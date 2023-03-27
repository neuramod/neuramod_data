
import pandas as pd
import numpy as np
import mne
from pyedflib import highlevel



# write an edf file

signalss = pd.read_csv("EEG.csv")
df_mne = signalss.drop(['timestamp', 'sequence', 'battery', 'flags'], axis=1)
signals = df_mne.to_numpy().transpose()
# channel names according to the layout
channels_names = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F3', 'Fz', 'F4', 'FC1', 'FCz', 
					'FC2', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CPz', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 
					'P4', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
signal_headers = highlevel.make_signal_headers(channels_names, sample_rate=256, physical_max=500000, 
													physical_min=-500000, prefiler='NaN')
header = highlevel.make_header(patientname='1285', gender='Female', equipment='BitBrain')
highlevel.write_edf('edf_eeg.edf', signals, signal_headers, header)
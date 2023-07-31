# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:40:41 2022

@author: neuramod
"""
import mne
import pybv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import os

#%%
def initial_processing(the_data):
    the_data_path = os.path.join(the_drive,os.sep,the_repo,the_subrepo,the_type, the_data)
    the_data_slicing = os.path.join(the_drive,os.sep,the_repo,the_subrepo,the_type, the_data_slice)
    the_data_raw = os.path.join(the_drive,os.sep,the_repo,the_subrepo,the_type,  data_raw)
    ch_layout = os.path.join(the_drive,os.sep,the_repo,the_subrepo,the_type,  ch_config)

    
    df = pd.read_csv(the_data_path)
    df.rename(columns={'rotation': 'event_label'}, inplace=True)
    event_label = df['event_label'].dropna()

    event_start = int(256 * 17.5)
    sfreq=256
    stim = 0.3
    n_stim = 300
    stim_len = math.ceil(sfreq * stim)
    rest=10

    x = []
    for trial_number in range(1, 6):
        trial_start = int((rest * sfreq) + x[-1]) if trial_number > 1 else event_start
        trial_length = int(trial_start + stim * sfreq * n_stim)
        for i in range(trial_start, trial_length + 1, stim_len):
            x.append(i)

    events = pd.DataFrame({'event_sequence': x, 'event_label': event_label})
    events['trial'] = np.repeat(np.arange(1, 6), n_stim)
    events.to_csv('00674_S027_T001_events.csv', index=False, encoding='utf-8')   
    
    # Slicing 
    df_slice = pd.read_csv(the_data_slicing)
    print(df_slice.head())
    df_slice['f'] = df_slice['Photodiode-ch1'].astype(str).str[:2].astype(int)
    # Extracting peak value 
    a = df_slice[df_slice.f >= 31]   # peak value depend on visual inspection of photodiode
    a.reset_index(level=0, inplace=True)
    value = a.iloc[0, 0]
    initial_corrected_data = df_slice.iloc[value:,0:5]
    df_raw = pd.read_csv(the_data_raw)
    result_EEG = df_raw.iloc[value:,:]
    result_EEG.to_csv('00674_S027_T001.csv',index=False, encoding='utf-8')
    
    # Data to brain vision format
    
    df_mne = result_EEG
    df_mne = df_mne.drop(['timestamp', 'sequence', 'battery', 'flags'], axis=1)
    df_mne["EEG"] = df_mne['EEG-ch1'] 

    layout_raw = pd.read_excel(ch_layout, index_col=0)

    ch_names = []
    eeg_cols = [col for col in df_mne.columns if 'EEG' in col]
    for i in range(len(eeg_cols)):
        the_name = layout_raw.index[layout_raw['Channel number'] == i+1].tolist()[0].replace('0','O')
        ch_names.append(the_name)

    data = df_mne.to_numpy().transpose()
    data = np.vstack(data/1000000)
    ch_types = ['eeg'] * 32 + ['eog'] * 1
    unit= ['µV'] * 33
    #event = pd.read_csv("00674_S027_T001_events.csv")
    events_int = np.array(events).astype(int)

    pybv.write_brainvision(data=data, sfreq=256, ch_names=ch_names,
                  folder_out='./', 
                  fname_base='00674_S027_T001', events=events_int,
                  unit=unit, overwrite=True)    
    
####################################################################################################
#
# SCRIPT CALL OR MODULE IMPORT
####################################################################################################
if __name__ == '__main__':


    data_name = "00674_S027_T001_rsvp_paradigm_2022_Dec_19_1036"
    layout = "Layout"
    data_slice = "Photodiode"
    the_raw_data = "EEG"
    the_drive = "D:"
    the_repo = "data"
    the_subrepo = "data_00674"
    the_type = "00674_S027_T001"
    the_id = f"{data_name.split('_')[0]}"
    
    
    the_data = f"{data_name}.csv"
    the_data_slice = f"{data_slice}.csv"
    data_raw=f"{the_raw_data}.csv"
    ch_config= f"{layout}.xlsx"
    initial_processing(the_data)



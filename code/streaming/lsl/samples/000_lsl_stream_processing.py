# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:36:24 2023

@author: neuramod
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pylsl
from mne_realtime import LSLClient, MockLSLStream
import mne
import sys
from pylsl import StreamInlet, resolve_stream, resolve_byprop
from icecream import ic
import math
import mne
import pandas as pd
import numpy as np
import random
import neurokit2 as nk
from utils import *


cutoff = 3100000
sfreq= 256
BUFFER_LENGTH= 90
EPOCH_LENGTH = 1
len_stimuli = 0.3
no_stimuli = 300
chunks = math.ceil(len_stimuli  * sfreq)
trail_length = math.ceil(len_stimuli  * sfreq)* no_stimuli
eeg_event = np.zeros((int(no_stimuli), 1))
stim_event=[]
eeg_emp= []

if __name__ == '__main__':
    print("looking for a photodiode stream...")
    photo_stream = resolve_stream('type', 'FOT')
    photo_inlet = StreamInlet(photo_stream[0]) 
    photo_time_correction = photo_inlet.time_correction()
    eeg_stream = resolve_byprop('type', 'EEG')
    eeg_inlet = StreamInlet(eeg_stream[0], max_chunklen=77)
    eeg_time_correction = eeg_inlet.time_correction()
    #info = pylsl.stream_info('marker_stream','Markers', 1, channel_format='int32', source_id= 'rsvp_marker_stream') 
    #outlet = pylsl.stream_outlet(info)  
    streams = resolve_stream('type','Markers')
    marker_inlet = StreamInlet(streams[0]) 
    marker_time_correction = marker_inlet.time_correction()

    info = eeg_inlet.info()
    n_channels = info.channel_count()
    channels = ['Ch %i' % i for i in range(n_channels)]
    eeg_buffer = np.zeros((int(trail_length), 32))
    filter_state = None



    
    #print('Press Ctrl-C in the console to break the while loop.')
    try:
        while True:
            sample, timestamp = photo_inlet.pull_sample()
            #ic(timestamp,  np.round(sample, 2)) x.astype(int)
            #samples = ic(timestamp, np.round(sample, 1))
            sample=np.round(sample, 1)
            sample = int(sample)
            
            ############# Start of 1st trial  #############
            if sample >= cutoff:
                print("looking for a EEG stream...")
                while True:
                    event, timestamp = marker_inlet.pull_sample()
                    chunk, timestamp = eeg_inlet.pull_chunk( timeout = 0.4,max_samples = chunks)#timeout = 0.3,
                    ch_data = np.array(chunk)
                    event_data = np.array(event)
                    stim_event = update_event_len(stim_event, event_data)
                    #stim_event = update_event_len(stim_event, event)
                    eeg_buffer, filter_state = update_buffer(
                            eeg_buffer, ch_data, notch=True,
                                   filter_state=filter_state)
                    eeg_event = event_buffer(eeg_event, event_data)             
                                    
                    ############# Start of 2nd trial  #############

                    if len(stim_event) == no_stimuli:
                        print("looking for a photodiode stream...")
                        classification = processing(eeg_event, eeg_buffer)
                        while True:
                            photo_sample, timestamp = photo_inlet.pull_sample()
                            photo_sample=np.round(photo_sample, 1)
                            photo_sample = int(photo_sample)  
                            eeg_event = np.zeros((int(no_stimuli), 1))
                            eeg_buffer = np.zeros((int(trail_length), 32))
                            stim_event_t2 = []
                        
                            if photo_sample >= cutoff:
                                print("looking for a EEG stream...")
                                while True:
                                    samples, timestamp = marker_inlet.pull_sample()
                                    chunk, timestamp = eeg_inlet.pull_chunk( timeout = 0.4,max_samples = chunks)
                                    ch_data = np.array(chunk)
                                    event_data = np.array(samples)
                                    stim_event_t2 = update_event_len(stim_event_t2, event_data)
                                    eeg_event = event_buffer(eeg_event, event_data)
                                    
                                    eeg_buffer, filter_state = update_buffer(
                                            eeg_buffer, ch_data, notch=True,
                                                filter_state=filter_state)

                        ############# Start of 3rd trial  #############

                                    if len(stim_event_t2) == no_stimuli:
                                        print("looking for a photodiode stream...")
                                        classification = processing(eeg_event, eeg_buffer)
                                        while True:
                                            photo_sample, timestamp = photo_inlet.pull_sample()
                                            photo_sample=np.round(photo_sample, 1)
                                            photo_sample = int(photo_sample)  
                                            eeg_event = np.zeros((int(no_stimuli), 1))
                                            eeg_buffer = np.zeros((int(trail_length), 32))
                                            stim_event_t3 = []

                                            if photo_sample >= cutoff:
                                                print("looking for a EEG stream...")
                                                while True:
                                                    samples, timestamp = marker_inlet.pull_sample()
                                                    chunk, timestamp = eeg_inlet.pull_chunk( timeout = 0.4,max_samples = chunks)
                                                    ch_data = np.array(chunk)
                                                    event_data = np.array(samples)
                                                    stim_event_t3 = update_event_len(stim_event_t3, event_data)
                                                    eeg_event = event_buffer(eeg_event, event_data)
                                                    
                                                    eeg_buffer, filter_state = update_buffer(
                                                            eeg_buffer, ch_data, notch=True,
                                                            filter_state=filter_state)
                                                
                            ############# Start of 4th trial  #############
                        
                                                    if len(stim_event_t3) == no_stimuli:
                                                        print("looking for a photodiode stream...")
                                                        classification = processing(eeg_event, eeg_buffer)
                                                        while True:
                                                            photo_sample, timestamp = photo_inlet.pull_sample()
                                                            photo_sample=np.round(photo_sample, 1)
                                                            photo_sample = int(photo_sample)  
                                                            eeg_event = np.zeros((int(no_stimuli), 1))
                                                            eeg_buffer = np.zeros((int(trail_length), 32))
                                                            stim_event_t4 = []

                                                            if photo_sample >= cutoff:
                                                                print("looking for a EEG stream...")
                                                                while True:
                                                                    samples, timestamp = marker_inlet.pull_sample()
                                                                    chunk, timestamp = eeg_inlet.pull_chunk( timeout = 0.4,max_samples = chunks)
                                                                    ch_data = np.array(chunk)
                                                                    event_data = np.array(samples)
                                                                    stim_event_t4 = update_event_len(stim_event_t4, event_data)
                                                                    eeg_event = event_buffer(eeg_event, event_data)
                                                                    
                                                                    eeg_buffer, filter_state = update_buffer(
                                                                            eeg_buffer, ch_data, notch=True,
                                                                                filter_state=filter_state)
                                                                    #ic(np.round(eeg_buffer, 1))
                                                                    #ic(np.round(stim_event_t4, 1))
                                                                    if len(stim_event_t4) == no_stimuli:
                                                                        print("looking for a photodiode stream...")
                                                                        classification = processing(eeg_event, eeg_buffer)
                                                                        samples.stop()
                                                                        chunk.stop()
                                                                        photo_sample.stop()
                
                

    except KeyboardInterrupt:
        print('Closing!')



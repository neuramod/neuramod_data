# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:05:12 2022

@author: neuramod
"""
import os
import mne
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

from scipy.signal import sosfiltfilt, butter
from mne.decoding import cross_val_multiscore
from mne.time_frequency import tfr_morlet
from sklearn.model_selection import  KFold
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report,confusion_matrix,ConfusionMatrixDisplay)

#%%
'''
            #####   Pre and post processing pipeline     ########

'''
def processing(the_data):
    
    the_data_path = os.path.join(the_drive,os.sep,the_folder,the_repo,the_subrepo,the_bids,the_type, the_sub, the_eeg, the_data)
    exp = mne.io.read_raw_brainvision(the_data_path,preload=True)
    #exp.set_channel_types(mapping={'EOG': 'eeg'})
    #exp.rename_channels(mapping={'EOG': 'EOG'})
    #exp.set_channel_types(mapping={'EOG': 'eog'})
    print(exp.info)
    exp.plot_psd()
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw_1020 = exp.copy().set_montage(ten_twenty_montage)
    bads, info = nk.eeg_badchannels(raw_1020, bad_threshold=0.5,distance_threshold=0.96, show=True)
    raw_1020.info['bads'] = bads
    print(raw_1020.info)
    ## band-pass filter ##
    filt=raw_1020.filter(0.1, 20,method='iir', iir_params=dict(order=5, ftype='butter', output='sos', picks='eeg', exclude='bads'))
    pltEEG = filt.plot(start=200., duration=10.,  remove_dc=True, n_channels=32)
    ## notch filter ##
    pl_freq=50.
    ny_freq=128.
    nth = filt.notch_filter(np.arange(pl_freq, ny_freq, pl_freq), fir_design='firwin')
    ## detrending ##
    b = nth._data
    sos = butter(20, 0.1, output='sos')
    y = sosfiltfilt(sos, b)
    nth._data = y
    ## interpolation ##
    nth.interpolate_bads(reset_bads=True)
    ## reference ##
    ref_data= nth.set_eeg_reference(ref_channels='average')
    pEEG = ref_data.plot(start=200., duration=10.,  remove_dc=True, n_channels=32)
    freqEEG = ref_data.plot_psd(fmin=0.1,fmax=20)
    ## event annotation ##
    events,event_ids = mne.events_from_annotations(ref_data)

    ## Epoching ##
    
    mapping = {0: 'target:0', 
           90:'ntarget:90'} 
    annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=exp.info['sfreq'],
                    orig_time=ref_data.info['meas_date'])
    ref_data.set_annotations(annot_from_events)
    event_id={'target:0': 0, 'ntarget:90': 90} 
    print(event_id)
    picks = mne.pick_types(ref_data.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
    epochs = mne.Epochs(ref_data, events, event_id=event_id, tmin=-0.2, tmax=0.8, 
                    baseline=(-0.2, 0), preload=True,reject=dict(eeg=250e-6),reject_by_annotation='bad',picks=picks) #, eog =100e-6
    print(epochs)
    print('sample drop %: ', (1 - len(epochs.events)/len(events)) * 100)

    ## Evoked response plot ##

    epochs.apply_baseline(baseline=(-0.2, 0))
    evoked = epochs.average()
    evokeds = [epochs.average() for cond in ['target', 'ntarget']]
    mne.write_evokeds(f"{data_name}-ave.fif", evokeds)
    evoked_target = epochs['target:0'].average() 
    evoked_ntarget = epochs['ntarget:90'].average()
    evoked_target.plot(gfp=True, spatial_colors=True)
    evoked_ntarget.plot(gfp=True, spatial_colors=True)
    evoked_target.plot_joint()
    evoked_ntarget.plot_joint()
    picks = [ 'Pz', 'POz'] 
    evokeds = dict(target=list(epochs['target:0'].iter_evoked()),natrget=list(epochs['ntarget:90'].iter_evoked()))
    mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks=picks) 

    ## Time frequency analysis --- pipeline --- ##

    freqs = np.arange(4, 21, 4)
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = tfr_morlet(epochs['target:0'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=3, n_jobs=None)
    for i in range(0,32):
        power.plot([i], baseline=(-0.5, 0), mode='mean', title= 'Target TFR channel: ' + power.ch_names[i]).close()
    power = tfr_morlet(epochs['ntarget:90'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=3, n_jobs=None)
    for i in range(0,32):
        power.plot([i], baseline=(-0.5, 0), mode='mean', title= 'Non-target TFR channel: ' + power.ch_names[i])

    ## XdawnCov + TS+ SVC --- pipeline --- ##

    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1]
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    pr = np.zeros(len(labels))
    print("Multiclass classification with Xdawn + TS + SVC")
    clf = make_pipeline(XdawnCovariances(nfilter=3, applyfilters=True, classes=None, 
            estimator='lwf', xdawn_estimator='lwf', baseline_cov=None), TangentSpace(), SVC())
    for train_idx, test_idx in cv.split(epochs_data, labels):
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(epochs_data[train_idx], y_train)
        pr[test_idx] = clf.predict(epochs_data[test_idx])
    print(classification_report(labels, pr))
    names = ["target:0","ntarget:90"]  #
    cm = confusion_matrix(labels, pr)
    ConfusionMatrixDisplay(cm, display_labels=names).plot()
    plt.show()
    ## CSP + SVC --- pipeline --- ##
    print("Multiclass classification with CSP + SVC")
    csp = CSP(n_components=3, reg='shrinkage',norm_trace=False, rank="full") #, rank="full" , reg= 'shrinkage'
    clf_csp = make_pipeline(csp,
    SVC(C=1, kernel='linear'))
    scores = cross_val_multiscore(clf_csp, epochs_data, labels, cv=cv, n_jobs=None)
    print('CSP: %0.1f%%' % (100 * scores.mean(),))
    ## LR, RF, NB, SVC, Ensemble --- pipeline --- ##
    print("Multiclass classification with LR, RF, NB, SVC, Ensemble")
    test = epochs_data[:,:,0]
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    clf4 = SVC()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svc', clf4)], voting='hard')
    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Support vector machine','Ensemble']):
        scores = cross_val_score(clf, test, labels, scoring='accuracy', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    ## Peak amplitude and latency w.r.t specific time window ##

    print('Peak amplitude and latency w.r.t specific time window')
    channels = ['Pz','P3','P4','POz','PO3','PO4', 'PO7', 'Oz']
    def print_peak_measures(ch, tmin, tmax, lat, amp):
        print(f'Channel: {ch}')
        print(f'Time Window: {tmin * 1e3:.3f} - {tmax * 1e3:.3f} ms')
        print(f'Peak Latency: {lat * 1e3:.3f} ms')
        print(f'Peak Amplitude: {amp * 1e6:.3f} µV')
    picks = [n for i,n in enumerate(channels)]
    for i in picks:
        evoked_target_roi = evoked_target.copy().pick(i)
        good_tmin, good_tmax = 0.3, 0.6
        ch, lat, amp = evoked_target_roi.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, return_amplitude=True) #mode='pos',
        print('** PEAK MEASURES FROM TARGET EVOKED w.r.t TIME WINDOW **')
        print_peak_measures(ch, good_tmin, good_tmax, lat, amp)
    for i in picks:
        evoked_ntarget_roi = evoked_ntarget.copy().pick(i)
        good_tmin, good_tmax = 0.3, 0.6
        ch, lat, amp = evoked_ntarget_roi.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, return_amplitude=True) #mode='pos',
        print('** PEAK MEASURES FROM NON-TARGET w.r.t TIME WINDOW **')
        print_peak_measures(ch, good_tmin, good_tmax, lat, amp)
        
####################################################################################################
# SCRIPT CALL OR MODULE IMPORT
####################################################################################################
if __name__ == '__main__':


    data_name = "sub-6103_task-T001_eeg"
    the_drive = "D:"
    the_folder= "neuramod_data"
    the_repo = "data"
    the_subrepo = "raw"
    the_bids = "bids"
    the_type = "6103_P000_S000_T001"
    the_sub= "sub-6103"
    the_eeg= "eeg"
    
    the_id = f"{data_name.split('_')[0]}"
    
    
    the_data = f"{data_name}.vhdr"
    processing(the_data)
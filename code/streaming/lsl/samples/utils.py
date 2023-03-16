# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:57:16 2023

@author: neuramod
"""

import mne
import pandas as pd
import numpy as np
import random
import neurokit2 as nk

from scipy.signal import butter, lfilter, lfilter_zi
from scipy.signal import sosfiltfilt, butter
from mne.decoding import cross_val_multiscore
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
chunks = 77
no_stimuli = 300
stim = 0.3
sfreq= 256

NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')

def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
            new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer
def update_event_len(event_buffers, new_datas):
                
    event = np.append([event_buffers], [new_datas]) 
    event = event
    return event

    
def event_buffer(event_buffer, new_data):
                
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, event_buffer.shape[1])

    new_event = np.concatenate((event_buffer, new_data), axis=0) 
    new_event = new_event[new_data.shape[0]:, :]
    return new_event

def processing(events, eeg_data):
    event_label = events
    
    x=[]
    trial_rest= 0
    trial_lenght= int(stim * no_stimuli * sfreq)
    for i in  range(trial_rest, trial_lenght+1,chunks):
        x.append(i)
    
    df = pd.DataFrame(event_label)
    df.rename(columns = {0:'event_label'}, inplace = True)
    event_label= df['event_label']
    events=event_label.to_frame()
    events['event_sequence'] = x
    events = events[["event_sequence", "event_label"]]
    
    eeg = pd.DataFrame(eeg_data)

    print(eeg.columns)
    eeg.columns=["EEG"+str(i) for i in range(0, 32)]

    eeg_cols = [col for col in eeg.columns if 'EEG' in col]

    ch_names=['Fp1','Fpz','Fp2','AF3','AF4','FCz','F3','Fz',
              'F4','CPz','PO3','FC1','Cz','PO4','PO7','P3',
              'FC2','C4','PO8','CP5','CP1','CP2','CP6','P7',
              'C3','Pz','P4','P8', 'POz','O1','Oz','O2']

    ch_types = ['eeg']*(len(eeg_cols))

    the_info = mne.create_info(ch_names=ch_names, ch_types=ch_types,sfreq=sfreq)
    the_stream = []
    data = eeg.to_numpy().transpose()
    data_mne = np.vstack((data*1e-6))
    the_stream=mne.io.RawArray(data=data_mne, info=the_info).set_montage( mne.channels.make_standard_montage('standard_1020'))
    # band pass
    bads, info = nk.eeg_badchannels(the_stream, bad_threshold=0.5,distance_threshold=0.96, show=True)
    the_stream.info['bads'] = bads
    print(the_stream.info)
    ## band-pass filter ##
    filt=the_stream.filter(0.1, 20,method='iir', iir_params=dict(order=5, ftype='butter', output='sos', picks='eeg', exclude='bads'))
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
    
    onset =  np.arange(0, 90, 0.3).tolist()
    duration =  np.arange(0, 90, 0.3).tolist()
    description = events['event_label']

    new_annotations = mne.Annotations(onset, duration,
                                      description)
    ref_data.set_annotations(new_annotations)
    events,event_ids = mne.events_from_annotations(ref_data)

    print(len(event_ids))
    if len((event_ids)) == 12:
        mapping = {1: 'target:0', 
           12:'ntarget:90'} 
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=ref_data.info['sfreq'],
                    orig_time=ref_data.info['meas_date'])
        ref_data.set_annotations(annot_from_events)
        event_id={'target:0': 1, 'ntarget:90': 12} 
        picks = mne.pick_types(ref_data.info, meg=False, eeg=True, stim=False, exclude='bads')
        epochs = mne.Epochs(ref_data, events=events,  event_id=event_id,  tmin=-0.2,   tmax= 0.8, 
                        baseline=None, reject=dict(eeg=150e-6), preload=True,verbose=False, picks=picks)
        ## XdawnCov + TS+ SVC --- pipeline --- ##
        epochs_data = epochs.get_data()
        labels = epochs.events[:, -1]
        # Define a monte-carlo cross-validation generator (reduce variance):
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
        
    else:
         mapping = {1: 'target:0', 
            11:'ntarget:90'} 
         annot_from_events = mne.annotations_from_events(
             events=events, event_desc=mapping, sfreq=ref_data.info['sfreq'],
                     orig_time=ref_data.info['meas_date'])
         ref_data.set_annotations(annot_from_events)
         event_id={'target:0': 1, 'ntarget:90': 11} 
         picks = mne.pick_types(ref_data.info, meg=False, eeg=True, stim=False, exclude='bads')
         epochs = mne.Epochs(ref_data, events=events,  event_id=event_id,  tmin=-0.2,   tmax= 0.8, 
                         baseline=None, reject=dict(eeg=150e-6), preload=True,verbose=False, picks=picks)   
         print(event_id)
         ## XdawnCov + TS+ SVC --- pipeline --- ##
         epochs_data = epochs.get_data()
         labels = epochs.events[:, -1]
         # Define a monte-carlo cross-validation generator (reduce variance):
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
    return cm
    
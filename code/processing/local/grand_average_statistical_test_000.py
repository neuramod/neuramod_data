# Import libraries
import mne
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.stats as stats
mne.set_log_level('error')  

def perform_ttest_and_grand_average_plot(data_files, conditions, roi, tmin, tmax):
    evokeds = {}
    for idx, c in enumerate(conditions):
        evokeds[c] = [mne.read_evokeds(d)[idx] for d in data_files]
    target_evokeds = evokeds['target:0']
    non_target_evokeds = evokeds['distractors']
    mean_amplitude_target = []
    mean_amplitude_ntarget = []
    channels = roi
    for i in target_evokeds:
        for j in channels:
            target = i.copy().pick(j).crop(tmin=0.30, tmax=0.60)
            mean_amp_roi = target.data.mean(axis=1) * 1e6
            mean_amplitude_target.append(mean_amp_roi[0]) 
    for i in non_target_evokeds:
        for j in channels:
            nontarget = i.copy().pick(j).crop(tmin=0.30, tmax=0.60)
            mean_amp_roi = nontarget.data.mean(axis=1) * 1e6
            mean_amplitude_ntarget.append(mean_amp_roi[0])  
    # Perform independent (two-sample) t-test
    t_value, p_value = stats.ttest_ind(
        mean_amplitude_target,
        mean_amplitude_ntarget,
        equal_var=False  
    )
    print("T-test results:")
    print(f"p-value = {p_value:.2e}")
    roi = roi 
    color_dict = {'target:0':'blue', 'distractors':'red'}
    linestyle_dict = {'target:0':'-', 'distractors':'-'}
    fig, ax = plt.subplots(figsize=(8, 6))
    mne.viz.plot_compare_evokeds(
        evokeds,
        combine='mean',
        legend='lower left',
        picks=roi,
        show_sensors='upper left',
        colors=color_dict,
        linestyles=linestyle_dict,
        show=False,  
        invert_y=None,
        title='Grand average waveform of channels',
        ylim=dict(eeg=[-4, 6]),
        axes=ax,  # Pass the axes to the plotting function
    )
    ax.axvspan(tmin +0.05, tmax, color='lightgreen', alpha=0.3)
    diff_waves = [mne.combine_evoked([evokeds['target:0'][subj], 
                                  evokeds['distractors'][subj]
                                 ],
                                 weights=[1, -1]
                                 ) 
              for subj in range(len(data_files))
              ]
    diff_waves
    contrast = 'Difference waveform target-distractors'
    mne.viz.plot_compare_evokeds({contrast:diff_waves}, combine='mean',
                            legend=None,
                            picks=roi, show_sensors='upper left',
                            title= contrast,
                            ylim=dict(eeg=[-4, 6])
                            )
    plt.show()
    
if __name__ == '__main__':
    data_dir = 'D:/data/data_testing/evoked'
    data_files = glob.glob(data_dir + '/*-ave.fif' )
    conditions = ['target:0','distractors']
    roi = ['Pz'] 
    tmin = 0.30
    tmax= 0.60
    perform_ttest_and_grand_average_plot(data_files,conditions, roi, tmin, tmax)

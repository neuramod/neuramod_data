Home repo page: [LINK](https://github.com/neuramod/neuramod_data)

# processing
> ## local
#### 000_csvtoedfconversion.py : Convert Raw Data from a EEG.csv file into EDF format for visual inspection [Link](https://github.com/neuramod/neuramod_data/tree/main/code/processing/local)
 1. [download edf viewer setup](https://www.teuniz.net/edfbrowser/)
 2. first convert EEG.csv to edf format using the script
 3. afterwards preview the recorded data

<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227936682-26f48a44-cedc-48a0-b40e-995391fec5f2.PNG"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/87472076/227936773-6e0d2d2d-b93f-4f29-bff6-b6b292a21ab7.PNG"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/87472076/227936900-23d824ee-4170-49ed-8ed6-8be509e648d3.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


#### 000_raw_processing.py
* event labeling
* EEG data slicing using photodiode cutoff value
* convert EEG data to brain vision format

#### 000_processing_pipeline.py
 * the script includes pre and post processing pipeline
 * initially standard 10-20 montage is applied
 ```
 ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
 raw_1020 = exp.copy().set_montage(ten_twenty_montage)
 fig = raw_1020.plot_sensors(show_names=True, ch_type='eeg')
 ```
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227962429-a429092c-352d-4d66-b2d9-b6172bdc90b3.png"  alt="" width = 50% height = auto></td>
</tr>
</table>


* channel rejection is applied having the parameter values of bad_threshold=0.5 and distance_threshold=0.96
```
bads, info = nk.eeg_badchannels(raw_1020, bad_threshold=0.5,distance_threshold=0.96, show=True)
```
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227963405-56faeebc-8c77-4630-82a5-848e30f91340.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


*  the signal is bandpass filtered between 0.1 and 20 Hz using fifth order infinite impulse response (IIR) Butterworth filter. To remove power line noise in the continuous EEG data a notch filter was applied with a stopband of 50 Hz
```
filt=raw_1020.filter(0.1, 20,method='iir', iir_params=dict(order=5, ftype='butter', output='sos', picks='eeg', exclude='bads'))
pl_freq=50.
ny_freq=128.
nth = filt.notch_filter(np.arange(pl_freq, ny_freq, pl_freq), fir_design='firwin')
```
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227967783-d7963cb8-cb09-41a6-b372-b0c04e3fe2d4.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


* detrending and bad channel interpolation is applied to the continous eeg data. Afterwards, reference is applied accross the channels (average).
```
b = nth._data
sos = butter(20, 0.1, output='sos')
y = sosfiltfilt(sos, b)
nth._data = y
eeg_data_interp = nth.copy().interpolate_bads(reset_bads=True)
ref_data= eeg_data_interp.set_eeg_reference(ref_channels='average')
```
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227969877-7375ef7d-1545-4975-ac3c-2732cb25adbd.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


#### 000_grand_average_plot.py
* import libraries
```
import mne
mne.set_log_level('error')  # reduce extraneous MNE output
import matplotlib.pyplot as plt
import numpy as np
import glob
```
* plot grand average
```
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
```
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/229495622-fd2f9e1e-23ff-404c-bfd4-043163c393bc.png"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/87472076/229495638-0cdde271-7503-4eff-b529-f16df2efb132.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


* create difference waves to more easily visualize the difference between conditions
```
diff_waves = []
for i in range(len(data_files)):
    diff_waves.append(mne.combine_evoked([evokeds['target'][i], evokeds['ntarget'][i]], weights=[1, -1]) )
diff_waves

contrast = 'Difference wave: target-ntarget'
bb=mne.viz.plot_compare_evokeds({contrast:diff_waves}, combine='mean',
                            legend=None,
                            picks=roi, show_sensors='upper right',
                            title=contrast
                            )
plt.show()
```
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/229495022-99f55e87-e228-4a2c-aaa7-3576af0a508f.png"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/87472076/229495045-50408221-5461-40ba-8d01-8753e244a561.png"  alt="" width = 100% height = auto></td>
</tr>
</table>



> ## openbis
#### 002_fromRawDatasetToRawBidsFormat.ipynb
* Script converts raw eeg data set to BIDS format, data slicing and assign events.
* To validate the BIDS format please click on the link and upload the BIDS folder. [BIDS dataset to validate](https://bids-standard.github.io/bids-validator/)
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227949996-0254c3fb-03a3-4912-a84d-58fc23ce715d.png"  alt="" width = 100% height = auto></td>
</tr>
</table>





Home repo page: [LINK](https://github.com/neuramod/neuramod_data)

# processing
## local
#### 000_csvtoedfconversion.py
* [download edf viewer setup](https://www.teuniz.net/edfbrowser/)
* first convert EEG.csv to edf format using the script
* afterwards preview the recorded data


<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227936682-26f48a44-cedc-48a0-b40e-995391fec5f2.PNG"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/87472076/227936773-6e0d2d2d-b93f-4f29-bff6-b6b292a21ab7.PNG"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/87472076/227936900-23d824ee-4170-49ed-8ed6-8be509e648d3.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


#### 000_processing_pipeline.py
* the script includes pre and post processing pipeline
* initially standard 10-20 monatge is applied
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227962429-a429092c-352d-4d66-b2d9-b6172bdc90b3.png"  alt="" width = 50% height = auto></td>
</tr>
</table>


* channel rejection is applied having the parameter vaules of bad_threshold=0.5 and distance_threshold=0.96
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227963405-56faeebc-8c77-4630-82a5-848e30f91340.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


*  the signal is bandpass filtered between 0.1 and 20 Hz using fifth order infinite impulse response (IIR) Butterworth filter. To remove power line noise in the continuous EEG data a notch filter was applied with a stopband of 50 Hz
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227967783-d7963cb8-cb09-41a6-b372-b0c04e3fe2d4.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


* detrending, bad interpolation is applied to the continous eeg data. Afterwards, reference is applied accross the channels (average).
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227969877-7375ef7d-1545-4975-ac3c-2732cb25adbd.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


## openbis
#### 002_fromRawDatasetToRawBidsFormat.ipynb
* Script converts raw eeg data set to BIDS format, data slicing and assign events. In order to validate the BIDS format please click on the link and upload the BIDS folder. [BIDS dataset to validate](https://bids-standard.github.io/bids-validator/)
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227949996-0254c3fb-03a3-4912-a84d-58fc23ce715d.png"  alt="" width = 100% height = auto></td>
</tr>
</table>





Home repo page: [LINK](https://github.com/neuramod/neuramod_data)
## Previewing recorded data
* Before uploading raw data to the OpenBis server, you can preview the recorded session a decide if it is ok or if it should be redone
...

## Datasets formatting and uploading
* for any new data saved locally, save the new data folder under the main standard folder for the three different data types:
> BLED: Bluetooth EEG recordings

    Z:\neuramod_data\data\raw\eeg\O+MATERIALS+NRMD+NRMD_UPLD_RAWD_BLED+RAWD_BLED\OPENBIS_UPLOADS
    
> SDCD: SD card EEG recordings

    Z:\neuramod_data\data\raw\eeg-sd\O+MATERIALS+NRMD+NRMD_UPLD_RAWD_SDCD+RAWD_SDCD\OPENBIS_UPLOADS
    
> STIM: Stimulation recordings

    Z:\neuramod_data\data\raw\stim\O+MATERIALS+NRMD+NRMD_UPLD_RAWD_STIM+RAWD_STIM\OPENBIS_UPLOADS
    
* rename the folder with the naming standards
`<ID>_<PROJECT_PHASE>_<SESSION_ID>_<GENERATED_DATAFOLDER_NAME>` (eg. 0092_P000_S000_trial_2021_Oct_12_1241 for 'STIM', 0092_P000_S000_BBT-E32-AAB052-2021-10-12_12-40-45 for 'BLED', 0092_P000_S000_BBT-E32-AAB052-2021-10-12_12-40-45 for 'SDCD').


<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/3306992/212129515-8c343b98-2890-4371-8795-90c8644c8871.png"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/3306992/212129743-00e23bf0-55dc-4202-b3ba-a0752d464e4c.png"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/3306992/212129877-427235b1-b189-4c8c-9dd2-081003527782.png"  alt="" width = 100% height = auto></td>
</tr>
</table>


* Zip the folder without nesting and do not leave the original (manually or with sample code)
```
import os
import shutil

BASE_FOLDERS = ["Z:\\neuramod_data\\data\\raw\\eeg\\O+MATERIALS+NRMD+NRMD_UPLD_RAWD_BLED+RAWD_BLED\\OPENBIS_UPLOADS",
                "Z:\\neuramod_data\\data\\raw\\eeg-sd\\O+MATERIALS+NRMD+NRMD_UPLD_RAWD_SDCD+RAWD_SDCD\\OPENBIS_UPLOADS",
                "Z:\\neuramod_data\\data\\raw\\stim\\O+MATERIALS+NRMD+NRMD_UPLD_RAWD_STIM+RAWD_STIM\\OPENBIS_UPLOADS"]
for _d in BASE_FOLDERS:
    subdirs = os.listdir(_d)
    print(f"found {len(subdirs)} folders in {_d}:")
    for i,_sd in enumerate(subdirs):
        print(f"zipping {i+1}/{len(subdirs)}: {_sd}")
        shutil.make_archive(os.path.join(_d,_sd), 'zip', os.path.join(_d,_sd))
        print(f"deleting original folder: {os.path.join(_d,_sd)}")
        shutil.rmtree(os.path.join(_d,_sd)) 
```
### SETUP MOBAXTERM
* [Download MOBAXTERM](https://mobaxterm.mobatek.net/)
* Launch MOBAXTERM
* Make sure to connect with ETH VPN [cisco secure client](https://it.arch.ethz.ch/vpn-mac-cisco-client/) 
* start a new session
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227911480-abe5a7c4-bda6-425b-a12f-0ae8948e6f1c.PNG"  alt="" width = 100% height = auto></td>
</tr>
</table>


* create a new SSH session
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/228225950-cb2bc592-2f0a-4acf-9b57-847c8693f76e.PNG"  alt="" width = 100% height = auto></td>
</tr>
</table>


* enter passphrase and verification code
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/228226892-94cf0734-2f3a-4e92-8857-e76a4b93111a.PNG"  alt="" width = 100% height = auto></td>
</tr>
</table>

### UPLOAD WITH MOBAXTERM
* once the session is created successfully upload data from local server to leomed server.
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227919541-561e5edc-e402-4ccc-9eff-aedc8d453915.PNG"  alt="" width = 100% height = auto></td>
</tr>
</table>
 
### UPLOAD WITH WLS
* Launch WLS
* Enter the passphrase (set with IT)
* mount the server drive

```
sudo mount -t drvfs Z: /mnt/z
```

* upload the whole repository with one command (Change the username with yours. Second Authenticator OTP will be asked, Must be done separately for each recording mode)
> BLED

```
rsync -rav /mnt/z/neuramod_data/data/raw/eeg/O+MATERIALS+NRMD+NRMD_UPLD_RAWD_BLED+RAWD_BLED -e "ssh -J pierrecu@jump-neuramod.leomed.ethz.ch -i ~/.ssh/known_hosts" pierrecu@login-neuramod.leomed.ethz.ch:/cluster/work/neuramod/openbis_dropboxes/eln-lims-dropbox/
```

> SD

```
rsync -rav /mnt/z/neuramod_data/data/raw/eeg-sd/O+MATERIALS+NRMD+NRMD_UPLD_RAWD_SDCD+RAWD_SDCD -e "ssh -J pierrecu@jump-neuramod.leomed.ethz.ch -i ~/.ssh/known_hosts" pierrecu@login-neuramod.leomed.ethz.ch:/cluster/work/neuramod/openbis_dropboxes/eln-lims-dropbox/
```

> STIM

```
rsync -rav /mnt/z/neuramod_data/data/raw/stim/O+MATERIALS+NRMD+NRMD_UPLD_RAWD_STIM+RAWD_STIM -e "ssh -J pierrecu@jump-neuramod.leomed.ethz.ch -i ~/.ssh/known_hosts" pierrecu@login-neuramod.leomed.ethz.ch:/cluster/work/neuramod/openbis_dropboxes/eln-lims-dropbox/
```

* the data will start appearing into the eln-dropboxes of the Leohmed instance. It will then be automatically uploaded to Openbis from there and delete it from the dropboxes
* once uploaded you can also delete the zip files locally from your computer

<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/3306992/212131882-d7b6921f-c177-494b-ac19-58fa2ed1ef40.png"  alt="" width = 100% height = auto></td>
<td><img src="https://user-images.githubusercontent.com/3306992/212132035-908d9ef5-230c-4f4f-9349-d5cd2ca36bce.png"  alt="" width = 100% height = auto></td>
</tr>
</table>

### PROCESSING DATA FROM OPENBIS

* in OpenBIS, Copy your current session token from `OpenBIS > Utilities > User Profile`

<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/3306992/212133465-60b018af-18d9-4a1c-8942-e10ff59e596f.png"  alt="" width = 100% height = auto></td>
</tr>
</table>

* Click on `Jupyter Workspace`

<table>
<tr>
<td><img src="https://paper-attachments.dropboxusercontent.com/s_949A6604B8388C6553D823212D3D8ECBAF18F8C84E945B5F72A47E6784EA7124_1673540404081_Screenshot+2023-01-12+171652.png"  alt="" width = 100% height = auto></td>
</tr>
</table>

* Click on `01 DATASETS`

<table>
<tr>
<td><img src="https://paper-attachments.dropboxusercontent.com/s_949A6604B8388C6553D823212D3D8ECBAF18F8C84E945B5F72A47E6784EA7124_1673540391206_Screenshot+2023-01-12+171626.png"  alt="" width = 100% height = auto></td>
</tr>
</table>

* Click on `000_AssignMetadataToNewUpl`

<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/3306992/212133561-ea52d7a7-590f-4d01-8809-4a90e12c2976.png"  alt="" width = 100% height = auto></td>
</tr>
</table>

* Paste your session token there and modify the parent_proj, parent_sensor and parent_stim if necessary (make sure the participant ID your are refering to is created on Openbis. as well as project, sensors and stim object)

<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/3306992/212133706-339f855f-d507-4a8b-bdcd-7f3926ef6586.png"  alt="" width = 100% height = auto></td>
</tr>
</table>

* Run the notebook

* When done, check your datasets. If you select one, it should like below with relevant child-parent relationships and metadata (Make sure you set the right sensor, stim and proj objects)
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/3306992/212733682-c59440ae-8bb5-4c13-8cbb-97e37e55d66e.png"  alt="" width = 100% height = auto></td>
</tr>
</table>

* This is what a RAW BLED dataset should contain (for a Bitbrain Versatile 32 recording device):
  - `signalsInfo.json`: signals report data
  - `signalsInfo.csv`: signals report data
  - `Photodiode.csv`: photodiode data
  - `Layout.xlsx`: custom eeg layout data
  - `IMU_B.csv`: IMU data
  - `ExG_B.csv`: analog input data
  - `EEG.csv`: eeg data
  - `EEG-impedances.csv`: electrodes impedance data
  - `D in.csv`: digital input data
  
* This is what a RAW SDCD dataset should contain (for a Bitbrain Versatile 32 recording device):
  - `signalsInfo.csv`: signals report data
  - `Photodiode.csv`: photodiode data
  - `IMU_B.csv`: IMU data
  - `ExG_B.csv`: analog input data
  - `EEG.csv`: eeg data
  - `EEG-impedances.csv`: electrodes impedance data
  - `D in.csv`: digital input data
  - `0_BBTCapV9.bbt`: legacy data file
  
 * This is what a RAW LSLD dataset should contain (for a Bitbrain Versatile 32 recording device):
  - `EEG.csv`: eeg data
  - `Photodiode.csv`: photodiode data 
  
 * This is what a RAW STIM dataset should contain (for a Psychopy presentation paradigm):
  - `4466_P001_S003_T001_rsvp_paradigm_2022_Dec_07_0754.psydat`: Psychopy backup file
  - `4466_P001_S003_T001_rsvp_paradigm_2022_Dec_07_0754.log`: Psychopy log file
  - `4466_P001_S003_T001_rsvp_paradigm_2022_Dec_07_0754.csv`: stimulation data
  
* ~~Go to `Jupyter Workspace` > `01 DATASETS` > `001_FromRawDatasetToRawBidsFormat`~~ (Deprecated)
* Go to `Jupyter Workspace` > `01 DATASETS` > `002_FromRawDatasetToRawBidsFormat` (New version)
* 
* Paste your session token again if necessary and run the notebook.
* When done, BIDS datasets should now appear like this with all its parents-children relationships:
<table>
<tr>
<td>
<img width="1264" alt="Screenshot 2023-01-19 143039" src="https://user-images.githubusercontent.com/3306992/213455421-c9898423-f887-4541-9a8a-10ac89165587.png">
</td>
</tr>
</table>

* This is what a RAW BIDS dataset should contain:
  - `1285_P000_S000_T001_events.csv`: events file used for bids generation
  - `1285_P000_S000_T001.zip`: the bids dataset folder compressed
  - `1285_P000_S000_T001.vmrk`: vmrk file used for bids generation
  - `1285_P000_S000_T001.vhdr`: vhdr file used for bids generation
  - `1285_P000_S000_T001.eeg`: eeg file used for bids generation

### DOWNLOAD DATA WITH MOBAXTERM
* open the session that has been already created and then enter the passphrase and verfication code afterwards download the desired files
<table>
<tr>
<td><img src="https://user-images.githubusercontent.com/87472076/227923754-f25dc17d-2586-4954-b1ba-e47949ed3e34.PNG"  alt="" width = 100% height = auto></td>
</tr>
</table>



### DOWNLOAD DATA WITH WLS

## Preprocessing (Locally)
* BIDS file checker and viewer
In order to validate the BIDS format please click on the link and upload the BIDS folder. [BIDS dataset to validate](https://bids-standard.github.io/bids-validator/)
Script to check the validation of BIDS format.
```
pip install -U bids_validator (install bids_validator)
```
```from bids_validator import BIDSValidator
validator = BIDSValidator()
filepaths = ["C:/Users/neuramod/Desktop/3807_P001_S001_T001"]
for filepath in filepaths:
    print(validator.is_bids(filepath))
```

* preprocessing pipeline documentation
* the script (000_processing_pipeline.py) includes pre and post processing pipeline [](https://github.com/neuramod/neuramod_data/tree/main/code/processing/local)
* initially standard 10-20 monatge is applied
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


* channel rejection is applied having the parameter vaules of bad_threshold=0.5 and distance_threshold=0.96
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
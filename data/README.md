Home repo page: [LINK](https://github.com/neuramod/neuramod_data)

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
  - `...`: ...
  
 * This is what a RAW STIM dataset should contain (for a Psychopy presentation paradigm):
  - `4466_P001_S003_T001_rsvp_paradigm_2022_Dec_07_0754.psydat`: Psychopy backup file
  - `4466_P001_S003_T001_rsvp_paradigm_2022_Dec_07_0754.log`: Psychopy log file
  - `4466_P001_S003_T001_rsvp_paradigm_2022_Dec_07_0754.csv`: stimulation data
  
* Go to `Jupyter Workspace` > `01 DATASETS` > `001_FromRawDatasetToRawBidsFormat`
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

## Datasets visual inspection
* before uploading (right after a recording session)
* after uploading on OpenBis

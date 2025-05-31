# CITATION
- Cutellic, P., Qureshi, N. (2022). EEG Recordings For Neural Correlates Of Visual Discrimination Under RSVP Oddball Stimulation with Dynamic Gratings Compositions. https://doi.org/10.3929/ethz-b-000738641

# FILE STRUCTURE
## NAME
`<PARTICIPANT_ID>` _ `<PHASE_ID>` _ `<SESSION_ID>` _ `<TASK_ID>`
- `<PARTICIPANT_ID>`: defaced participant ID number
- `<PHASE_ID>`: project phase number
- `<SESSION_ID>`: participant-wise recording session
- `<TASK_ID>`: task number

## CONTENT
EEG-BIDS recordings according to:
- Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896). https://doi.org/10.21105/joss.01896
- Pernet, C. R., Appelhoff, S., Gorgolewski, K. J., Flandin, G., Phillips, C., Delorme, A., Oostenveld, R. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. Scientific Data, 6, 103. https://doi.org/10.1038/s41597-019-0104-8

Files:
All files in zip folders per participant's session
```
└- sub-<PARTICIPANT_ID> : data recording folder
  └- eeg : data recording subfolder by modality
    └- sub-<PARTICIPANT_ID>_space-CapTrak_coordsystem.json : electrodes coordinate system
    └- sub-<PARTICIPANT_ID>_space-CapTrak_electrodes.tsv : electrodes coordinates
    └- sub-<PARTICIPANT_ID>_task-<TASK_ID>_channels.tsv : channels (electrodes) description
    └- sub-<PARTICIPANT_ID>_task-<TASK_ID>_eeg.eeg : eeg data recording (*.eeg format)
    └─ sub-<PARTICIPANT_ID>_task-<TASK_ID>_eeg.json : eeg data recording description
    └─ sub-<PARTICIPANT_ID>_task-<TASK_ID>_eeg.vhdr : eeg data recording (*.vhdr format)
    └─ sub-<PARTICIPANT_ID>_task-<TASK_ID>_eeg.vrmk : eeg data recording (*.vrmk format)
    └─ sub-<PARTICIPANT_ID>_task-<TASK_ID>_events.tsv : stimulation events recording
  └─ sub-<PARTICIPANT_ID>_scans.tsv : recording description, unused
└─ dataset_description.json : dataset basic description file
└─ participants.json : participant description file (*.json format)
└─ participants.tsv : participant description file (*.tsv format)
└─ README.json : participant dataset description
└─ References.txt : Citation refernces for EEG-BIDS imaging standards
```

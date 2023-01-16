Home repo page: [LINK](https://github.com/neuramod/neuramod_data)

### Population Management
* registration and session recordings: `population\population.xlsx`
* openbis update or register participants: `population\openbis\`
1. Fill-in the population sheet
2. copy paste (with values only) the generated data from the OPENBIS sheet to the population_register file if there are new participants, or population_update if there is new data for existing participants.
3. Then run the command in WLS:

    sudo mkdir /mnt/z
    sudo mount -t drvfs Z: /mnt/z
    rsync -rav /mnt/z/neuramod_data/population/openbis_participants -e "ssh -J pierrecu@jump-neuramod.leomed.ethz.ch -i ~/.ssh/known_hosts" pierrecu@login-neuramod.leomed.ethz.ch:/cluster/work/neuramod/openbis_dropboxes/eln-lims-dropbox/
    
    
4. In OpenBis, go to Participants > More > and XLS Batch Update/Register Objects

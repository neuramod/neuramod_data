# neuramod_data
temporary datasets for internal use and project development

- current python version 3.9.6 (available installer from neuramod_softwares/coding)
- works with your local virtual environment "neuramod"
- check requirements.txt for updated packages. They are the same than for neuramod_experiments
## REPOSITORY STRUCTURE
### Code
A list of Python scripts and Jupyter notebooks to be used on the OpenBis server or locally.
* `README.md`: General info [LINK](https://github.com/neuramod/neuramod_data/blob/main/code/README.md)
#### - Population
* `000_PopulationStratifiedSegmentPlot.ipynb`: a simple script that plots the distribution of the population segment. [LINK]()
* `001_VisualizeWholePopulationDistribution.ipynb`: plots the population sample distribution. [LINK]()
#### - Processing
* `000_initial_processing.py`: a script that assign events, data slicing and conversion of raw data to brain vision file format
* `000_processing_pipeline.py`: Pre and post processing script i.e., montage, bad channel rejection, high/low pass filter, notch filter, deternd, channel interpolation, reference, epoching, evoked response, time-frequency analysis, classifier pipelines, Peak amplitude and latency w.r.t specific time window.
* `000_grand_average_plot.py.py`: plots grand average and difference wave between two stimuli.
#### - Streaming
* `000_lsl_stream_processing.py`: a lsl script that pull eeg data in chunks and pull data (photodiode and psychopy) in sample.
* `utlis.py`: pre and post processing script for the data received from "000_lsl_stream_processing.py" i.e., assign events, montage, bad channel rejection, high/low pass filter, notch filter, deternd, channel interpolation, reference, epoching, classifier pipelines.


### Data
Raw and Processed datasets. Large generated files such as generated stimulations are kept on servers

#### - Raw
#### - Processed

### Population
Population data recordings

*

## DATA SPECIFICATIONS
### BIDS
* https://bids.neuroimaging.io/
* https://www.nature.com/articles/s41597-019-0104-8

## WLS (Windows Linux Subsystem)
* Install WLS : https://learn.microsoft.com/en-us/windows/wsl/install
* Install creddentials with IT

## OPENBIS
### Official documentation
* User: https://openbis.ch/index.php/docs/user-documentation/
* Admin: https://openbis.ch/index.php/docs/admin-documentation/

* Installing new python packages for JupyterHub: https://unlimited.ethz.ch/pages/viewpage.action?spaceKey=LeoMed2&title=Installing+Python+packages
(check which python version is installed first before enabling the modules)

### Official site
Make sure to have your VPN enabled first !
* Leohmed VM : https://rdesk-neuramod.leomed.ethz.ch/guacamole/#/client/MQBjAHBvc3RncmVzcWw=
* OpenBIS User: https://openbis-neuramod.ethz.ch/openbis/webapp/eln-lims/
* OpenBIS Admin: https://openbis-neuramod.ethz.ch/openbis/webapp/openbis-ng-ui/

# import libraries
import mne
import matplotlib.pyplot as plt
mne.viz.set_browser_backend('matplotlib')

def evoked_plot(the_data):
    evoked = mne.read_evokeds(the_data)
    colors = "blue", "red"
    title = "Evoked topoplot"
    mne.viz.plot_evoked_topo(evoked, color=colors, title=title, background_color="w", ylim=dict(eeg=[-1, 3]))
    evoked_tar = evoked[0]
    evoked_dist = evoked[1]
    ylim = dict(eeg =(-3, 4))
    ts_args = dict(gfp= True, ylim=ylim)
    topomap_args = dict(sensors = False)
    evoked_tar.plot_joint(title ='target evoked plot',ts_args=ts_args,topomap_args= topomap_args)
    evoked_dist.plot_joint(title ='distractors evoked plot',ts_args=ts_args,topomap_args= topomap_args)
    plt.close()
                        #####################################################################
                                        # SCRIPT CALL OR MODULE IMPORT #
                        #####################################################################
if __name__ == '__main__':


    data_name = "04466_S021_T002"
    the_drive = "D:"
    the_repo = "data"
    the_subrepo = "data_testing"
    the_type = "04466_S021_T002"
    the_data = f"{data_name}-ave.fif"
    evoked_plot(the_data)
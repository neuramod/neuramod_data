# import libraries
import mne
import matplotlib.pyplot as plt
mne.viz.set_browser_backend('matplotlib')

def evoked_plot(fnmae):
    evoked = mne.read_evokeds(fname)
    colors = "blue", "red" # add another color in case of multi class evoked response #
    title = "Evoked topoplot"
    mne.viz.plot_evoked_topo(evoked, color=colors, title=title, background_color="w", ylim=dict(eeg=[-1, 3]))
    # add another evoked response for multi class scenario #
    evoked_tar = evoked[0]
    evoked_dist = evoked[1]
    ylim = dict(eeg =(-3, 4))
    ts_args = dict(gfp= True, ylim=ylim)
    topomap_args = dict(sensors = False)
    # add another plot joint in case of multi class evoked response #
    evoked_tar.plot_joint(title ='target evoked plot',ts_args=ts_args,topomap_args= topomap_args)
    evoked_dist.plot_joint(title ='distractors evoked plot',ts_args=ts_args,topomap_args= topomap_args)
    plt.close()
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import h5py

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, roc_auc_score
import os

img_rows, img_cols, nb_channels = 32, 32, 2
input_dir = 'data'
decays = ['SinglePhotonPt50_IMGCROPS_n249k_RHv1',
          'SingleElectronPt50_IMGCROPS_n249k_RHv1']
channelMap = {0: 'Energy', 1: 'Time'}
decayMap = {0: 'Photon', 1: 'Electron'}


def load_data(start, stop):
    """Load photon and electron data from data/ directory

    Args:
        start (int): start index in the photon and electron dataset
        stop (int): ending index in the phton and electron dataset

    Returns:
        X: Photon and electron collider images with two channels: Energy, Time
        y: Labels for the collider images: Photon = 0, Electron = 1 
    """
    dsets = [h5py.File('%s/%s.hdf5' % (input_dir, decay)) for decay in decays]
    X = np.concatenate([dset['/X'][start:stop] for dset in dsets])
    y = np.concatenate([dset['/y'][start:stop] for dset in dsets])
    assert len(X) == len(y)
    return X, y


def remove_empty_pixels(X):
    """Set empty energy deposit pixels to np.nan

    Args:
        X (numpy.array): array of collider images, shape(-1,32,32,2)

    Returns:
        X (numpy.array): array of collider images, shape(-1,32,32,2)
    """
    e_X = np.where(X[:, :, :, 0] == 0, np.nan, X[:, :, :, 0])
    t_X = np.where(X[:, :, :, 0] == 0, np.nan, X[:, :, :, 1])
    X = np.concatenate([e_X[:, :, :, None], t_X[:, :, :, None]], axis=-1)
    return X

def unique_times(X, remove_empty=True):
    """Create an array of unique hit times for each event

    Args:
        X (numpy.array): array of collider images, shape(-1,32,32,2)
        remove_empty (bool, optional): mask out pixels without any energy deposits. Defaults to True.

    Returns:
        numpy.array: array of sorted unique hit times for each event
    """
    if remove_empty: X = remove_empty_pixels(X)
    X_t = X[:,:,:,1].reshape(-1,32*32) # get the time channel and flatten 
    X_t_sorted = np.sort(X_t,axis=-1) # sort timing
    dup_runs = ak.run_lengths(X_t_sorted.flatten()) # count number of hits that occured at the same times 
    mask = ak.unflatten(ak.sum(dup_runs)*[False], dup_runs) # create mask to only keep one of each time
    mask = ak.flatten(ak.concatenate(
        [~mask[:, 0, None], mask[:, 1:]], axis=-1)).to_numpy().reshape(X_t_sorted.shape)
    unique_X_t_sorted = np.sort(np.where(mask, X_t_sorted, np.nan), axis=-1)
    return unique_X_t_sorted


def timeordered(X, cumulative=False):
    X_unraveled = X.reshape(-1, 32*32, 2)
    X_t_timeordered = unique_times(X)
    maxframes = np.max(np.sum(~np.isnan(X_t_timeordered), axis=-1))
    X_t_timeordered = X_t_timeordered[:, :maxframes]

    if cumulative:
        frame_masks = (X_unraveled[:, None, :, 1] <=
                       X_t_timeordered[:, :, None])
    else:
        frame_masks = (X_unraveled[:, None, :, 1] ==
                       X_t_timeordered[:, :, None])

    X_e_timeordered = np.where(
        ~frame_masks, np.nan, X_unraveled[:, None, :, 0]).reshape(-1, maxframes, 32, 32, 1)
    return X_e_timeordered, X_t_timeordered, maxframes


def timeordered_BC(X, cumulative=False, remove_empty=True, min_t=-0.05, max_t=0.05, t_step=0.0099):
    """
    X: Image dataset of 32x32 pixels
    cumulative: Keep earlier hits in later time slices
    min/max_t: min and max time to consider (elimnate noise and empty data)
    t_step: time step 
    """
    if remove_empty:
        X = remove_empty_pixels(X)
    X_e, X_t = X[:, :, :, 0], X[:, :, :, 1]  # Decompose energy and time
    n_images, width, height, channels = X.shape  # Find shape of images
    t_bins = np.arange(min_t, max_t, t_step)  # Bin separation for images
    print(t_bins)
    t_mats = [np.full(shape=(width, height), fill_value=t) for t in t_bins]
    max_frames = len(t_mats)
    X_e_timeordered = np.zeros(shape=(n_images, max_frames, width, height))
    X_t_timeordered = np.zeros(shape=(n_images, max_frames, width, height))
    for i in range(n_images):
        for t in range(max_frames-1):
            lower = X[i, :, :, 1] > t_mats[t]  # Lower bound
            upper = X[i, :, :, 1] <= t_mats[t+1]  # Upper bound

            if cumulative:
                is_between = upper
            else:
                # Between upper and lower
                is_between = np.logical_and(lower, upper)

            X_e_timeordered[i, t, :, :] = np.where(
                ~is_between, np.nan, X_e[i, :, :])
            X_t_timeordered[i, t, :, :] = np.where(
                ~is_between, np.nan, X_t[i, :, :])

    if remove_empty:
        X_e_timeordered = np.where(
            np.isnan(X_e_timeordered), 0, X_e_timeordered)

    return X_e_timeordered, X_t_timeordered, max_frames, t_bins


def animate(X, y, t_bins, images=range(-1, 1), interval=500):
    """
    Animate images, from -5 to 5 (es and photons)
    """
    for i in images:
        img = []
        fig = plt.figure()
        if y[i] == 1:
            fig.suptitle('Electron')
        else:
            fig.suptitle('Photon')
        for j in range(X.shape[1]-1):
            label = f'{t_bins[j]:.2f} < t < {t_bins[j+1]:.2f}'
            temp_img = X[i, j, :, :]
            img.append([plt.imshow(temp_img, animated=True),
                       plt.text(18, 28, label)])
        ani = animation.ArtistAnimation(
            fig, img, interval=interval, blit=True, repeat_delay=0)
        plt.show()
        ani.save(f'Energy_{i+abs(images[0])}.gif')
        plt.close()
    os.system('mv *.gif gifs/')
    plt.close()

def inline_animation(X,y,tbins,event=0,interval=500,**kwargs):
    decay = decayMap[y[event]]
    fig, ax = plt.subplots()
    frames = [
        [ax.imshow( np.where(frame==0,np.nan,frame) ),ax.text(1,1,f"{decay}: ({tlo:.2f},{thi:.2f})")] for frame,tlo,thi in zip(X[event],tbins[:-1],tbins[1:])
    ]
    ani = animation.ArtistAnimation(fig,frames,interval=interval,**kwargs)
    return ani

def plot_event(X, y, event=0, channel=-1):
    """Plot channels for given event

    Args:
        X (numpy.array): array of collider images, shape(-1,32,32,2)
        y (numpy.array): array of collider image labels: Photon = 0, Electron = 1
        event (int, optional): index of event to plot. Defaults to 0.
        channel (int, optional): channel to plot: Energy = 0, Time = 1, Both = -1. Defaults to -1.
    """
    if channel == -1:
        channels = [0, 1]
    else:
        channels = [channel]

    fig, axs = plt.subplots(nrows=1, ncols=len(channels), figsize=(12, 5))

    decay = decayMap[y[event]]
    if len(channels) == 1:
        im = axs.imshow(X[event, :, :, channel])
        axs.set_title(channelMap[channel])
        axs.grid(True)
        fig.colorbar(im, ax=axs)

    else:
        for i, channel in enumerate(channels):
            im = axs[i].imshow(X[event, :, :, channel])
            axs[i].set_title(channelMap[channel])
            axs[i].grid(True)
            fig.colorbar(im, ax=axs[i])
    fig.suptitle(decay)
    fig.tight_layout()


def plot_spacetime(X, y, event=0, azim=0, elev=0, interactive=False):
    """Plot 3D spacetime of specified event

    Args:
        X (numpy.array): array of collider images, shape(-1,32,32,2)
        y (numpy.array): array of collider image labels: Photon = 0, Electron = 1
        event (int, optional): index of event to plot. Defaults to 0.
    """
    index_map = np.indices((32, 32))

    decay = decayMap[y[event]]
    x = index_map[0][~np.isnan(X[event, :, :, 1])]
    y = index_map[1][~np.isnan(X[event, :, :, 1])]
    z = X[event, :, :, 1][~np.isnan(X[event, :, :, 1])]
    c = X[event, :, :, 0][~np.isnan(X[event, :, :, 1])]
    fig = plt.figure(figsize=(8,8))

    if interactive:
        ax = Axes3D(fig)
    else:
        ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter(x, y, z, s=1000*c, alpha=0.5, c=c)
    ax.set_xlim(10, 20)
    ax.set_ylim(10, 24)
    ax.set(xlabel='X', ylabel='Y', zlabel='Time')
    ax.view_init(azim=azim, elev=elev)
    ax.set_title(f'{decay}')
    return fig

def plot_roc(y_true, y_pred):
    """Plot ROC Curve

    Args:
        y_true (numpy.array): array of true labels
        y_pred (numpy.array): array of predicted labels
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    line = [0,1]
    plt.plot(line,line,'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {auc:.3f})')
    plt.show()

def plot_history(history,metric='loss'):
    loss = history.history[metric]
    val_loss = history.history[f'val_{metric}']
    
    plt.plot(loss,label='Training')
    plt.plot(val_loss,label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

def plot_abstime(X,y,bins=100,energy_weights=False,figax=None):
    if figax is None: figax = plt.subplots()
    fig,ax = figax 

    ph_times = X[:,:,:,1][y == 0].flatten()
    ph_energy = X[:,:,:,0][y == 0].flatten()
    ph_weight = ph_energy if energy_weights else None

    el_times = X[:,:,:,1][y == 1].flatten()
    el_energy = X[:,:,:,0][y == 1].flatten()
    el_weight = el_energy if energy_weights else None

    ph_hist,bins,_ = ax.hist(np.abs(ph_times),bins=bins,density=1,histtype='step',label='Photon',weights=ph_weight)
    el_hist,bins,_ = ax.hist(np.abs(el_times),bins=bins,density=1,histtype='step',label='Electron',weights=el_weight)
    ax.set(xlabel='Absolute Time',ylabel='Density')
    ax.legend()
    return fig,ax

def plot_abstime_cdf(X,y,bins=100,energy_weights=False,figax=None):
    if figax is None: figax = plt.subplots()
    fig,ax = figax 

    ph_times = X[:,:,:,1][y == 0].flatten()
    ph_energy = X[:,:,:,0][y == 0].flatten()
    ph_weight = ph_energy if energy_weights else None

    el_times = X[:,:,:,1][y == 1].flatten()
    el_energy = X[:,:,:,0][y == 1].flatten()
    el_weight = el_energy if energy_weights else None

    ph_cdf,bins,_ = ax.hist(np.abs(ph_times.flatten()),bins=bins,density=1,histtype='step',cumulative=True,label='Photon',weights=ph_weight)
    el_cdf,bins,_ = ax.hist(np.abs(el_times.flatten()),bins=bins,density=1,histtype='step',cumulative=True,label='Electron',weights=el_weight)
    ax.set(xlabel='Absolute Time',ylabel='Cumulative Percent',ylim=(0.5,1.1))
    ax.grid()
    ax.legend()
    return fig,ax
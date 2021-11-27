import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, roc_auc_score

from .helpers import *


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

def inline_animation(X,y,tbins,event=0,lo=0,interval=500,**kwargs):
    decay = decayMap[y[event]]
    fig, ax = plt.subplots()
    frames = [
        [ax.imshow( np.where(frame<=lo,np.nan,frame) ),ax.text(1,1,f"{decay}: ({tlo:.2f},{thi:.2f})")] for frame,tlo,thi in zip(X[event],tbins[:-1],tbins[1:])
    ]
    ani = animation.ArtistAnimation(fig,frames,interval=interval,**kwargs)
    return ani

def plot_image(X,mask=True,lo=0,figax=None, log=False):
    if figax is None: figax = plt.subplots()
    fig,ax = figax 
    if mask: X = np.where(X<=lo,np.nan,X)
    im = ax.imshow(X, norm=clrs.LogNorm() if log else clrs.Normalize())
    fig.colorbar(im)
    return fig,ax

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


def plot_spacetime(X, y, event=0, azim=0, elev=0, lo=0, interactive=False):
    """Plot 3D spacetime of specified event

    Args:
        X (numpy.array): array of collider images, shape(-1,32,32,2)
        y (numpy.array): array of collider image labels: Photon = 0, Electron = 1
        event (int, optional): index of event to plot. Defaults to 0.
    """
    index_map = np.indices((32, 32))

    X = remove_empty_pixels(X,lo)

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
    sc = ax.scatter(x, y, z, s=1000*c, alpha=0.5, c=c)
    ax.set_xlim(10, 20)
    ax.set_ylim(10, 20)
    ax.set_zlim(-0.015,0.01)
    ax.set(xlabel='X', ylabel='Y', zlabel='Time')
    ax.view_init(azim=azim, elev=elev)
    ax.set_title(f'{decay}')
    # fig.colorbar(sc)
    return fig,ax

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
    if type(history) != dict: history = history.history

    loss = history[metric]
    val_loss = history[f'val_{metric}']
    
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
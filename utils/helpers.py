import numpy as np
import matplotlib.pyplot as plt
import h5py

img_rows, img_cols, nb_channels = 32, 32, 2        
input_dir = 'data'
decays = ['SinglePhotonPt50_IMGCROPS_n249k_RHv1', 'SingleElectronPt50_IMGCROPS_n249k_RHv1']
channelMap = {0:'Energy',1:'Time'}
decayMap = {0:'Photon',1:'Electron'}

def load_data( start, stop):
    dsets = [h5py.File('%s/%s.hdf5'%(input_dir,decay)) for decay in decays]
    X = np.concatenate([dset['/X'][start:stop] for dset in dsets])
    y = np.concatenate([dset['/y'][start:stop] for dset in dsets])
    assert len(X) == len(y)
    return X, y

def remove_empty_pixels(X):
    e_X = np.where(X[:,:,:,0] == 0,np.nan,X[:,:,:,0])
    t_X = np.where(X[:,:,:,0] == 0,np.nan,X[:,:,:,1])
    X = np.concatenate([e_X[:,:,:,None],t_X[:,:,:,None]],axis=-1)
    return X

def plot_event(X,y,event=0,channel=-1):
    if channel == -1: channels = [0,1]
    else: channels = [channel]

    fig,axs = plt.subplots(nrows=1,ncols=len(channels),figsize=(12,5))

    decay = decayMap[y[event]]
    if len(channels) == 1:
        im = axs.imshow(X[event,:,:,channel])
        axs.set_title(channelMap[channel])
        axs.grid(True)
        fig.colorbar(im,ax=axs)

    else:
        for i,channel in enumerate(channels):
            im = axs[i].imshow(X[event,:,:,channel])
            axs[i].set_title(channelMap[channel])
            axs[i].grid(True)   
            fig.colorbar(im,ax=axs[i])
    fig.suptitle(decay)
    fig.tight_layout()

def plot_spacetime(X,y,event=0):
    index_map = np.indices((32,32))

    decay = decayMap[y[event]]
    x = index_map[0][~np.isnan(X[event,:,:,1])]
    y = index_map[1][~np.isnan(X[event,:,:,1])]
    z = X[event,:,:,1][~np.isnan(X[event,:,:,1])]
    c = X[event,:,:,0][~np.isnan(X[event,:,:,1])]
    fig = plt.figure(figsize = (16,16))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(x,y,z, s = 1000*c, alpha=0.5, c=c)
    ax.set_xlim(10,20)
    ax.set_ylim(10,24)
    ax.set(xlabel='X Coord',ylabel='Y Coord',zlabel='Time')
    ax.view_init(azim=0, elev=0)
    ax.set_title(f'{decay}: SpaceTime Scatter')
    fig.tight_layout()
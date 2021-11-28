import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import h5py
from keras.layers import AveragePooling3D,TimeDistributed, Reshape, Input, Dense, Dropout, Flatten, Conv3D, MaxPooling3D,Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D, concatenate
from sklearn.metrics import roc_curve, auc, confusion_matrix,roc_auc_score
from keras.models import Model

from matplotlib import animation
import os
from datetime import date

#Plot organization
def format_plots():
  if 'science' in plt.style.available:
    plt.style.use('science')
  else:
    os.system('pip install SciencePlots -q')
    plt.style.reload_library()
    plt.style.use('science')



def make_plot_dir():
    day = date.today().strftime("%d_%m_%Y")
    isDir = os.path.isdir("Plots/Plots_"+day)
    if isDir == False:
        os.system("mkdir -p Plots_" +day)
        os.system("mv -n Plots_" +day+"/ Plots/")

def save_plot(fname):
    day = date.today().strftime("%d_%m_%Y")
    plt.savefig(fname,bbox_inches = "tight")
    os.system("mv " + fname + "* Plots/Plots_" +day+"/")

def plot_stuff():
  format_plots()
  make_plot_dir()

def animate(X,y,t_bins,images=range(-1,1),interval=500):
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
            temp_img = X[i,j,:,:]
            img.append([plt.imshow(temp_img,animated=True),plt.text(18,28,label)])
        ani = animation.ArtistAnimation(fig, img, interval=interval, blit=True,repeat_delay=0)
        plt.show()
        ani.save(f'Energy_{i+abs(images[0])}.gif')
        plt.close()
    os.system('mv *.gif gifs/')
    plt.close()

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


def timeordered(X,cumulative=False):
    X_unraveled = X.reshape(-1,32*32,2)
    X_t_timeordered = np.sort(X_unraveled[:,:,1],axis=-1)
    dup_runs = ak.run_lengths(X_t_timeordered.flatten())
    mask = ak.unflatten(ak.sum(dup_runs)*[False],dup_runs)
    mask = ak.flatten(ak.concatenate([~mask[:,0,None],mask[:,1:]],axis=-1)).to_numpy().reshape(X_t_timeordered.shape)
    X_t_timeordered = np.sort(np.where(mask,X_t_timeordered,np.nan),axis=-1)
    maxframes = np.max(np.sum(~np.isnan(X_t_timeordered),axis=-1))
    X_t_timeordered = X_t_timeordered[:,:maxframes]

    if cumulative:
        frame_masks = (X_unraveled[:,None,:,1] <= X_t_timeordered[:,:,None])
    else:
        frame_masks = (X_unraveled[:,None,:,1] == X_t_timeordered[:,:,None])

    X_e_timeordered = np.where(~frame_masks,np.nan,X_unraveled[:,None,:,0]).reshape(-1,maxframes,32,32,1)
    return X_e_timeordered,X_t_timeordered,maxframes
def timeordered_BC(X,cumulative=False,cutoff=0.,remove_empty=True,normalize=False,min_t = -0.05,max_t = 0.05,t_step=0.0099):
    """
    X: Image dataset of 32x32 pixels
    cumulative: Keep earlier hits in later time slices
    normalize: Max pixel is 1
    min/max_t: min and max time to consider (elimnate noise and empty data)
    t_step: time step 
    """
    if remove_empty:
        X = remove_empty_pixels(X)
    X_e,X_t = X[:,:,:,0],X[:,:,:,1] #Decompose energy and time
    n_images,width,height,channels = X.shape #Find shape of images
    t_bins = np.arange(min_t, max_t, t_step) #Bin separation for images
    t_mats = [np.full(shape=(width,height),fill_value=t) for t in t_bins]
    max_frames = len(t_mats)
    X_e_timeordered = np.zeros(shape=(n_images,max_frames,width,height))
    X_t_timeordered = np.zeros(shape=(n_images,max_frames,width,height))
    X_e_max = np.zeros(shape=(n_images,width,height))
    X_t_max = np.zeros(shape=(n_images,width,height))
    for i in range(n_images):
        counts = [] #Keep track of number of nonzero pixels in each frame
        for t in range(max_frames-1):
            lower = X[i,:,:,1] > t_mats[t] #Lower bound
            upper = X[i,:,:,1] <= t_mats[t+1] #Upper bound
            is_between = np.logical_and(lower,upper) #Between upper and lower
            counts.append(np.count_nonzero(is_between))
            X_e_timeordered[i,t,:,:] = np.where(~is_between,np.nan,X_e[i,:,:])
            X_t_timeordered[i,t,:,:] = np.where(~is_between,np.nan,X_t[i,:,:])
            index = np.argmax(counts) #Index of max counts
            X_e_max[i,:,:] = X_e_timeordered[i,index,:,:]
            X_t_max[i,:,:] = X_t_timeordered[i,index,:,:]


    #did it work
    cutoff_mat = np.full(X_e_timeordered.shape,cutoff)
    X_e_timeordered = np.where(X_e_timeordered >= cutoff_mat,X_e_timeordered,0.)
    
        
    if remove_empty:
        X_e_timeordered = np.where(
            np.isnan(X_e_timeordered), 0, X_e_timeordered)
    if normalize:
        X_e_timeordered = X_e_timeordered/np.nanmax(X_e_timeordered)
    return X_e_timeordered,X_t_timeordered,max_frames,t_bins,X_e_max,X_t_max

def TimeDistributed_Conv(input,filters=32,kernel_size=2):
  conv2d = Conv2D(filters=filters, 
                  kernel_size=kernel_size, 
                  activation='relu', 
                  padding='same')
  return TimeDistributed(conv2d)(input)
    

def inception2D(input,filter_sizes=[60,50,40]):
  """
  input: previous layer's output
  filter_sizes: sizes of kernels (1x1,2x2,3x3)
  returns inception layer
  """
  #1x1 convolution
  layer_1 = Conv2D(filters=filter_sizes[0], 
                  activation='relu', 
                  kernel_size=1,
                  padding='same')(input)
  #2x2 convolution
  layer_2 = Conv2D(filters=filter_sizes[0], 
                  activation='relu', 
                  kernel_size=1,
                  padding='same')(input)
  layer_2 = Conv2D(filters=filter_sizes[1], 
                  activation='relu', 
                  kernel_size=2,
                  padding='same')(layer_2)
        
  #3x3 convolution
  layer_3 = Conv2D(filters=filter_sizes[0], 
                  activation='relu', 
                  kernel_size=1,
                  padding='same')(input)
  layer_3 = Conv2D(filters=filter_sizes[2], 
                  activation='relu', 
                  kernel_size=3,
                  padding='same')(layer_3)
  #2x2 max pooling
  layer_4 = MaxPooling2D(pool_size=2,strides=1,
                             padding='same')(input)
  layer_4 = Conv2D(filters=filter_sizes[0], 
                  activation='relu', 
                  kernel_size=1,
                  padding='same')(layer_4)

  mid_1 = concatenate([layer_1, layer_2, layer_3, layer_4], axis = 3)
  #mid_1 = concatenate([layer_2, layer_3, layer_4], axis = 3)
  return mid_1

def inception3D(input,filter_sizes=[32,16,8]):
  """
  input: previous layer's output
  filter_sizes: sizes of kernels (1x1,2x2,3x3)
  returns inception layer
  """
  #1x1 convolution
  layer_1 = Conv3D(filters=filter_sizes[0], 
                  activation='relu', 
                  kernel_size=1,
                  padding='same')(input)
  #2x2 convolution
  layer_2 = Conv3D(filters=filter_sizes[0], 
                  activation='relu', 
                  kernel_size=1,
                  padding='same')(input)
  layer_2 = Conv3D(filters=filter_sizes[1], 
                  activation='relu', 
                  kernel_size=2,
                  padding='same')(layer_2)
        
  #3x3 convolution
  layer_3 = Conv3D(filters=filter_sizes[0], 
                  activation='relu', 
                  kernel_size=1,
                  padding='same')(input)
  layer_3 = Conv3D(filters=filter_sizes[2], 
                  activation='relu', 
                  kernel_size=3,
                  padding='same')(layer_3)
  #2x2 max pooling
  layer_4 = MaxPooling3D(pool_size=2,strides=1,
                             padding='same')(input)
  layer_4 = Conv3D(filters=filter_sizes[0], 
                  activation='relu', 
                  kernel_size=1,
                  padding='same')(layer_4)

  #mid_1 = concatenate([layer_1, layer_2, layer_3, layer_4], axis = 3)
  mid_1 = concatenate([layer_2, layer_3, layer_4], axis = 4)
  return mid_1


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
    os.system('mv *.gif ParticleImages/gifs/')
    plt.close()

def inline_animation(X,y,tbins,event=0,interval=500,**kwargs):
    decay = decayMap[y[event]]
    fig, ax = plt.subplots()
    frames = [
        [ax.imshow( np.where(frame==0,np.nan,frame) ),ax.text(1,1,f"{decay}: ({tlo:.2f},{thi:.2f})")] for frame,tlo,thi in zip(X[event],tbins[:-1],tbins[1:])
    ]
    ani = animation.ArtistAnimation(fig,frames,interval=interval,**kwargs)
    return ani


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

def plot_energy(X,bw=0.005):
  #Feed raw X data
  fig = plt.figure()
  ax = fig.add_subplot(111)
  bw = bw
  bins = np.arange(min(X[:,:,:,0].flatten()), max(X[:,:,:,0].flatten()) + bw, bw)
  ax.hist(X[:,:,:,0].flatten(),bins=bins)
  ax.set_yscale('log')
  ax.set_ylabel(r'$E$ [GeV]')
  ax.set_title('Calorimeter Energy')

def time_channels(X,normalize=False):
    #X from raw data
    X_temp,_,_,_,_,_ = timeordered_BC(X,remove_empty=True,cumulative=True,normalize=normalize,min_t=-0.015,max_t=0.01,t_step=0.00079)
    xt = np.sum(X_temp,axis=3)
    xy = np.sum(X_temp,axis=1)
    yt = np.sum(X_temp,axis=2)
    return np.stack((xy,xt,yt),axis=3) #xy is channel 0, xt is channel 1, yt is channel 2
def plot_history(history,metric='loss',save=False,fname=''):
    loss = history.history[metric]
    val_loss = history.history[f'val_{metric}']
    
    plt.plot(loss,label='Training')
    plt.plot(val_loss,label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    if save:
        save_plot(fname)
        plt.close()
    else:
        plt.show()
def plot_roc(y_true, y_pred,save=False,fname=''):
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
    if save:
        save_plot(fname)
        plt.close()
    else:
        plt.show()

def average_slice(tc_data,y):
  tc_data_e = [] #Electron
  tc_data_p = [] #Photon
  for i in range(tc_data.shape[0]):
    if y[i] == 0: #Electron
      tc_data_e.append(tc_data[i,:,:,:])
    if y[i] == 1: #Photon
      tc_data_p.append(tc_data[i,:,:,:])
  tc_data_e = np.asarray(tc_data_e) #Electron
  tc_data_p = np.asarray(tc_data_p) #Photon
  average_data_e = np.average(tc_data_e,axis=0)
  average_data_p = np.average(tc_data_p,axis=0)
  return tc_data_e,tc_data_p,average_data_e,average_data_p

def plot_avgtcdecomposition(avg_tc,figtitle='None'):
  fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
  fig.set_figheight(4)
  fig.set_figwidth(12)
  fig.suptitle(figtitle)
  ax1.imshow(avg_tc[:,:,0])
  ax1.set(xlabel='x',ylabel='y')

  ax2.imshow(avg_tc[:,:,1])
  ax2.set(xlabel='t',ylabel='x')

  ax3.imshow(avg_tc[:,:,2])
  ax3.set(xlabel='t',ylabel='y')
def plot_tcdecomposition(tc,figtitle='None',event=0):
  fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
  fig.set_figheight(4)
  fig.set_figwidth(12)
  fig.suptitle(f'{figtitle} Event: {event}')
  ax1.imshow(tc[event,:,:,0])
  ax1.set(xlabel='x',ylabel='y')

  ax2.imshow(tc[event,:,:,1])
  ax2.set(xlabel='t',ylabel='x')

  ax3.imshow(tc[event,:,:,2])
  ax3.set(xlabel='t',ylabel='y')

def google_net(input):
  conv2d_1 = Conv2D(strides=1,
                  filters=200, 
                  activation='relu', 
                  kernel_size=3, 
                  padding='same', 
                  kernel_initializer='TruncatedNormal')(input)
  max_pool_1 = MaxPooling2D(pool_size=2,strides=2,padding='same')(conv2d_1)
  conv2d_2 = Conv2D(strides=1,
                  filters=400, 
                  activation='relu', 
                  kernel_size=1, 
                  padding='same', 
                  kernel_initializer='TruncatedNormal')(max_pool_1)
  max_pool_2 = MaxPooling2D(pool_size=2,strides=1,padding='same')(conv2d_2)
  inception_1 = inception2D(max_pool_2)
  inception_2 = inception2D(inception_1)
  max_pool_3 = MaxPooling2D(pool_size=2,strides=2,padding='same')(inception_2)
  filter_sizes=[80,60,40]
  inception_3 = inception2D(max_pool_3,filter_sizes=filter_sizes)
  inception_4 = inception2D(inception_3,filter_sizes=filter_sizes)
  inception_5 = inception2D(inception_4,filter_sizes=filter_sizes)
  inception_6 = inception2D(inception_5,filter_sizes=filter_sizes)
  inception_7 = inception2D(inception_6,filter_sizes=filter_sizes)
  max_pool_4 = MaxPooling2D(pool_size=2,strides=2,padding='same')(inception_7)
  filter_sizes=[80,60,40]
  inception_8 = inception2D(max_pool_4,filter_sizes=filter_sizes)
  inception_9 = inception2D(inception_8,filter_sizes=filter_sizes)
  avg_pool_5 = AveragePooling2D(pool_size=2,padding='same')(inception_9)
  flat_1 = Flatten()(avg_pool_5)
  dropout_1 = Dropout(0.4)(flat_1)
  dense_1 = Dense(1000,activation='relu')(dropout_1)
  output = Dense(1, activation='sigmoid', kernel_initializer='TruncatedNormal')(dense_1)
  return Model([input],output)

def Bear_net_3D(input,model_name):
  x = Conv3D(strides=1,
                  filters=100, 
                  activation='relu', 
                  kernel_size=3, 
                  padding='same', 
                  kernel_initializer='TruncatedNormal')(input)
  x = MaxPooling3D(pool_size=2,strides=2,padding='same')(x)
  x = Conv3D(strides=1,
                  filters=200, 
                  activation='relu', 
                  kernel_size=1, 
                  padding='same', 
                  kernel_initializer='TruncatedNormal')(x)
  x = MaxPooling3D(pool_size=2,strides=1,padding='same')(x)
  x = inception3D(x)
  x = inception3D(x)
  x = MaxPooling3D(pool_size=2,strides=2,padding='same')(x)
  filter_sizes=[60,40,20]
  x = inception3D(x,filter_sizes=filter_sizes)
  x = inception3D(x,filter_sizes=filter_sizes)
  x = AveragePooling3D(pool_size=2,padding='same')(x)
  x = inception3D(x,filter_sizes=filter_sizes)
  x = inception3D(x,filter_sizes=filter_sizes)
  x = AveragePooling3D(pool_size=2,padding='same')(x)
  x = Flatten()(x)
  x = Dropout(0.4)(x)
  x = Dense(100,activation='relu')(x)
  output = Dense(2, activation='softmax', kernel_initializer='TruncatedNormal')(x)
  return Model([input],output,name=model_name)

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

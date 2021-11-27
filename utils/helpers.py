import numpy as np
import awkward as ak
import h5py

import os

img_rows, img_cols, nb_channels = 32, 32, 2
input_dir = 'data'
decays = ['SinglePhotonPt50_IMGCROPS_n249k_RHv1',
          'SingleElectronPt50_IMGCROPS_n249k_RHv1']
channelMap = {0: 'Energy', 1: 'Time'}
decayMap = {0: 'Photon', 1: 'Electron'}


class MinMaxScaler:
    def __init__(self, energy_cutoff=0.0):
        self.cutoff = energy_cutoff

    def fit(self, X):
        self.minim = np.nanmin(X)
        self.maxim = np.nanmax(X)
        self.cutoff = (self.cutoff-self.minim)/(self.maxim-self.minim)
        return self

    def transform(self, X):
        X = (X-self.minim)/(self.maxim-self.minim)
        X = np.where(X > self.cutoff, X, 0)
        return X


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


def load_train_valid_test(train_size, valid_size, test_size, batch_size):
    """Load training, validation, and test data 

    Args:
        train_size (int): training size
        valid_size (int): validation size
        test_size (int): testing size
        batch_size (int): batch size
    """
    # Set range of training set
    train_start, train_stop = 0, train_size
    assert train_stop > train_start
    assert (len(decays)*train_size) % batch_size == 0
    X_train, y_train = load_data(train_start, train_stop)

    # Set range of validation set
    valid_start, valid_stop = 160000, 160000+valid_size
    assert valid_stop > valid_start
    assert valid_start >= train_stop
    X_valid, y_valid = load_data(valid_start, valid_stop)

    # Set range of test set
    test_start, test_stop = 204800, 204800+test_size
    assert test_stop > test_start
    assert test_start >= valid_stop
    X_test, y_test = load_data(test_start, test_stop)

    samples_requested = len(decays) * (train_size + valid_size + test_size)
    samples_available = len(y_train) + len(y_valid) + len(y_test)
    assert samples_requested == samples_available
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def remove_empty_pixels(X, low_e=0, abstime=100):
    """Set empty energy deposit pixels to np.nan

    Args:
        X (numpy.array): array of collider images, shape(-1,32,32,2)

    Returns:
        X (numpy.array): array of collider images, shape(-1,32,32,2)
    """
    mask = (X[:, :, :, 0] > low_e) & (np.abs(X[:, :, :, 1]) < abstime)
    e_X = np.where(~mask, np.nan, X[:, :, :, 0])
    t_X = np.where(~mask, np.nan, X[:, :, :, 1])
    X = np.concatenate([e_X[:, :, :, None], t_X[:, :, :, None]], axis=-1)
    return X

def crop_images(X,crop):
    """Crop 32x32 collider images

    Args:
        X (numpy.array): array of collider images, shape(-1,32,32,2)
        crop (int): square dimension to crop collider images too, crop = 22 -> 22x22

    Returns:
        numpy.array: array of cropped collider images
    """
    crop = (32 - crop)//2
    lo,hi = crop, 32 - crop
    return X[:,lo:hi,lo:hi]

def unique_times(X, remove_empty=True):
    """Create an array of unique hit times for each event

    Args:
        X (numpy.array): array of collider images, shape(-1,32,32,2)
        remove_empty (bool, optional): mask out pixels without any energy deposits. Defaults to True.

    Returns:
        numpy.array: array of sorted unique hit times for each event
    """
    if remove_empty:
        X = remove_empty_pixels(X)
    X_t = X[:, :, :, 1].reshape(-1, 32*32)  # get the time channel and flatten
    X_t_sorted = np.sort(X_t, axis=-1)  # sort timing
    # count number of hits that occured at the same times
    dup_runs = ak.run_lengths(X_t_sorted.flatten())
    # create mask to only keep one of each time
    mask = ak.unflatten(ak.sum(dup_runs)*[False], dup_runs)
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


def timeordered_BC(X, cumulative=False, remove_empty=True, low_e=0.005, min_t=-0.05, max_t=0.05, t_step=0.0099):
    """
    X: Image dataset of 32x32 pixels
    cumulative: Keep earlier hits in later time slices
    min/max_t: min and max time to consider (elimnate noise and empty data)
    t_step: time step 
    """
    if remove_empty:
        X = remove_empty_pixels(X, low_e)
    X_e, X_t = X[:, :, :, 0], X[:, :, :, 1]  # Decompose energy and time
    n_images, width, height, channels = X.shape  # Find shape of images
    t_bins = np.arange(min_t, max_t, t_step)  # Bin separation for images
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

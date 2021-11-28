#!/usr/bin/env python3
from sys import path
path.append(".")

import utils.helpers as pic
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import AUC
from tensorflow.keras import callbacks
from tensorflow import keras
from argparse import ArgumentParser
import talos


np.random.seed(1337)  # for reproducibility

parser = ArgumentParser()
parser.add_argument("--train-size", help="Training size",
                    type=int, default=10000)
parser.add_argument("--valid-size", help="Validation size",
                    type=int, default=10000)
parser.add_argument(
    "--t-range", help="Time range cutoff (-t_range,t_range)", type=float, default=0.1)
parser.add_argument("--t-step", help="Time step", type=float, default=0.0099)
parser.add_argument(
    "--iteration", help="Specify interaction running incase of parallel models", type=int, default=-1)
parser.add_argument(
    "--crop", help="Specify crop dimension for collider images", type=int, default=32)
parser.add_argument(
    "--epochs", help="Number of epochs to train", type=int, default=20)

args = parser.parse_args()

# Set range of training set
train_start, train_stop = 0, args.train_size
assert train_stop > train_start
X_train, y_train = pic.load_data(train_start, train_stop)
X_train = pic.crop_images(X_train,args.crop)

# Set range of validation set
valid_start, valid_stop = 160000, 160000+args.valid_size
assert valid_stop > valid_start
assert valid_start >= train_stop
X_valid, y_valid = pic.load_data(valid_start, valid_stop)
X_valid = pic.crop_images(X_valid,args.crop)

X_e_train, X_t_train, maxframes, time_bins = pic.timeordered_BC(
    X_train, cumulative=True, min_t=-args.t_range, max_t=args.t_range, t_step=args.t_step)
y_b_train = to_categorical(y_train)

X_e_valid, X_t_valid, _, _ = pic.timeordered_BC(
    X_valid, cumulative=True, min_t=-args.t_range, max_t=args.t_range, t_step=args.t_step)
y_b_valid = to_categorical(y_valid)

def cnn3d_model(X_train,y_train,X_valid,y_valid,params):
    model = keras.Sequential()

    model.add(layers.Reshape((maxframes, args.crop, args.crop, 1),
            input_shape=(maxframes, args.crop, args.crop)))

    model.add(layers.Conv3D(params['nfilters_1'],params['shape_1'],activation=params['activation']))
    model.add(layers.Dropout(params['dropout']))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D())

    model.add(layers.Conv3D(params['nfilters_2'],params['shape_2'],activation=params['activation']))
    model.add(layers.Dropout(params['dropout']))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D())

    model.add(layers.Flatten())
    model.add(layers.Dense(params['nodes_1'],activation=params['activation']))
    model.add(layers.Dense(params['nodes_2'],activation=params['activation']))
    model.add(layers.Dense(2, activation='softmax'))

    # model.summary()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)

    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=['accuracy',AUC()])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=args.epochs,
        batch_size=params['batchsize'],
        shuffle=True,
        callbacks=[reduce_lr],
        verbose=1
    )
    return history,model


params = {'activation': ['relu', 'elu'],
          'optimizer': ['Nadam', 'Adam'],
          'nfilters_1': [32, 64, 96],
          'shape_1': [1, 3, 5],
          'nfilters_2': [32, 64, 96],
          'shape_2': [1, 3, 5],
          'nodes_1': [32, 64, 96],
          'nodes_2': [32, 64, 96],
          'dropout': [0.2, 0.4, 0.6],
          'batchsize': [50, 100, 150],
          }

output = f'scans/cnn3d-v2-{args.t_range:.2e}-{args.t_step:0.2e}-{args.crop}x{args.crop}'

t = talos.Scan(x=X_e_train,
               y=y_b_train,
               x_val=X_e_valid,
               y_val=y_b_valid,
               model=cnn3d_model,
               fraction_limit=.001,
               experiment_name=output,
               params=params,
               clear_session=True
               )

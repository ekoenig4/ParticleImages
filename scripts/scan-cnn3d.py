#!/usr/bin/env python3
from sys import path
path.append(".")

import utils.helpers as pic
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
from tensorflow import keras
from argparse import ArgumentParser
import talos


np.random.seed(1337)  # for reproducibility

parser = ArgumentParser()
parser.add_argument("--train-size", help="Training size",
                    type=int, default=7500)
parser.add_argument("--valid-size", help="Validation size",
                    type=int, default=7500)
parser.add_argument(
    "--batch-size", help="Training batch size", type=int, default=50)
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

y_b_train = to_categorical(y_train)

y_b_valid = to_categorical(y_valid)


def cnn3d_model(X_train, y_train, X_valid, y_valid, params):
    X_train, _, maxframes, _ = pic.timeordered_BC(
        X_train, cumulative=params['cumulative'], min_t=-12*params['t_step'], max_t=12*params['t_step'], t_step=params['t_step'])
    X_valid, _, _, _ = pic.timeordered_BC(
        X_valid, cumulative=params['cumulative'], min_t=-12*params['t_step'], max_t=12*params['t_step'], t_step=params['t_step'])

    model = keras.Sequential()

    model.add(layers.Reshape((maxframes, args.crop, args.crop, 1),
                             input_shape=(maxframes, args.crop, args.crop)))

    model.add(layers.Conv3D(params['nfilters_1'], params['shape_1'],
              padding='same', activation=params['activation']))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D())

    model.add(layers.Conv3D(params['nfilters_2'], params['shape_2'],
              padding='same', activation=params['activation']))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool3D())

    model.add(layers.Flatten())
    model.add(layers.Dense(params['nodes_1'], activation=params['activation']))
    model.add(layers.Dense(params['nodes_2'], activation=params['activation']))
    model.add(layers.Dense(2, activation='softmax'))

    # model.summary()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[reduce_lr],
        verbose=1
    )
    return history,model


params = {
    'activation': ['elu'],
    'nfilters_1': [64],
    'shape_1': [5],
    'nfilters_2': [96],
    'shape_2': [3],
    'nodes_1': [64],
    'nodes_2': [32],
    'cumulative':[True,False],
    't_step': [0.00249, 0.0049, 0.0099, 0.049, 0.0249],
    'dropout':[0.4,0.6,0.8]
}

output = f'scans/cnn3d-v2-{args.crop}x{args.crop}'

talos.utils.gpu_utils.parallel_gpu_jobs(fraction=0.8)
scan = talos.Scan(x=X_train,
               y=y_b_train,
               x_val=X_valid,
               y_val=y_b_valid,
               model=cnn3d_model,
               experiment_name=output,
               fraction_limit=0.5,
               params=params,
               clear_session=True
               )

talos.Evaluate(scan).evaluate



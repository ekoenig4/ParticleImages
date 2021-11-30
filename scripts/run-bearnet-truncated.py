#!/usr/bin/env python3
from sys import path
path.append(".")

import utils.helpers as pic

import pickle
from argparse import ArgumentParser
from tensorflow.keras.metrics import AUC
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

import numpy as np
from keras.layers import GlobalAveragePooling2D, MaxPool2D,GlobalAveragePooling3D,AveragePooling3D,ConvLSTM2D,TimeDistributed, Reshape, Input, Dense, Dropout, Flatten, Conv3D, MaxPooling3D,Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D, concatenate
from keras.models import Sequential, Model,load_model
from keras.initializers import TruncatedNormal,Constant

np.random.seed(1337)  # for reproducibility


parser = ArgumentParser()
parser.add_argument("--lr-init", help="Learning rate",
                    type=float, default=1.e-3)
parser.add_argument(
    "--batch-size", help="Training batch size", type=int, default=500)
parser.add_argument("--train-size", help="Training size",
                    type=int, default=100000)
parser.add_argument("--valid-size", help="Validation size",
                    type=int, default=50000)
parser.add_argument(
    "--epochs", help="Number of epochs to train", type=int, default=100)
parser.add_argument(
    "--iteration", help="Specify interaction running incase of parallel models", type=int, default=-1)
parser.add_argument(
    "--crop", help="Specify crop dimension for collider images", type=int, default=32)
parser.add_argument("--no-save",help="Don't save model",action="store_true",default=False)

args = parser.parse_args()

# Set range of training set
train_start, train_stop = 0, args.train_size
assert train_stop > train_start
assert (len(pic.decays)*args.train_size) % args.batch_size == 0
X_train, y_train = pic.load_data(train_start, train_stop)
y_b_train = to_categorical(y_train)

# Set range of validation set
valid_start, valid_stop = 160000, 160000+args.valid_size
assert valid_stop > valid_start
assert valid_start >= train_stop
X_valid, y_valid = pic.load_data(valid_start, valid_stop)
y_b_valid = to_categorical(y_valid)


kernel_init='TruncatedNormal'
bias_init=Constant(value=5e-4)

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

def bearnet_truncated(input):
  x = Conv2D(32, (3, 3), padding='same', strides=1, activation='relu', name='conv_1_3x3')(input)
  x = MaxPool2D((2, 2), padding='same', strides=2, name='max_pool_1_3x3')(x)
  x = Conv2D(32, (2, 2), padding='same', strides=(1, 1), activation='relu', name='conv_2a_2x2')(x)
  x = Conv2D(96, (2, 2), padding='same', strides=(1, 1), activation='relu', name='conv_2b_2x2')(x)
  x = MaxPool2D((2, 2), padding='same', strides=1, name='max_pool_2_2x2/2')(x)

  x = inception_module(x,
                     filters_1x1=32,
                     filters_3x3_reduce=48,
                     filters_3x3=64,
                     filters_5x5_reduce=16,
                     filters_5x5=16,
                     filters_pool_proj=16,
                     name='inception_3a')
  x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=64,
                     filters_3x3=96,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=32,
                     name='inception_3b')

  x = MaxPool2D((2, 2), padding='same', strides=(2, 2), name='max_pool_3_2x2/2')(x)
  x = inception_module(x,
                     filters_1x1=96,
                     filters_3x3_reduce=48,
                     filters_3x3=104,
                     filters_5x5_reduce=16,
                     filters_5x5=24,
                     filters_pool_proj=32,
                     name='inception_4a')


  x1 = AveragePooling2D((2, 2), strides=2)(x)
  x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
  x1 = Flatten()(x1)
  x1 = Dense(1024, activation='relu')(x1)
  x1 = Dropout(0.4)(x1)
  x1 = Dense(2, activation='softmax', name='output')(x1)
  model = Model(input, [x1], name='inception_v1')
  return model

input = Input(shape=(32,32,1))
model = bearnet_truncated(input)

model.summary()

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)
earlystop = callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(
    learning_rate=args.lr_init), metrics=['accuracy',AUC()])

history=model.fit(X_train[:,:,:,0], y_b_train,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  validation_data=(X_valid[:,:,:,0], y_b_valid),
                  callbacks=[reduce_lr,earlystop],
                  verbose=1, shuffle=True)

if not args.no_save:
    output = f'models/bearnet-truncated'

    if args.iteration > -1:
        output = f'{output}-v{args.iteration}'

    model.save(output)
    with open(f'{output}/history.pkl', 'wb') as f_history:
        pickle.dump(history.history, f_history)

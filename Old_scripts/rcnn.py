import numpy as np
import awkward as ak
np.random.seed(1337)  # for reproducibility

from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.layers import AveragePooling3D,ConvLSTM2D,TimeDistributed, Reshape, Input, Dense, Dropout, Flatten, Conv3D, MaxPooling3D,Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D, concatenate
from keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam

from os import system
from os.path import exists

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import utils as pic

lr_init     = 1.e-3    # Initial learning rate  
batch_size  = 64       # Training batch size
train_size  = 2048     # Training size
valid_size  = 1024     # Validation size
test_size   = 1024     # Test size
epochs      = 10       # Number of epochs
doGPU       = False    # Use GPU
min_t       =0 #Min time
max_t       =0.01 #Max time
t_step      =0.00199 #time step

# Set range of training set
train_start, train_stop = 0, train_size
assert train_stop > train_start
assert (len(pic.decays)*train_size) % batch_size == 0
X_train, y_train = pic.load_data(train_start,train_stop)

# Set range of validation set
valid_start, valid_stop = 160000, 160000+valid_size
assert valid_stop  >  valid_start
assert valid_start >= train_stop
X_valid, y_valid = pic.load_data(valid_start,valid_stop)

# Set range of test set
test_start, test_stop = 204800, 204800+test_size
assert test_stop  >  test_start
assert test_start >= valid_stop
X_test, y_test = pic.load_data(test_start,test_stop)

samples_requested = len(pic.decays) * (train_size + valid_size + test_size)
samples_available = len(y_train) + len(y_valid) + len(y_test)
assert samples_requested == samples_available

X_e_train,X_t_train,maxframes,t_bins,X_e_max_train,X_t_max_train = pic.timeordered_BC(X_train,cutoff=0.005,remove_empty=True,normalize=True,min_t = min_t,max_t = max_t,t_step=t_step)
y_b_train = to_categorical(y_train)
X_e_valid,X_t_valid,_,t_bins,X_e_max_valid,X_t_max_valid = pic.timeordered_BC(X_valid,cutoff=0.005,remove_empty=True,normalize=True,min_t = min_t,max_t = max_t,t_step=t_step)
y_b_valid = to_categorical(y_valid)
X_e_test,X_t_test,_,t_bins,X_e_max_test,X_t_max_test = pic.timeordered_BC(X_test,cutoff=0.005,remove_empty=True,normalize=True,min_t = min_t,max_t = max_t,t_step=t_step)
y_b_test = to_categorical(y_test)


input_img = Input(shape=(maxframes,32, 32, 1))
conv3d_1 = Conv3D(filters=16,
                  kernel_size=(1,3,3),
                  activation='relu',
                  padding='same')(input_img)
conv3d_2 = Conv3D(filters=32,
                  kernel_size=(1,2,2),
                  activation='relu',
                  padding='same')(conv3d_1)
mp3d_1 = MaxPooling3D(strides=(1,2,2))(conv3d_2)
conv3d_3 = Conv3D(filters=32,
                  kernel_size=2,
                  activation='relu',
                  padding='same')(mp3d_1)
ap3d_1 = AveragePooling3D(strides=2)(conv3d_3)
conv3d_4 = Conv3D(filters=64,
                  kernel_size=1,
                  activation='relu',
                  padding='same')(ap3d_1)
ap3d_2 = AveragePooling3D(strides=2)(conv3d_4)
flatten_1 = Flatten()(ap3d_2)
dense_1 = Dense(100,activation='relu')(flatten_1)
#dropout_1 = Dropout(0.3)(dense_1)
#dense_2 = Dense(100,activation='relu')(dropout_1)
output = Dense(2,activation='softmax')(dense_1)

model = Model([input_img],output)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
plot_model(model,show_shapes=True,to_file='rcnn_4.png')
system('mv *.png Models/')

print('Running model\n*\n*\n*\n*\n*\n_________________________________________________________________\n')

history = model.fit(
    X_e_train, y_b_train,
    validation_data=(X_e_valid,y_b_valid),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    verbose=1
)

np.save('rcnn_history.npy',history.history)
system('mv *history.npy History/')



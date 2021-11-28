import numpy as np
import awkward as ak
np.random.seed(1337)  # for reproducibility

from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.layers import GlobalAveragePooling3D,AveragePooling3D,ConvLSTM2D,TimeDistributed, Reshape, Input, Dense, Dropout, Flatten, Conv3D, MaxPooling3D,Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D, concatenate
from keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

from os import system
from os.path import exists

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import utils as pic

lr_init     = 1.e-3    # Initial learning rate  
batch_size  = 100       # Training batch size
train_size  = 4000     # Training size
valid_size  = 1000     # Validation size
test_size   = 1000     # Test size
epochs      = 20       # Number of epochs
doGPU       = False    # Use GPU
min_t       =-0.1 #Min time
max_t       =0.1 #Max time
t_step      =0.0099 #time step

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

X_e_train,X_t_train,maxframes,t_bins,X_e_max_train,X_t_max_train = pic.timeordered_BC(X_train,min_t = min_t,max_t = max_t,t_step=t_step)
y_b_train = to_categorical(y_train)
X_e_valid,X_t_valid,_,t_bins,X_e_max_valid,X_t_max_valid = pic.timeordered_BC(X_valid,min_t = min_t,max_t = max_t,t_step=t_step)
y_b_valid = to_categorical(y_valid)
X_e_test,X_t_test,_,t_bins,X_e_max_test,X_t_max_test = pic.timeordered_BC(X_test,min_t = min_t,max_t = max_t,t_step=t_step)
y_b_test = to_categorical(y_test)

#Input image
input_3d = Input(shape=(maxframes,32, 32, 1))

#Evan 3D
model_name = 'enet_3d_1'

x = Conv3D(filters=64, kernel_size=3,padding='same', activation='relu')(input_3d)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = MaxPooling3D()(x)

x = Conv3D(filters=32, kernel_size=2,padding='same', activation='relu')(input_3d)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = MaxPooling3D()(x)

x = Flatten()(x)
x = Dense(units=64, activation='relu')(x)
x = Dense(units=32, activation='relu')(x)

output = Dense(2, activation='softmax', kernel_initializer='TruncatedNormal')(x)
model_4 = Model([input_3d],output,name=model_name)
#model_4.summary()

plot_model(model_4,show_shapes=True,to_file=f'{model_name}.png')
system('mv *.png Models/');
#Checkpoint and reduce lr
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)
checkpoint_cb = ModelCheckpoint("conv3d_1.h5", save_best_only=True)

#Compile
model_4.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_init),metrics=['accuracy'])

history_4 = model_4.fit(
    X_e_train, y_b_train,
    validation_data=(X_e_valid,y_b_valid),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    callbacks=[checkpoint_cb,reduce_lr],
    verbose=1
)

pic.plot_history(history_4,metric='loss',save=True,fname=f'history_{model_name}.png')
pic.plot_roc(y_b_test[:,0],model_4.predict(X_e_test)[:,0],save=True,fname=f'ROC_{model_name}.png')

#MRI scan conv3d 
model_name = 'cnn_3d_2'

x = Conv3D(filters=64, kernel_size=3,padding='same', activation='relu')(input_3d)
x = MaxPooling3D(pool_size=2)(x)
x = BatchNormalization()(x)

x = Conv3D(filters=64, kernel_size=3,padding='same',activation='relu')(x)
x = MaxPooling3D(2)(x)
#x = BatchNormalization()(x)

x = Conv3D(filters=128, kernel_size=2,padding='same', activation='relu')(x)
x = MaxPooling3D(2)(x)
#x = BatchNormalization()(x)

x = Conv3D(filters=256, kernel_size=1,padding='same', activation='relu')(x)
x = MaxPooling3D(2)(x)
#x = BatchNormalization()(x)

#x = GlobalAveragePooling3D()(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)


output = Dense(2, activation='softmax', kernel_initializer='TruncatedNormal')(x)
model_1 = Model([input_3d],output,name='cnn_3d_2')
#model_1.summary()

plot_model(model_1,show_shapes=True,to_file=f'{model_name}.png')
system('mv *.png Models/');
#Checkpoint and reduce lr
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)
checkpoint_cb = ModelCheckpoint("conv3d_1.h5", save_best_only=True)

#Compile
model_1.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_init),metrics=['accuracy'])

history_1 = model_1.fit(
    X_e_train, y_b_train,
    validation_data=(X_e_valid,y_b_valid),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    callbacks=[checkpoint_cb,reduce_lr],
    verbose=1
)

pic.plot_history(history_1,metric='loss',save=True,fname=f'history_{model_name}.png')
pic.plot_roc(y_b_test[:,0],model_1.predict(X_e_test)[:,0],save=True,fname=f'ROC_{model_name}.png')

#RCNN 3D
model_name = 'rcnn_3d_2'

x = ConvLSTM2D(filters=32, kernel_size=3,padding='same', activation='relu',return_sequences=True)(input_3d)
x = AveragePooling3D(pool_size=2)(x)
x = BatchNormalization()(x)

x = ConvLSTM2D(filters=64, kernel_size=2,padding='same',activation='relu',return_sequences=True)(x)
x = AveragePooling3D(pool_size=2)(x)
#x = BatchNormalization()(x)

x = Conv3D(filters=128, kernel_size=2,padding='same', activation='relu')(x)
x = MaxPooling3D(2)(x)
#x = BatchNormalization()(x)

x = Conv3D(filters=256, kernel_size=1,padding='same', activation='relu')(x)
x = MaxPooling3D(2)(x)
#x = BatchNormalization()(x)

#x = GlobalAveragePooling3D()(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)


output = Dense(2, activation='softmax', kernel_initializer='TruncatedNormal')(x)
model_2 = Model([input_3d],output,name=model_name)
#model_2.summary()

plot_model(model_2,show_shapes=True,to_file=f'{model_name}.png')
system('mv *.png Models/');
#Checkpoint and reduce lr
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)
checkpoint_cb = ModelCheckpoint("conv3d_1.h5", save_best_only=True)

#Compile
model_2.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_init),metrics=['accuracy'])
history_2 = model_2.fit(
    X_e_train, y_b_train,
    validation_data=(X_e_valid,y_b_valid),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    callbacks=[checkpoint_cb,reduce_lr],
    verbose=1
)

pic.plot_history(history_2,metric='loss',save=True,fname=f'history_{model_name}.png')
pic.plot_roc(y_b_test[:,0],model_2.predict(X_e_test)[:,0],save=True,fname=f'ROC_{model_name}.png')

model_name = 'bnet_3d_1'
model_3 = pic.Bear_net_3D(input_3d,model_name)
#model_3.summary()

plot_model(model_3,show_shapes=True,to_file=f'{model_name}.png')
system('mv *.png Models/');
#Checkpoint and reduce lr
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)
checkpoint_cb = ModelCheckpoint("conv3d_1.h5", save_best_only=True)

#Compile
model_3.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_init),metrics=['accuracy'])

history_3 = model_3.fit(
    X_e_train, y_b_train,
    validation_data=(X_e_valid,y_b_valid),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    callbacks=[checkpoint_cb,reduce_lr],
    verbose=1
)

pic.plot_history(history_3,metric='loss',save=True,fname=f'history_{model_name}.png')
pic.plot_roc(y_b_test[:,0],model_3.predict(X_e_test)[:,0],save=True,fname=f'ROC_{model_name}.png')


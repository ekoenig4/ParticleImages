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
batch_size  = 10       # Training batch size
train_size  = 400     # Training size
valid_size  = 100     # Validation size
test_size   = 100     # Test size
epochs      = 2       # Number of epochs
doGPU       = False    # Use GPU

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

y_b_train = to_categorical(y_train)
y_b_valid = to_categorical(y_valid)
y_b_test = to_categorical(y_test)

#Input image
input_2d = Input(shape=(32, 32, 1))

#MRI scan conv3d 
model_name = 'plot_test'

x = Conv2D(filters=64, kernel_size=3,padding='same', activation='relu')(input_2d)
x = MaxPooling2D(pool_size=16)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(units=12, activation='relu')(x)

output = Dense(2, activation='softmax', kernel_initializer='TruncatedNormal')(x)
model_1 = Model([input_2d],output,name=model_name)
model_1.summary()

#plot_model(model_1,show_shapes=True,to_file=f'{model_name}.png')
#system('mv *.png Models/');
#Checkpoint and reduce lr
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)
checkpoint_cb = ModelCheckpoint(f'{model_name}.h5', save_best_only=True)

#Compile
model_1.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_init),metrics=['accuracy'])

history_1 = model_1.fit(
    X_train, y_b_train,
    validation_data=(X_valid,y_b_valid),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    callbacks=[checkpoint_cb,reduce_lr],
    verbose=1
)

pic.plot_history(history_1,metric='loss',save=True,fname=f'history_{model_name}.png')
pic.plot_roc(y_test[:,0],model_1.predict(X_test)[:,0],save=True,fname=f'ROC_{model_name}.png')
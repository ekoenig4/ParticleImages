import numpy as np
import awkward as ak
np.random.seed(1337)  # for reproducibility

from tensorflow import keras,squeeze
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.layers import TimeDistributed, Reshape, Input, Dense, Dropout, Flatten, Conv3D, MaxPooling3D,Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D, concatenate
from keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from os import system
from os.path import exists
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

from sklearn.metrics import roc_curve, auc, confusion_matrix,roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import utils as pic

lr_init     = 1.e-3    # Initial learning rate  
batch_size  = 100       # Training batch size
train_size  = 4000     # Training size
valid_size  = 1000     # Validation size
test_size   = 1000     # Test size
epochs      = 20       # Number of epochs
doGPU       = False    # Use GPU
min_t        = -0.1    # Minimum time cutoff
max_t        = 0.1     # Maximum time cutoff
t_step       = 0.0199   # Time steps

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

#X_e_train,X_t_train,maxframes,t_bins,X_e_max_train,X_t_max_train = pic.timeordered_BC(X_train,cutoff=0.005,remove_empty=True,normalize=True,min_t = min_t,max_t = max_t,t_step=t_step)
#y_b_train = to_categorical(y_train)
#X_e_valid,X_t_valid,_,t_bins,X_e_max_valid,X_t_max_valid = pic.timeordered_BC(X_valid,cutoff=0.005,remove_empty=True,normalize=True,min_t = min_t,max_t = max_t,t_step=t_step)
#y_b_valid = to_categorical(y_valid)
#X_e_test,X_t_test,_,t_bins,X_e_max_test,X_t_max_test = pic.timeordered_BC(X_test,cutoff=0.005,remove_empty=True,normalize=True,min_t = min_t,max_t = max_t,t_step=t_step)
#y_b_test = to_categorical(y_test)



input_img = Input(shape=(32, 32, 1))
model_name = 'googlenet'

model = pic.google_net(input_img)
model.summary()
plot_model(model,show_shapes=True,to_file=f'{model_name}.png')
system('mv *.png Models/')

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=lr_init),metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)
checkpoint_cb = ModelCheckpoint(f'{model_name}.h5', save_best_only=True)

history = model.fit(
    X_train[:,:,:,0], y_train,
    validation_data=(X_valid[:,:,:,0],y_valid),
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    callbacks=[reduce_lr],
    verbose=1
)

pic.plot_history(history,metric='loss',save=True,fname=f'history_{model_name}.png')
pic.plot_roc(y_test[:,0],model.predict(X_test)[:,0],save=True,fname=f'ROC_{model_name}.png')

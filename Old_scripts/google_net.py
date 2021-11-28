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

from sklearn.metrics import roc_curve, auc, confusion_matrix,roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import utils as pic

lr_init     = 1.e-3    # Initial learning rate  
batch_size  = 100       # Training batch size
train_size  = 5000     # Training size
valid_size  = 1000     # Validation size
test_size   = 1000     # Test size
epochs      = 10       # Number of epochs
doGPU       = False    # Use GPU
tmin        = -0.1    # Minimum time cutoff
tmax        = 0.1     # Maximum time cutoff
tstep       = 0.0199   # Time steps

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

tc_train = pic.time_channels(X_train,normalize=False)
y_b_train = to_categorical(y_train)
tc_valid = pic.time_channels(X_valid,normalize=False)
y_b_valid = to_categorical(y_valid)
tc_test = pic.time_channels(X_test,normalize=False)
y_b_test = to_categorical(y_test)

input_img = Input(shape=(32, 32, 3))
model = pic.google_net(input_img)
model.summary()
plot_model(model,show_shapes=True,to_file='googlenet_3chan_1.png')
system('mv *.png Models/')

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=lr_init),metrics=['accuracy'])

history = model.fit(
    tc_train, y_train,
    validation_data=(tc_valid,y_valid),
    epochs=10,
    batch_size=batch_size,
    shuffle=True,
    verbose=1
)
pic.plot_history(history,metric='loss')
pic.plot_roc(y_b_test[:,0], model.predict(tc_test)[:,0])



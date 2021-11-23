#!/uscms/home/ekoenig/nobackup/anaconda3/bin/python

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from sys import path
from tensorflow import keras
from argparse import ArgumentParser
import pickle

path.append(".")
import utils as pic

np.random.seed(1337)  # for reproducibility


parser = ArgumentParser()
parser.add_argument("--lr-init", help="Learning rate",
                    type=float, default=1.e-3)
parser.add_argument(
    "--batch-size", help="Training batch size", type=int, default=100)
parser.add_argument("--train-size", help="Training size",
                    type=int, default=4000)
parser.add_argument("--valid-size", help="Validation size",
                    type=int, default=2000)
parser.add_argument(
    "--epochs", help="Number of epochs to train", type=int, default=20)
parser.add_argument(
    "--t-range", help="Time range cutoff (-t_range,t_range)", type=float, default=0.015)
parser.add_argument("--t-step", help="Time step", type=float, default=0.001)
parser.add_argument("--iteration",help="Specify interaction running incase of parallel models",type=int,default=-1)

args = parser.parse_args()

# Set range of training set
train_start, train_stop = 0, args.train_size
assert train_stop > train_start
assert (len(pic.decays)*args.train_size) % args.batch_size == 0
X_train, y_train = pic.load_data(train_start,train_stop)

# Set range of validation set
valid_start, valid_stop = 160000, 160000+args.valid_size
assert valid_stop  >  valid_start
assert valid_start >= train_stop
X_valid, y_valid = pic.load_data(valid_start,valid_stop)

X_e_train, X_t_train, maxframes, time_bins = pic.timeordered_BC(
    X_train, cumulative=False, min_t=-args.t_range, max_t=args.t_range, t_step=args.t_step)
y_b_train = to_categorical(y_train)

X_e_valid, X_t_valid, _, _ = pic.timeordered_BC(
    X_valid, cumulative=False, min_t=-args.t_range, max_t=args.t_range, t_step=args.t_step)
y_b_valid = to_categorical(y_valid)

model = keras.Sequential()

model.add(layers.Reshape((maxframes, 32, 32, 1),input_shape=(maxframes, 32, 32)))
model.add(layers.Conv3D(32, 3, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool3D())
model.add(layers.Conv3D(16,3,activation='relu',padding='same'))
model.add(layers.MaxPool3D())
model.add(layers.Flatten())
model.add(layers.Dense(25,activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(
    learning_rate=args.lr_init), metrics=['accuracy'])

history = model.fit(
    X_e_train, y_b_train,
    validation_data=(X_e_valid, y_b_valid),
    epochs=args.epochs,
    batch_size=args.batch_size,
    shuffle=True,
    verbose=1
)

output = f'models/cnn3d-{args.t_range:.2e}-{args.t_step:0.2e}'

if args.iteration > -1: output = f'{output}-v{args.iteration}'

model.save(output)
with open(f'{output}/history.pkl','wb') as f_history: pickle.dump(history.history,f_history)

#Gad, Abdi,Kieran

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
#Adam --> ization algorithm that can used instead of tochastic gradient descent
#to update network weights iterative based in training data.
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


train_data = pd.read_json('data/train.json')
train_data.inc_angle = train_data.inc_angle.replace('na',0)

train_data.tail()

X = train_data.drop(['is_iceberg'], axis=1)
y = train_data.is_iceberg

def scale_data(df):
    imgs = []
    # use for loop inside the list
    for index, row in df.iterrows():
        # data has two bands
        band_1 = np.array(row['band_1']).reshape(75,75)
        band_2 = np.array(row['band_2']).reshape(75,75)
        b_scale_1 = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b_scale_2 = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        imgs.append(np.dstack((b_scale_1, b_scale_2)))
    return np.array(imgs)

X = scale_data(X)
X
print(X.shape)
print(y.shape)


# using keras in-built function to get more data on training data
train_keras_more_data = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)


#making our model
model = Sequential()

# layer
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(75, 75, 2),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Dropout(0.2))

# layer
model.add(Conv2D(128,  kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

# layer
model.add(Conv2D(128,  kernel_size=(3, 3),activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

# layer
model.add(Conv2D(64,  kernel_size=(3, 3),activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

#Dense layer

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

# Output
model.add(Dense(1, activation="sigmoid"))

optimizer = Adam(lr=0.001, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

batch_size = 12
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
save_best_score = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
monitor_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')



model.fit(X, y, batch_size=batch_size,
          epochs=10, verbose=1, callbacks=[earlyStopping,
          save_best_score, monitor_loss], validation_split=0.25)

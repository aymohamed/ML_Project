#!/usr/bin/env python


#Gad, Abdi, Kieran

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import datetime
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Hide unwanted tf output
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam  # imports Adams, an algorithm for stochastic optimization
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model


def loadTrainData():
    CVsetSize = 0.25
    #loading training data, modifying inc angles and extracting training examples and
    #target values from the data

    train_data = pd.read_json('Data/train.json')
    # fix entries where inc angle is not a number:
    train_data.inc_angle = train_data.inc_angle.replace('na',0)
    X = train_data.drop(['is_iceberg'], axis=1)
    Y = train_data.is_iceberg

    CVsetSize = int(CVsetSize * len(Y))
    #return X_train, Y_train, X_CV, Y_CV
    return X[CVsetSize:], Y[CVsetSize:], X[:CVsetSize], Y[:CVsetSize]

def scaleData(df):
    imgs = []
    # use for loop inside list
    for index, row in df.iterrows():
        # the data has two bands
        band_1 = np.array(row['band_1']).reshape(75,75)
        band_2 = np.array(row['band_2']).reshape(75,75)
        b_scale_1 = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b_scale_2 = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        imgs.append(np.dstack((b_scale_1, b_scale_2)))
    return np.array(imgs)


def createModel():
    model = Sequential()
    dropoutRate = 0.5
    print("Dropout Rate: " + str(dropoutRate))

    # Input layer
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(75, 75, 2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Dropout(dropoutRate))

    # hidden layer 1
    model.add(Conv2D(64,  kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(dropoutRate))
    
    # hidden layer 2
    model.add(Conv2D(64,  kernel_size=(3, 3),activation='relu'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(dropoutRate))
    
    
    # hidden layer 3
    model.add(Conv2D(64,  kernel_size=(3, 3),activation='relu'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(dropoutRate))
    

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropoutRate))

    #Dense layer

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropoutRate))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer, metrics=['accuracy'])

    return model

def trainModel(model, X_train, Y_train, X_CV, Y_CV):
    batch_size = 32
    epochs = 20


    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    save_best_score = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    monitor_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    # Use Keras data generators on train set
    dataGen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
    dataGen.fit(X_train)

    model.fit_generator(dataGen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=len(X_train) / batch_size, epochs=epochs, verbose=2, callbacks=[earlyStopping, save_best_score, monitor_loss], validation_data=(X_CV, Y_CV))

    return model

def runOnTestData(model):
    X_test = pd.read_json('data/test.json')
    X_test.inc_angle = X_test.inc_angle.replace('na',0)
    X_scaled = scaleData(X_test)
    pred = model.predict(X_scaled)
    with open('predictions.csv', 'w') as outputfile:
        writer = csv.writer(outputfile, dialect='excel')
        writer.writerow(['id', 'is_iceberg'])

        for i in range(len(X_test)):
            writer.writerow([X_test.id[i], pred[i][0]])


def printModel(model):
    model.summary()

def main():
    now = datetime.datetime.now()
    print("*** TEST - " + now.strftime("%Y-%m-%d %H:%M") + " ***")
    X_train, Y_train, X_CV, Y_CV = loadTrainData()
    X_train = scaleData(X_train)
    X_CV = scaleData(X_CV)
    model = createModel()
    printModel(model)
    model = trainModel(model, X_train, Y_train, X_CV, Y_CV)
    runOnTestData(model)
    
    return


if __name__ == '__main__':
    main()

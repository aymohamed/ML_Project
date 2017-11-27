
#Gad, Abdi, Kieran

import numpy as np
import pandas as pd


#functionless loading of data
#loading train data
train_data = pd.read_json('Data/train.json')
train_images = train_data.apply(lambda rows: [np.stack([rows['band_1'], rows['band_2']], -1).reshape((75, 75, 2))],1)
train_images = np.stack(train_data).squeeze()
print("Training Data: ", train_data.shape)
print("Training Image Data: ", train_images.shape)

#loading test data
test_data= pd.read_json('Data/test.json')
test_images= test_data.apply(lambda rows: [np.stack([rows['band_1'], rows['band_2']], -1).reshape((75, 75, 2))],1)
test_images= np.stack(test_data).squeeze()
print("Testing Data: ", test_data.shape)
print("Testing Image Data: ", test_images.shape)

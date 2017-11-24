
#Gad, Abdi, Kieran

import json
from __future__ import division, print_function, absolute_import
import numpy as np
from skimage import color, io
from sklearn.cross_validation import train_test_split

#function for loading the data stored in the json files that contain the training
#and testing files

def openData():
    #loading training data
with open("train.json") as json_file:  # need to fill with the correct path for train data
    trainData  = json.load(json_file)
    trainingX=[]
    #Gad mentioned joining the 2 bands. Here is a rough suggestion.
    for j, row in trainingX.iterrows():
        traininingX.append([np.array(row['band_1']), np.array(row['band_2'])])
  trainingX = np.array(trainingX)

  #loading testing data

  with open("test.json") as json_file:  # need to fill with the correct path for test data 
      testData = json.load(json_file)
      testX=[]
      #Gad mentioned joining the 2 bands. Here is a rough suggestion.
      for k, row in testData.iterrows():
          testX.append([np.array(row['band_1']), np.array(row['band_2'])])
    testX = np.array(testX)

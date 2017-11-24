
#Gad, Abdi, Kieran

import json
from __future__ import division, print_function, absolute_import
import numpy as np
from skimage import color, io
from sklearn.cross_validation import train_test_split
from glob import glob
from tflearn.data_preprocessing import ImagePreprocessing


#import training data

with open("training.json") as json_file:
    trainingData  = json.load(json_file)
    print(trainingData)

    

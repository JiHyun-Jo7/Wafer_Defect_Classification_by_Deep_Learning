import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statistics import median
import time
import sys
import warnings


training_data=pd.read_pickle('./datasets/LSWMD_Train.pickle')
training_data.info()

# target = [] #############
# (X_train, Y_train), (X_test, Y_test) = train_test_split(training_data, target, test_size = 0.2)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)


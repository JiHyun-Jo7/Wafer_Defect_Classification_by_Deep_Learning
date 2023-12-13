import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPool2D, Dropout  # CNN의 핵심: Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils        # from tensorflow.python.keras.utils import np_utils 로 바꼈단다...
from tensorflow.keras import datasets

(X_train, Y_train), (X_test, Y_test) = datasets.fashion_mnist.load_data()
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
import sys, os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, AveragePooling2D, BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.datasets import mnist

# Do the testing data
test_file = pd.DataFrame(pd.read_csv(sys.argv[1]))
test_raw_data = np.array(test_file.values)
test_X = np.zeros((len(test_raw_data[:, 0]), 48, 48, 1))
for i in range(len(test_raw_data[:, 0])):
    test_X[i, :, :, 0] = np.array(test_raw_data[i, 1].split(' '), dtype='float').reshape(48, 48) / 255

model = load_model('my_model.h5?dl=1')

test_Y = model.predict(test_X)
output = np.zeros((len(test_raw_data[:, 0]), 1))
for i in range(len(test_raw_data[:, 0])):
    output[i, 0] = np.argmax(test_Y[i, :])
show = np.array(np.zeros((len(output), 2)), dtype='int')
id = np.array(range(len(output)))

show[:, 0] = id
show[:, 1] = output[:, 0]
output_file = pd.DataFrame(show, columns=['id', 'label'])
output_file.to_csv(sys.argv[2], index=False, header=True)
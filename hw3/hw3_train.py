import sys, os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization

ToWhich = 25000
train_file = pd.DataFrame(pd.read_csv(sys.argv[1]))
train_raw_data = np.array(train_file.values)
train_X = np.zeros((len(train_raw_data[:, 0]), 48, 48, 1))
train_Y_source = train_raw_data[:, 0]
train_Y = np.zeros((len(train_raw_data[:, 0]), 7))
for i in range(len(train_raw_data[:, 0])):
    train_X[i, :, :, 0] = np.array(train_raw_data[i, 1].split(' '), dtype='float').reshape(48, 48) / 255
    train_Y[i, train_Y_source[i]] += 1


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(train_X[:ToWhich, :, :, :], train_Y[:ToWhich, :], batch_size=256, shuffle=True, epochs=200)
score1 = model.evaluate(train_X[:ToWhich, :, :, :], train_Y[:ToWhich, :])
print('The train score:')
print(score1[1])
score2 = model.evaluate(train_X[ToWhich:, :, :, :], train_Y[ToWhich:, :])
print('The validation score:')
print(score2[1])

model.save('my_model.h5')
import pandas as pd
import numpy as np
import sys
import logging
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Text8Corpus
from gensim.models import word2vec
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.models import load_model


data = pd.read_table(sys.argv[1])
X_train_Adddata = np.array(data.values).reshape(1176366)
X_train_Adddata = X_train_Adddata[:1000000]


seglist = np.array(np.zeros((int(len(X_train_Adddata)/5), 40)), dtype='str')
for i in range(int(len(X_train_Adddata)/5)):
    input = X_train_Adddata[i].split(' ')
    seglist[i, :len(input)] = input[:40]
X_train_data = seglist

model = word2vec.Word2Vec.load("text300.model")

X_train_vector = np.zeros((int(len(X_train_Adddata)/5), 40, 300))
for i in range(len(X_train_data[:, 0])):
    for j in range(40):
        try:
            X_train_vector[i, j, :] = model[X_train_data[i, j]]
        except:
            if X_train_data[i, j] != '0.0':
                print('The Word "{}" not found and set it to zero!'.format(X_train_data[i, j]))


model_nn = load_model('my_model.h5')
result = model_nn.predict(X_train_vector)
print(result)
This_is_OK_pos = []
This_is_OK_neg = []
for i in range(int(len(X_train_Adddata)/5)):
    if result[i] > 0.8:
        This_is_OK_pos.append(i)
    elif result[i] < 0.2:
        This_is_OK_neg.append(i)


This_is_OK_pos_array = np.zeros((len(This_is_OK_pos), 40, 300))
This_is_OK_neg_array = np.zeros((len(This_is_OK_neg), 40, 300))
print('This_is_OK_pos_array')
for i in range(len(This_is_OK_pos)):
    This_is_OK_pos_array[i, :, :] = X_train_vector[This_is_OK_pos[i], :, :]
    print(i)

for i in range(len(This_is_OK_neg)):
    This_is_OK_neg_array[i, :, :] = X_train_vector[This_is_OK_neg[i], :, :]
    print(i)

X_add_train = np.vstack((This_is_OK_pos_array, This_is_OK_neg_array))
Y_add_train = np.hstack((np.ones(len(This_is_OK_pos)), np.zeros(len(This_is_OK_neg))))

# print(X_add_train.shape)
# print(Y_add_train.shape)
# np.save('X_add_train.npy', X_add_train)
# np.save('Y_add_train.npy', Y_add_train)
# X_add_train = np.load('X_add_train.npy')
# Y_add_train = np.load('Y_add_train.npy')

data = pd.read_csv('training_label.txt', sep='\+\+\+\$\+\+\+ ', header=None,)
train_data = np.array(data.values)
Y_train_data_O = np.array(train_data[:, 0], dtype='int')
X_train_data = train_data[:, 1]

seglist = np.array(np.zeros((len(train_data[:, 0]), 40)), dtype='str')
for i in range(len(train_data[:, 0])):
    input = X_train_data[i].split(' ')
    seglist[i, :len(input)] = input
X_train_data = seglist


X_train_vector = np.zeros((len(train_data[:, 0]), 40, 300))
for i in range(len(X_train_data[:, 0])):
    for j in range(40):
        try:
            X_train_vector[i, j, :] = model[X_train_data[i, j]]
        except:
            if X_train_data[i, j] != '0.0':
                print('The Word "{}" not found and set it to zero!'.format(X_train_data[i, j]))

X_train_vector_O = X_train_vector

X_train_vector_O = np.load('train_word_vector.npy')
Y_train_data_O = np.load('train_Y.npy')
print('OK Load')

X_validation = X_train_vector_O[180000:, :, :]
Y_validation = Y_train_data_O[180000:]

X_train_vector_O = X_train_vector_O[:180000, :, :]
Y_train_data_O = Y_train_data_O[:180000]


X_train_vector = np.vstack((X_train_vector_O, X_add_train))
Y_train_data = np.hstack((Y_train_data_O, Y_add_train))

print(X_train_vector.shape)



BATCH_SIZE = 640
NUM_EPOCHS = 20
model = Sequential()
model.add(LSTM(input_shape=(40, 300), units=512, activation='tanh', dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train_vector, Y_train_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_validation, Y_validation))
model.save('my_model_withAdd2.h5')

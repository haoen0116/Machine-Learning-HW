import os
import sys
import csv
import h5py
import pickle
import numpy as np
import keras.backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Embedding, Reshape, Dot, Add
from keras.callbacks import ModelCheckpoint, EarlyStopping


def normalizeRating(data):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	np.save('mean.npy', mean)
	np.save('std.npy', std)
	data = (data - mean) / std
	return data


def loadModel():
    model = model_from_json(open('model.json').read())
    model.load_weights('best_weights.hdf5')
    return model

def modelPredict(model, X_test, result_path, mean, std):
	pred = model.predict([X_test[:,0], X_test[:,1]], verbose=0)
	pred = (pred * std) + mean
	with open(result_path, 'w', encoding='big5') as file:
		file_data = csv.writer(file, delimiter=',', quotechar='\r')
		file_data.writerow(['TestDataId', 'Rating'])
		for i in range(np.shape(pred)[0]):
			file_data.writerow([str(i+1),float(pred[i])])


def readTestData(data_path):
	r_data = np.zeros((100336, 2), dtype=np.int)
	with open(data_path, 'r', encoding='utf-8') as f:
		data = csv.reader(f, delimiter=',', quotechar='\r')
		next(data, None)
		for i, row in enumerate(data):
			r_data[i] = [int(row[1]), int(row[2])]
	return r_data


def getMeanStd():
	mean = np.load('mean.npy')
	std = np.load('std.npy')
	return mean, std


def buildNNmodel():
    user_input = Input(shape=(1,))
    movies_input = Input(shape=(1,))
    # output tensor shape: (batch_size,1,embedding_dim) -> (batch_size,embedding_dim)
    users_embedding = Reshape((embedding_dim,))(Embedding(6040+1, embedding_dim, embeddings_initializer='random_normal',
                                                          input_length=1, trainable=True)(user_input))
    movies_embedding = Reshape((embedding_dim,))(Embedding(3952+1, embedding_dim, embeddings_initializer='random_normal',
                                                           input_length=1, trainable=True)(movies_input))
    dot = Dot(axes=1, normalize=True)([users_embedding, movies_embedding])
    users_bias = Reshape((1,))(Embedding(6040+1, 1, embeddings_initializer='zeros',
                                input_length=1, trainable=True)(user_input))
    movies_bias = Reshape((1,))(Embedding(3952+1, 1, embeddings_initializer='zeros',
                                input_length=1, trainable=True)(movies_input))
    out = Add()([dot, users_bias, movies_bias])
    model = Model(inputs=[user_input, movies_input], outputs=out)
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    return model


def trainModel(model, X_train, Y_train):
	earlystopping = EarlyStopping(monitor='loss', patience = 5, verbose=1, mode='auto')
	checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', verbose=1, save_best_only=True,
                                 save_weights_only=False, monitor='loss', mode='auto')
	model.fit([X_train[:,0], X_train[:,1]], Y_train, epochs=nb_epoch, batch_size=batch_size,
				callbacks=[earlystopping, checkpoint], verbose=1)
	return model


def saveModel(model):
	json_string = model.to_json()
	open('model.json', 'w').write(json_string)
	model.save_weights('weights.h5', overwrite=True)


# Start
np.random.seed(20171215)

# get path
train_path = 'train.csv'
test_path = sys.argv[1]
users_path = sys.argv[4]
movies_path = sys.argv[3]
result_path = sys.argv[2]

# train_path = 'train.csv'
# test_path = 'test.csv'
# users_path = 'users.csv'
# movies_path = 'movies.csv'
# result_path = 'prediction.csv'


Train = False

if Train:
	embedding_dim = 16
	nb_epoch = 400
	batch_size = 1024

	X_train = np.zeros((899873, 2), dtype=np.int)
	Y_train = np.zeros((899873), dtype=np.int)
	with open(train_path, 'r', encoding='utf-8') as f:
		data = csv.reader(f, delimiter=',', quotechar='\r')
		next(data, None)
		for i, row in enumerate(data):
			X_train[i] = [int(row[1]), int(row[2])]
			Y_train[i] = int(row[3])

	Y_train = normalizeRating(Y_train)
	model = buildNNmodel()
	model = trainModel(model, X_train, Y_train)
	saveModel(model)

X_test = readTestData(test_path)
mean, std = getMeanStd()
modelPredict(loadModel(), X_test, result_path, mean, std)
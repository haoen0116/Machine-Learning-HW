import sys
import pickle
import random
import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers.recurrent import LSTM,GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model,model_from_json
from keras.layers import Input, LSTM, Dense,Embedding,Dot
from gensim.models.word2vec import Word2Vec


def loadModel():
    model = model_from_json(open('model.json').read())
    model.load_weights('best_weights.hdf5')
    print('Loading best model...')
    return model


def output_voice_vector(file_path):
    with open(file_path, 'rb') as file:      # input the voice file
        train_data = pickle.load(file,encoding='latin1')
    voice_vector = np.zeros((2000, 246, 39))  # test.csv has 2000samples and 39 dims
    for i in range(len(train_data)):
        voice_vector[i, :len(train_data[i][:, 0]), :] += train_data[i]
    return voice_vector



def sen_index(file_path, dictionary):
    caption = pd.read_csv(file_path, header=None, )  # doing the "train.caption" file
    test_option = caption.values
    print(test_option.shape)
    test_caption = np.array(np.zeros((2000, 4)), dtype='str')  # There has 2000 samples, 4 option, and each option has max 13 words
    for i in range(2000):
        temp = list(test_option[i, ])
        test_caption[i, ] = temp

    all_sentence_a = []
    all_sentence_b = []
    all_sentence_c = []
    all_sentence_d = []
    for i in range(2000):
        all_sentence_a.append(test_caption[i, 0])
        all_sentence_b.append(test_caption[i, 1])
        all_sentence_c.append(test_caption[i, 2])
        all_sentence_d.append(test_caption[i, 3])
    a_sentence = []
    b_sentence = []
    c_sentence = []
    d_sentence = []
    for idx, caption in enumerate(all_sentence_a):
        word_counter = 0
        one_sentence = []
        for word in ("<S> " + caption + " </S>").split():
            if word in dictionary:
                one_sentence.append(word)
            else:
                one_sentence.append("<UNK>")
            word_counter += 1
        if word_counter < 17:
            for pad_word in range(17-word_counter):
                one_sentence.append("<PAD>")
        a_sentence += [one_sentence]
    for idx, caption in enumerate(all_sentence_b):
        word_counter = 0
        one_sentence = []
        for word in ("<S> " + caption + " </S>").split():
            if word in dictionary:
                one_sentence.append(word)
            else:
                one_sentence.append("<UNK>")
            word_counter += 1
        if word_counter < 17:
            for pad_word in range(17 - word_counter):
                one_sentence.append("<PAD>")
        b_sentence += [one_sentence]
    for idx, caption in enumerate(all_sentence_c):
        word_counter = 0
        one_sentence = []
        for word in ("<S> " + caption + " </S>").split():
            if word in dictionary:
                one_sentence.append(word)
            else:
                one_sentence.append("<UNK>")
            word_counter += 1
        if word_counter < 17:
            for pad_word in range(17 - word_counter):
                one_sentence.append("<PAD>")
        c_sentence += [one_sentence]
    for idx, caption in enumerate(all_sentence_d):
        word_counter = 0
        one_sentence = []
        for word in ("<S> " + caption + " </S>").split():
            if word in dictionary:
                one_sentence.append(word)
            else:
                one_sentence.append("<UNK>")
            word_counter += 1
        if word_counter < 17:
            for pad_word in range(17 - word_counter):
                one_sentence.append("<PAD>")
        d_sentence += [one_sentence]
    a_sentence = np.array(a_sentence).reshape(-1, 17)
    b_sentence = np.array(b_sentence).reshape(-1, 17)
    c_sentence = np.array(c_sentence).reshape(-1, 17)
    d_sentence = np.array(d_sentence).reshape(-1, 17)
    a_sentence_with_index = np.zeros((2000, 17))
    b_sentence_with_index = np.zeros((2000, 17))
    c_sentence_with_index = np.zeros((2000, 17))
    d_sentence_with_index = np.zeros((2000, 17))
    for i in range(2000):
        for j in range(17):
            a_sentence_with_index[i, j] = dictionary.index(a_sentence[i][j])
            b_sentence_with_index[i, j] = dictionary.index(b_sentence[i][j])
            c_sentence_with_index[i, j] = dictionary.index(c_sentence[i][j])
            d_sentence_with_index[i, j] = dictionary.index(d_sentence[i][j])

    return a_sentence_with_index, b_sentence_with_index, c_sentence_with_index, d_sentence_with_index, a_sentence, b_sentence, c_sentence, d_sentence


dictionary = list(np.load('dictionaries.npy'))
test_vector = output_voice_vector(sys.argv[1]) #'test.data'
a_sentence_with_index, b_sentence_with_index, c_sentence_with_index, d_sentence_with_index, a_sentence, b_sentence, c_sentence, d_sentence = sen_index(sys.argv[2], dictionary) #'test.csv'

model = loadModel()
prediction1 = model.predict([test_vector, a_sentence_with_index])
prediction2 = model.predict([test_vector, b_sentence_with_index])
prediction3 = model.predict([test_vector, c_sentence_with_index])
prediction4 = model.predict([test_vector, d_sentence_with_index])


model200 = word2vec.Word2Vec.load("text200.model")
ans = np.hstack((prediction1, prediction2, prediction3, prediction4))

out = np.zeros((2000, 2), dtype='int')
for i in range(2000):
    out[i, 0] = i + 1
    out[i, 1] = np.argmax(ans[i, :])
output = pd.DataFrame(out, columns=['id', 'answer'])
output.to_csv(sys.argv[3], index=False, header=True)
import pandas as pd
import numpy as np
import sys
import logging
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Text8Corpus
from gensim.models import word2vec
from gensim.models import word2vec
from keras.models import load_model

test_data = pd.read_table(sys.argv[1], sep=r'([0-9]+),')
testing_raw_data = np.array(test_data.values, dtype='str').reshape(200000)
seglist = np.array(np.zeros((len(testing_raw_data[:]), 40)), dtype='str')
for i in range(len(testing_raw_data[:])):
    input = testing_raw_data[i].split(' ')
    seglist[i, :len(input)] = input
X_test_data = seglist

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# sentences = LineSentence('training_nolabel.txt')
# model1 = word2vec.Word2Vec(sentences, size=300, sg=1, min_count=1)
model1 = word2vec.Word2Vec.load("text300.model")

X_test_vector = np.zeros((len(X_test_data), 40, 300))
for i in range(len(X_test_data[:, 0])):
    for j in range(40):
        try:
            X_test_vector[i, j, :] = model1[X_test_data[i, j]]
        except:
            if X_test_data[i, j] != '0.0':
                print('The Word "{}" not found and set it to zero!'.format(X_test_data[i, j]))
print(X_test_vector)

model2 = load_model('my_model_withAdd2.h5')
result = model2.predict(X_test_vector)
print(result)
for i in range(200000):
    if result[i] > 0.6:
        result[i] = 1
    else:
        result[i] = 0
print(result)
l = len(result)
out = np.zeros((l, 2), dtype='int')
for i in range(l):
    out[i, 0] = i
    out[i, 1] = result[i]
output = pd.DataFrame(out, columns=['id', 'label'])
output.to_csv(sys.argv[2], index=False, header=True)

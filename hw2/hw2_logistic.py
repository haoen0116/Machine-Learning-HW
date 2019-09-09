import scipy, sys, os
import numpy as np
import pandas as pd
import sys

def sigmoid(zz):
    ans = 1 / (1 + np.exp(-zz))
    return np.clip(ans, 0.00000000000001, 0.99999999999999)


b = np.load('LR_b.npy')
w = np.load('LR_w.npy')

X_test = pd.DataFrame(pd.read_csv(sys.argv[5]))#"X_test"
X_testing_data = np.array(X_test.values)

dim = len(X_testing_data[0, :])
sample_data_size = len(X_testing_data[:, 0])

Y_prediction = np.zeros(sample_data_size)
Y_testing_ans = np.zeros((sample_data_size, 2))
z = np.dot(X_testing_data, w) + b
Y_prediction = sigmoid(z)

for i in range(len(X_testing_data[:, 0])):
    Y_testing_ans[i, 0] = i + 1
    # print(predict(X_testing_data[i, :], mean1, mean0, sigma, counter1, counter0))
    if Y_prediction[i] > 0.5:
        Y_testing_ans[i, 1] = 1
    else:
        Y_testing_ans[i, 1] = 0

show = np.array(np.zeros([len(X_testing_data[:, 0]), 2]), dtype='int')
show[:, 0] = Y_testing_ans[:, 0]
show[:, 1] = Y_testing_ans[:, 1]
output_file = pd.DataFrame(show, columns=['id', 'label'])
output_file.to_csv(sys.argv[6], index=False, header=True)
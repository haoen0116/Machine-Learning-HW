import pandas as pd
import numpy as np
import sys

def predict(X, u1, u0, sigma_share, N1, N0):
    sigma_inverse = np.linalg.inv(sigma_share)
    w = np.dot(np.transpose(u1-u0), sigma_inverse)
    w = w.T
    b = (-0.5) * np.dot(np.dot(np.transpose(u1), sigma_inverse), u1) \
        + (0.5) * np.dot(np.dot(np.transpose(u0), sigma_inverse), u0) + np.log(float(N1)/N0)
    Z = np.dot(X, w) + b
    ans = sigmoid(Z)
    return ans


def sigmoid(Z):
    ans = 1 / (1 + np.exp(-Z))
    return np.clip(ans, 0.00000000000001, 0.99999999999999)


X_data = pd.DataFrame(pd.read_csv(sys.argv[3]))#"X_train"
Y_data = pd.DataFrame(pd.read_csv(sys.argv[4]))#"Y_train"
X = np.array(X_data.values)
Y = np.array(Y_data.values)

X_test = pd.DataFrame(pd.read_csv(sys.argv[5]))#"X_test"
X_testing_data = np.array(X_test.values)

dim = len(X[0, :])
sample_data_size = len(X[:, 0])

mean1 = np.zeros((dim, 1))
mean0 = np.zeros((dim, 1))
counter1 = 0
counter0 = 0
w = np.zeros(dim)
b = 0
# Find mean!
for WhichSample in range(sample_data_size):
    if Y[WhichSample, 0]:
        mean1 += np.transpose([X[WhichSample, :]])
        counter1 += 1
    else:
        mean0 += np.transpose([X[WhichSample, :]])
        counter0 += 1
mean1 /= counter1
mean0 /= counter0
# print(mean0)
# Find sigma!
sigma1 = np.zeros((dim, dim))
sigma0 = np.zeros((dim, dim))
for WhichSample in range(sample_data_size):
    if Y[WhichSample, 0]:
        temp = X[WhichSample, :]-np.transpose(mean1)
        sigma1 += np.dot(np.transpose(temp), temp)
    else:
        temp = X[WhichSample, :] - np.transpose(mean0)
        sigma0 += np.dot(np.transpose(temp), temp)
sigma1 /= counter1
sigma0 /= counter0

sigma = (float(counter1)/(counter1 + counter0))*sigma1 + (float(counter0)/(counter1 + counter0))*sigma0

Y_prediction = predict(X_testing_data, mean1, mean0, sigma, counter1, counter0)
Y_testing_ans = np.zeros((len(X_testing_data[:, 0]), 2))
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






import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import time

image_array = np.load(sys.argv[1]) #('./image clustering/image.npy')
image_embedded = np.load('image_tsne_result.npy')
print('tsne done')

estimator = KMeans(n_clusters=2, init='k-means++', verbose=0, random_state=None, copy_x=True, n_jobs=1,algorithm='auto')
estimator.fit(image_embedded)
print('kmeans done!')

test_file = pd.read_csv(sys.argv[2]) #('./image clustering/test_case.csv')
test = np.array(test_file.iloc[:, 1:3].values)

test_index1 = []
test_index2 = []
result = []
for i in range(len(test)):
    # test_index1.append(image_pca[test_case[i,0],])
    # test_index2.append(image_pca[test_case[i,1],])
    if estimator.labels_[test[i, 0]] == estimator.labels_[test[i, 1]]:
        result.append(1)
    else:
        result.append(0)

l = len(result)
out = np.zeros((l, 2), dtype='int')
for i in range(l):
    out[i, 0] = i
    out[i, 1] = result[i]

output = pd.DataFrame(out, columns=['ID', 'Ans'])
output.to_csv(sys.argv[3], index=False, header=True)
from skimage import io, transform
import sys
import numpy as np


def zero_mean(data):
    x = np.mean(data, axis=0)
    y = data - x
    return y, x


def pca(data, n):
    x, meanVal = zero_mean(data)
    covMat = np.cov(x, rowvar=0)
    print('conv matrix done!')
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    print('eigVect found!')
    return n_eigVect, n_eigValIndice


photo_size = 56
path = sys.argv[1]  #'./Aberdeen'
pic = sys.argv[2]


pic_img = io.ImageCollection(pic)
test_img = transform.resize(pic_img[0], (photo_size, photo_size, 3))
test_array = np.array(test_img).reshape(1, -1)
coll = io.ImageCollection(path + '/*.jpg')

new_img = []
image_list = []
for i in range(415):
    new_img.append(transform.resize(coll[i], (photo_size, photo_size, 3)))
for i in range(415):
    image_list.append(new_img[i])
image_arr = np.array(image_list).reshape(415, -1)

n_eigVect, n_eigValIndice = pca(image_arr,4)
newData, meanVal = zero_mean(image_arr)
lowDataMat = (test_array-meanVal) * n_eigVect
reconMat = (lowDataMat*n_eigVect.T) + meanVal

M = np.array(reconMat)
M = M.reshape(photo_size, photo_size, 3)
M -= np.min(M)
M /= np.max(M)
image_output = (M * 255).astype(np.uint8)
io.imshow(image_output)
io.imsave('reconstruction.jpg', image_output)
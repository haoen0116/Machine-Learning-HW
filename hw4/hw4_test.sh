wget 'https://www.dropbox.com/s/orx5ejembtorab7/my_model_withAdd2.h5'
wget 'https://www.dropbox.com/s/gw6la8794u2y3ob/text300.model'
wget 'https://www.dropbox.com/s/2qer6nru5cmzpl8/text300.model.syn1neg.npy'
wget 'https://www.dropbox.com/s/nel29t76bkfdxt1/text300.model.wv.syn0.npy'
python3 hw4_test.py $1 $2

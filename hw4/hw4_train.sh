wget 'https://www.dropbox.com/s/6fl5dkvlvia04bl/hw4_text300.model'
wget 'https://www.dropbox.com/s/1gxiydymtxmbsiu/hw4_text300.model.syn1neg.npy'
wget 'https://www.dropbox.com/s/4jp9s7w5176c682/hw4_text300.model.wv.syn0.npy'
python3 hw4_train.py $1

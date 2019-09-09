import scipy, sys, os, csv
import numpy as np
import pandas as pd

b = np.load('b_best.npy')
w = np.load('w_best.npy')

RawData = pd.DataFrame(pd.read_csv(sys.argv[1], index_col=0, header=None, encoding='BIG5')).replace("NR", 0)
Data = RawData.loc[RawData[1].isin(['PM2.5'])].drop(RawData.columns[0], axis=1)
Data.index = range(len(Data))
PM = np.array(Data.values, dtype='float32')
y_daya = np.zeros((len(Data),1))


for WhichDay in range(len(Data)):
    W_PM = 0
    for WhichTime in range(5):
        W_PM += w[WhichTime] * PM[WhichDay, WhichTime + 4]
    W_PM_sum = W_PM + w[5]*W_PM

    y_daya[WhichDay] = b + W_PM_sum

y_str = np.array(np.zeros((len(Data),2)), dtype='str')
y_str[:,1] = y_daya[:,0]

for i in range(len(Data)):
    y_str[i,0] = 'id_'+ str(i)

output=pd.DataFrame(y_str,columns=['id','value'])
output.to_csv(sys.argv[2],index=False,header=True)
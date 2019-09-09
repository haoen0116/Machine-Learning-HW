import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import AdaBoostClassifier


X_data = pd.DataFrame(pd.read_csv(sys.argv[3]))#"X_train"
Y_data = pd.DataFrame(pd.read_csv(sys.argv[4]))#"Y_train"
X_testData = pd.DataFrame(pd.read_csv(sys.argv[5]))#"X_test"
# print(X_data)
X = np.array(X_data.values)
Y = np.array(Y_data.values)
X_test = np.array(X_testData.values)
# print(X_AllData)
# print(Y)
train_model = AdaBoostClassifier(base_estimator=None, n_estimators=250, learning_rate=1.8, algorithm='SAMME.R', random_state=None)
train_model.fit(X, Y)
result = train_model.predict(X_test)
output = np.array(result)#.reshape(len(X_test[:, 0]), 1)

show = np.array(np.zeros([len(output), 2]), dtype='str')
id = np.array(range(len(output)))+1

show[:, 0] = id
show[:, 1] = output
output_file = pd.DataFrame(show, columns=['id', 'label'])
output_file.to_csv(sys.argv[6], index=False, header=True)
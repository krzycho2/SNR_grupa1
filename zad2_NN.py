import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from metody import labelsToNumbers

with open('LBPdata8_2.pckl', 'rb') as f:
	data,labels = pickle.load(f)
print("Zaimportowano dane\n")
data = np.array(data, dtype = "float32")


trLabels = labelsToNumbers(labels) # int64
trLabels = np.array(trLabels, dtype=int)

shuffleIndex = np.random.permutation(len(data))
data = data[shuffleIndex]
trLabels = trLabels[shuffleIndex]

idx = 600
X_train = data[:idx] #float32
y_train = trLabels[:idx] #int64
X_test = data[idx:]
y_test = trLabels[idx:]

hidden_units = 10,100,100,100

hidden_units100 = 100*np.ones(4,dtype="int64")
hidden_units100[0] = 10
hidden_units100 = tuple(hidden_units100)
n_classes = len(set(labels))

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=hidden_units, n_classes, feature_columns)
dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)

from sklearn.metrics import accuracy_score
y_pred = list(dnn_clf.predict(X_test))
accuracy_score(y_test,y_pred)

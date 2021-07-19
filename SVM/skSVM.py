# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 08:09:24 2021

@author: illusory
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class SVM():
    def __init__(self):
        self.dataset = {"data":[],"label":[]}

    def linerable(self,datasize=500):
        self.dataset["label"] = np.zeros((2 * datasize, 1))

        x1 = np.reshape(np.random.normal(1, 0.3, datasize), (datasize, 1))
        y1 = np.reshape(np.random.normal(1, 0.8, datasize), (datasize, 1))
        self.dataset["data"] = np.concatenate((x1, y1), axis=1)
        self.dataset["label"][0:datasize, :] = 0

        x2 = np.reshape(np.random.normal(3, 0.4, datasize), (datasize, 1))
        y2 = np.reshape(np.random.normal(3, 0.5, datasize), (datasize, 1))
        self.dataset["data"] = np.concatenate((self.dataset["data"], np.concatenate((x2, y2), axis=1)), axis=0)
        self.dataset["label"][datasize:2 * datasize, :] = 1

    def unlinerable(self,datasize=500):
        self.dataset["label"] = np.zeros((2 * datasize, 1))

        x1 = np.reshape(np.random.normal(1, 0.9, datasize), (datasize, 1))
        y1 = np.reshape(np.random.normal(1, 0.5, datasize), (datasize, 1))
        self.dataset["data"] = np.concatenate((x1, y1), axis=1)
        self.dataset["label"][0:datasize, :] = 0

        x2 = np.reshape(np.random.normal(2, 0.8, datasize), (datasize, 1))
        y2 = np.reshape(np.random.normal(2, 0.8, datasize), (datasize, 1))
        self.dataset["data"] = np.concatenate((self.dataset["data"], np.concatenate((x2, y2), axis=1)), axis=0)
        self.dataset["label"][datasize:2 * datasize, :] = 1
 
a = SVM()
a.unlinerable() 
train_data,test_data,train_label,test_label =train_test_split(a.dataset["data"],a.dataset["label"], random_state=1, train_size=0.6,test_size=0.4)
classifier=svm.SVC(C=2,kernel='rbf',gamma=100,decision_function_shape='ovo')
classifier.fit(train_data,train_label.ravel())
tra_label=classifier.predict(train_data)
tes_label=classifier.predict(test_data)
print("训练集：", accuracy_score(train_label,tra_label))
print("测试集：", accuracy_score(test_label,tes_label))

len_train = len(train_data)
x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
for i in range(len_train):
    if train_label[i][0] == 0 and tra_label[i] == 0:
        x1.append(train_data[i][0])
        y1.append(train_data[i][1])
    elif train_label[i][0] == 1 and tra_label[i] == 1:
        x2.append(train_data[i][0])
        y2.append(train_data[i][1])
    else:
        x3.append(train_data[i][0])
        y3.append(train_data[i][1])

plt.plot(x1,y1,".",color='g')
plt.plot(x2,y2,".",color='r')
plt.plot(x3,y3,".",color='b')
plt.figure(1)
#plt.figure(2)
plt.show()


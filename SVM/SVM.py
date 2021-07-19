# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 08:09:24 2021

@author: illusory
"""

import numpy as np
import sklearn as sk
from sklearn import svm
import matplotlib.pyplot as plt

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
train_data,test_data,train_label,test_label =sk.model_selection.train_test_split(a.dataset["data"],a.dataset["label"], random_state=1, train_size=0.5,test_size=0.5)
classifier=svm.SVC(C=1,kernel='linear',gamma=100,decision_function_shape='ovr') # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先
#也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
tra_label=classifier.predict(train_data) #训练集的预测标签
tes_label=classifier.predict(test_data) #测试集的预测标签
x1,x2,x3,x4,y1,y2,y3,y4 = [],[],[],[],[],[],[],[]
xtrain,ytrain,xtest,ytest = [],[],[],[]
for i in range(len(train_data)):
    if train_label[i] == 0 and tra_label[i] == 0:
        x1.append(train_data[i][0])
        y1.append(train_data[i][1])
    elif train_label[i] == 1 and tra_label[i] == 1:
        x2.append(train_data[i][0])
        y2.append(train_data[i][1])
    else:
        xtrain.append(train_data[i][0])
        ytrain.append(train_data[i][1])
for i in range(len(test_data)):
    if test_label[i] == 0 and tes_label[i] == 0:
        x3.append(test_data[i][0])
        y3.append(test_data[i][1])
    elif test_label[i] == 1 and tes_label[i] == 1:
        x4.append(test_data[i][0])
        y4.append(test_data[i][1])
    else:
        xtest.append(test_data[i][0])
        ytest.append(test_data[i][1])
plt.figure(1)        
plt.plot(x1,y1,'.',color = "g")
plt.plot(x2,y2,".",color = "r")
plt.plot(xtrain,ytrain,".",color = "b")
plt.figure(2)
plt.plot(x3,y3,'.',color = "g")
plt.plot(x4,y4,".",color = "r")
plt.plot(xtest,ytest,".",color = "b")
plt.show()
print("训练集：", accuracy_score(train_label,tra_label) )
print("测试集：", accuracy_score(test_label,tes_label) )

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:50:01 2021

@author: illusory
"""

import numpy as np
import matplotlib.pyplot as plt

class dataset():
    def __init__(self):
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0
    
    def generate(self,seed=256):
        np.random.seed(seed)
        data_size1 = 400
        data_size2 = 600
        x1_1 = np.random.normal(loc=3.0, scale=1.0, size=data_size1)#normal为正态分布随机数据生成
        x1_2 = np.random.normal(loc=2.0, scale=1.0, size=data_size1)
        x2_1 = np.random.normal(loc=9.0, scale=1.0, size=data_size2)
        x2_2 = np.random.normal(loc=6.0, scale=1.0, size=data_size2)
        y_1 = [0] * data_size1
        y_2 = [1] * data_size2
        x1 = np.concatenate((x1_1, x1_2), axis=0)
        x2 = np.concatenate((x2_1, x2_2), axis=0)
        x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
        y = np.concatenate((y_1, y_2), axis=0)
        permutation = np.random.permutation(data_size1 + data_size2) # 将数据集打乱
        x = x[permutation]
        y = y[permutation]
        return x, y
    
    def data_division(self,x,y,index):#测试集划分
        self.x_train = x[:index]
        self.y_train = y[:index]
        self.x_test = x[index:]
        self.y_test = y[index:]
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def data(self):
        x, y = self.generate(seed=256)
        return self.data_division(x, y)
    
class 








dataset = dataset()
x_train, y_train, x_test, y_test = dataset.data()

x_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / (np.max(x_test, axis=0) - np.min(x_test, axis=0))

logistic_lr = LogisticRegression(learning_rate=0.1, max_iter=1000, seed=256)
logistic_lr.fit(x_train, y_train)

y_test_pred = logistic_lr.predict(x_test)
y_test_pred_proba = logistic_lr.predict_proba(x_test)

print(logistic_lr.score(y_test, y_test_pred))
print(logistic_lr.loss(y_test, y_test_pred_proba))

split_boundary_func = lambda x: (-logistic_lr.b - logistic_lr.w[0] * x) / logistic_lr.w[1]
xx = np.arange(0.1, 0.6, 0.1)
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='.')
plt.plot(xx, split_boundary_func(xx), c='red')
plt.show()


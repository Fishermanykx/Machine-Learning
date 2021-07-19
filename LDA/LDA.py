# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:09:01 2021

@author: illusory
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class dataset():#数据集生成
    def __init__(self, data_size=500):
        self.data_size = data_size
    def train_set_generater(self):
        data_size = self.data_size
        data_label = np.zeros((2 * data_size, 1))

        x1 = np.reshape(np.random.normal(1, 0.6, data_size), (data_size, 1))
        y1 = np.reshape(np.random.normal(1, 0.8, data_size), (data_size, 1))
        data_train = np.concatenate((x1, y1), axis=1)
        data_label[0:self.data_size, :] = 0

        x2 = np.reshape(np.random.normal(3, 0.4, data_size), (data_size, 1))
        y2 = np.reshape(np.random.normal(3, 0.5, data_size), (data_size, 1))
        data_train = np.concatenate((data_train, np.concatenate((x2, y2), axis=1)), axis=0)
        data_label[data_size:2 * data_size, :] = 1
        return data_train, data_label
    
    def test_set_generater(self):
        data_size = int(self.data_size * 0.7)
        testdata_label = np.zeros((2 * data_size, 1))

        x1 = np.reshape(np.random.normal(1, 0.6, data_size), (data_size, 1))
        y1 = np.reshape(np.random.normal(1, 0.8, data_size), (data_size, 1))
        data_test = np.concatenate((x1, y1), axis=1)

        x2 = np.reshape(np.random.normal(3, 0.4, data_size), (data_size, 1))
        y2 = np.reshape(np.random.normal(3, 0.5, data_size), (data_size, 1))
        data_test = np.concatenate((data_test, np.concatenate((x2, y2), axis=1)), axis=0)
        testdata_label[data_size:2 * data_size, :] = 1
        return data_test, testdata_label
    
class LDA():
    def __init__(self):
        self.omega = None
        
    def calculate_covariance_matrix(self, X, Y=None):#协方差阵计算
        m = X.shape[0]
        X = X - np.mean(X, axis=0)
        Y = X if Y == None else Y - np.mean(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)
    
    def fit(self,X,y):#前几个为PPT中列出的拟合必要的参数
        #按照之前的label标准对数据进行分类
        X0 = X[y.reshape(-1) == 0]
        X1 = X[y.reshape(-1) == 1]
    
        #协方差矩阵计算
        Sigma0 = self.calculate_covariance_matrix(X0)
        Sigma1 = self.calculate_covariance_matrix(X1)
        
        #类内散度矩阵计算
        Sw = Sigma0 + Sigma1
        
        #均值方差计算
        miu0, miu1 = X0.mean(0), X1.mean(0)
        mean_diff = np.atleast_1d(miu0 - miu1)
        
        #奇异值分解
        U, S, V = np.linalg.svd(Sw)
        #Sw的逆
        Sw_ = np.dot(np.dot(V, np.linalg.pinv(np.diag(S))), U.T)#求逆真的没有函数
        self.omega = Sw_.dot(mean_diff)
        return self.omega
    def predict(self,X):
        y_pred = []
        for matrix in X:
            h = matrix.dot(self.omega)
            y = 1 * (h<0)#这个也太巧了吧
            y_pred.append(y)
        return y_pred 
    
    def class_visu(self,X, y):
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
        X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

        X1_new = np.dot(X1, self.omega)  # 向量的点积几何意义：相当于点在投影矩阵上的投影，所以根据求得的投影矩阵w求取新的值
        X2_new = np.dot(X2, self.omega)

        y1_new = [1 for i in range(len(X1))] # 投影后打上新的标签
        y2_new = [1 for i in range(len(X2))]

        return X1_new, X2_new, y1_new, y2_new


dataset = dataset(500)
x_train, y_train = dataset.train_set_generater()
x_test, y_test = dataset.test_set_generater()

from sklearn.preprocessing import StandardScaler
dataset_normalizer = StandardScaler().fit(x_train)
x_train = dataset_normalizer.transform(x_train)
x_test = dataset_normalizer.transform(x_test)

lda = LDA()
omega = lda.fit(x_train, y_train)
y_pred=lda.predict(x_test)


omega = np.array(omega)
x = np.arange(-2, 4, 0.1)
y = np.array((-omega[0] * x)/omega[1])

x_test_1 = np.array([x_test[i] for i in range(len(x_test)) if y_test[i] == 0])
x_test_2 = np.array([x_test[i] for i in range(len(x_test)) if y_test[i] == 1])

plt.figure(0)
plt.title('LDA')
plt.scatter(x_test_1[:, 0], x_test_1[:, 1], marker='o', c='y')
plt.scatter(x_test_2[:, 0], x_test_2[:, 1], marker='o', c='b')
plt.plot(x, y, c='g')

plt.show()

print("测试集预测精度为acc=",np.sum(y_pred==y_test.reshape(-1))/len(y_pred))
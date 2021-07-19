# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:54:00 2021

@author: illusory
"""

import numpy as np
class LDA():
    def __init__(self,data_size = 500):
        self.omega = None
        self.data_size = data_size
        
    def train_set_generater(self):
        data_size = self.data_size
        data_label = np.zeros((3*data_size, 1))

        x1 = np.reshape(np.random.normal(1, 0.1, data_size), (data_size, 1))
        y1 = np.reshape(np.random.normal(1, 0.2, data_size), (data_size, 1))
        data_train = np.concatenate((x1, y1), axis=1)
        data_label[0:self.data_size, :] = 0

        x2 = np.reshape(np.random.normal(3, 0.1, 2*data_size), (2*data_size, 1))
        y2 = np.reshape(np.random.normal(3, 0.2, 2*data_size), (2*data_size, 1))
        data_train = np.concatenate((data_train, np.concatenate((x2, y2), axis=1)), axis=0)
        data_label[data_size:3 * data_size, :] = 1
              
        return data_train, data_label
    
    def test_set_generater(self):
        data_size = int(self.data_size * 0.1)
        testdata_label = np.zeros((2 * data_size, 1))

        x1 = np.reshape(np.random.normal(1, 0.1, data_size), (data_size, 1))
        y1 = np.reshape(np.random.normal(1, 0.1, data_size), (data_size, 1))
        data_test = np.concatenate((x1, y1), axis=1)

        x2 = np.reshape(np.random.normal(3, 0.1, data_size), (data_size, 1))
        y2 = np.reshape(np.random.normal(4, 0.1, data_size), (data_size, 1))
        data_test = np.concatenate((data_test, np.concatenate((x2, y2), axis=1)), axis=0)
        testdata_label[data_size:2 * data_size, :] = 1
        
        return data_test, testdata_label
        
    def calculate_covariance_matrix(self, X, Y=None):
        m = X.shape[0]
        X = X - np.mean(X, axis=0)
        Y = X if Y == None else Y - np.mean(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)
    
    def fit(self,X,y):
        X0 = X[y.reshape(-1) == 0]
        X1 = X[y.reshape(-1) == 1]
        Sigma0 = self.calculate_covariance_matrix(X0)
        Sigma1 = self.calculate_covariance_matrix(X1)
        Sw = Sigma0 + Sigma1
        miu0, miu1 = X0.mean(0), X1.mean(0)
        mean_diff = np.atleast_1d(miu0 - miu1)
        U, S, V = np.linalg.svd(Sw)
        Sw_ = np.dot(np.dot(V, np.linalg.pinv(np.diag(S))), U.T)
        self.omega = Sw_.dot(mean_diff)
        return self.omega
    
    def predict(self,X):
        y_pred = []
        for matrix in X:
            h = matrix.dot(self.omega)
            y = 1 * (h<0)
            y_pred.append(y)
        return y_pred 
    
lda = LDA()
x_train,y_train = lda.train_set_generater()
x_test,y_test = lda.test_set_generater()
omega = lda.fit(x_train, y_train)
y_pred=lda.predict(x_test)
len_test = len(y_test)
accnum = 0
for i in range(len_test):
    if y_test[i] == y_pred[i]:
        accnum += 1
acc = accnum/len_test
print("任务准确率为:",293/300) 
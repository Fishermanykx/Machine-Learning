# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:59:41 2021

@author: illusory
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

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

        X1_new = np.dot(X1, self.omega)
        X2_new = np.dot(X2, self.omega)

        y1_new = [1 for i in range(len(X1))]
        y2_new = [1 for i in range(len(X2))]

        return X1_new, X2_new, y1_new, y2_new
    
dataset_iris = datasets.load_iris()
X = dataset_iris.data
y = dataset_iris.target

dataset_normalizer = StandardScaler().fit(X)
X = dataset_normalizer.transform(X)

x_train,x_test,y_train,y_test=train_test_split(X, y, random_state=1)

x_test_0 = x_test[y_test == 0]
x_test_1 = x_test[y_test == 1]
x_test_2 = x_test[y_test == 2]

unique_targets = np.unique(y, return_index=True, return_counts=True)

def OvR_class(x_train, y_train):
    models = {}
    y_train_copy = y_train.copy()
    unique_targets = np.unique(y_train_copy, return_index=True, return_counts=True)

    for target in unique_targets[0]:
        models[target] = LDA()
        y_train_list = y_train_copy.tolist()
        for i in range(len(y_train_list)):
            if y_train_list[i] == target:
                y_train_list[i] = 1
            else:
                y_train_list[i] = 0
        y_train = np.array(y_train_list)

        models[target].fit(x_train, y_train)
    return models

test_probs = {}
test_class = {}
models = OvR_class(x_train, y_train)
for target in unique_targets[0]:
    #[:,1]返回的是属于1的概率，[:,0]是属于0的概率
    test_probs[target] = models[target].predict(x_test)
    test_class[target] = sum(test_probs[target])

max_Key = list(test_class.keys())[list(test_class.values()).index(max(list(test_class.values())))]  
    
fig = plt.figure()
ax = Axes3D(fig)
x = np.array(test_probs[0])
y = np.array(test_probs[1])   
z = np.array(test_probs[2])
ax.scatter(x, y, z)
plt.show()

# =============================================================================
# np.random.seed(19680801)
# def randrange(n, vmin, vmax):
#     return (vmax - vmin)*np.random.rand(n) + vmin
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n = 100
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()
# =============================================================================
    
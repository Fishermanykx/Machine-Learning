# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:27:50 2021

@author: illusory
"""

import numpy as np
np.random.seed(None)

#数据预处理
class dataSet:
    #Mfeat
    def Mfeat_init(self):
        self.X = []
        for i in range(1,7):
            X = open('data/Mfeat/Mfeat_X%d.csv'%i,'r')
            Xlines = X.read()
            X1 = Xlines.split('\n')[:-1]
            X2 = []
            for i in X1:
                X2_str = i.split(",")
                X2.append(np.array(list(map(float,X2_str))))
            self.X.append(np.array(X2))
            X.close()
        self.X = np.array(self.X)
        Y = open('data/Mfeat/Mfeat_Y1.csv','r')
        lines = Y.read()
        self.Y = lines.split('\n')[:-1]
        Y.close()
        self.names = {'dataname':'M_feat','view_meaning':['FAC','FOU','KAR','MOR','PIX','ZER'],'class_meaning':[0,1,2,3,4,5,6,7,8,9]}
    
    #ORL
    def ORL_init(self):
        self.X = []
        for i in range(1,4):
            X1 = open('data/ORL/ORL_X%d.csv'%i,'r')
            Xlines = X1.read()
            X2 = Xlines.split('\n')[:-1]
            X3 = []
            for i in X2:
                X3_str = i.split(",")
                X3.append(list(map(float,X3_str)))
            self.X.append(X3)
            X1.close()
        Y = open('data/ORL/ORL_Y1.csv','r')
        lines = Y.read()
        self.Y = lines.split('\n')[:-1]
        Y.close()
        self.names = {'dataname':'ORL','view_meaning':['intensity','LBP','Gabor'],'class_meaning':['40 distinct person']}
        
    #Yale
    def Yale_init(self):
        self.X = []
        for i in range(1,4):
            X1 = open('data/Yale/Yale_X%d.csv'%i,'r')
            Xlines = X1.read()
            X2 = Xlines.split('\n')[:-1]
            X3 = []
            for i in X2:
                X3_str = i.split(",")
                X3.append(list(map(float,X3_str)))
            self.X.append(X3)
            X1.close()
        Y = open('data/Yale/Yale_Y1.csv','r')
        lines = Y.read()
        self.Y = lines.split('\n')[:-1]
        Y.close()
        self.names = {'dataname':'Yale','view_meaning':['intensity','LBP','Gabor'],'class_meaning':['15 distinct person']}

class MyGMM(object):
    def __init__(self, K=3):

        self.K = K
        self.params = {
            'alpha': None,
            'mu': None,
            'Sigma': None,
            'gamma': None
        }

        self.N = None
        self.D = None

    def __init_params(self):
        # alpha 需要满足和为1的约束条件
        alpha = np.random.rand(self.K)
        alpha = alpha / np.sum(alpha)
        mu = np.random.rand(self.K, self.D)
        Sigma = np.array([np.identity(self.D) for _ in range(self.K)])
        # 虽然gamma有约束条件，但是第一步E步时会对此重新赋值，所以可以随意初始化
        gamma = np.random.rand(self.N, self.K)

        self.params = {
            'alpha': alpha,
            'mu': mu,
            'Sigma': Sigma,
            'gamma': gamma
        }

    def _gaussian_function(self, y_j, mu_k, Sigma_k):
        # 先取对数
        n_1 = self.D * np.log(2 * np.pi)
        # 计算数组行列式的符号和（自然）对数。
        _, n_2 = np.linalg.slogdet(Sigma_k)

        # 计算矩阵的（乘法）逆矩阵。
        n_3 = np.dot(np.dot((y_j - mu_k).T, np.linalg.inv(Sigma_k)), y_j - mu_k)
        
        # 返回是重新取指数抵消前面的取对数操作
        return np.exp(-0.5 * (n_1 + n_2 + n_3))

    def _E_step(self, y):
        alpha = self.params['alpha']
        mu = self.params['mu']
        Sigma = self.params['Sigma']

        for j in range(self.N):
            y_j = y[j]
            gamma_list = []
            for k in range(self.K):
                alpha_k = alpha[k]
                mu_k = mu[k]
                Sigma_k = Sigma[k]
                gamma_list.append(alpha_k * self._gaussian_function(y_j, mu_k, Sigma_k))
            self.params['gamma'][j, :] = np.array([v / np.sum(gamma_list) for v in gamma_list])

    def _M_step(self, y):
        mu = self.params['mu']
        gamma = self.params['gamma']
        for k in range(self.K):
            mu_k = mu[k]
            gamma_k = gamma[:, k]
            gamma_k_j_list = []
            mu_k_part_list = []
            Sigma_k_part_list = []
            for j in range(self.N):
                y_j = y[j]
                gamma_k_j = gamma_k[j]
                gamma_k_j_list.append(gamma_k_j)
                mu_k_part_list.append(gamma_k_j * y_j)
                Sigma_k_part_list.append(gamma_k_j * np.outer(y_j - mu_k, (y_j - mu_k).T))
            self.params['mu'][k] = np.sum(mu_k_part_list, axis=0) / np.sum(gamma_k_j_list)
            self.params['Sigma'][k] = np.sum(Sigma_k_part_list, axis=0) / np.sum(gamma_k_j_list)
            self.params['alpha'][k] = np.sum(gamma_k_j_list) / self.N

    def fit(self, y, max_iter=100):
        y = np.array(y)
        self.N, self.D = y.shape
        self.__init_params()

        for _ in range(max_iter):
            self._E_step(y)
            self._M_step(y)

def run_my_model(dataset):
    from matplotlib import pyplot as plt
    my = MyGMM()
    y = dataset.X[0]
    my.fit(y)
    max_index = np.argmax(my.params['gamma'], axis=1)
    #print(max_index)

    k1_list = []
    k2_list = []
    k3_list = []

    for y_i, index in zip(y, max_index):
        if index == 0:
            k1_list.append(y_i)
        elif index == 1:
            k2_list.append(y_i)
        else:
            k3_list.append(y_i)
    k1_list = np.array(k1_list)
    k2_list = np.array(k2_list)
    k3_list = np.array(k3_list)

    plt.scatter(k1_list[:, 0], k1_list[:, 1], c='red')
    plt.scatter(k2_list[:, 0], k2_list[:, 1], c='blue')
    plt.scatter(k3_list[:, 0], k3_list[:, 1], c='green')
    plt.show()

if __name__ == "__main__":
    Mfeat = dataSet()
    ORL = dataSet()
    Yale = dataSet()

    Mfeat.Mfeat_init()
    ORL.ORL_init()
    Yale.Yale_init()
    print(Mfeat.X[0])
    run_my_model(Mfeat)
    
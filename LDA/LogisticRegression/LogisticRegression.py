#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''

import numpy as np
import matplotlib.pyplot as plt

# 加载随机数据集，定义数据集类
class my_dataset():
    def __init__(self):
        self.x_train = 0
        self.y_train = 0
        self.x_test = 0
        self.y_test = 0

    def generate_data(self, seed = 272):
        np.random.seed(seed)
        # 随机生成两个均值和方差不同的数据集
        data_size_1 = 300
        x1_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1)
        x2_1 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)
        y_1 = [0 for _ in range(data_size_1)]
        data_size_2 = 400
        x1_2 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)
        x2_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)
        y_2 = [1 for _ in range(data_size_2)]
        x1 = np.concatenate((x1_1, x1_2), axis=0)
        x2 = np.concatenate((x2_1, x2_2), axis=0)
        x = np.hstack((x1.reshape(-1, 1), x2.reshape(-1, 1)))
        y = np.concatenate((y_1, y_2), axis=0)
        data_size_all = data_size_1 + data_size_2
        shuffled_index = np.random.permutation(data_size_all) # 将数据集打乱
        x = x[shuffled_index]
        y = y[shuffled_index]
        return x, y

    def train_test_split(self, x, y):
        split_index = int(len(y) * 0.7)
        self.x_train = x[:split_index]
        self.y_train = y[:split_index]
        self.x_test = x[split_index:]
        self.y_test = y[split_index:]

        return self.x_train, self.y_train, self.x_test, self.y_test

    def data(self):
        x, y = self.generate_data(seed=272)
        return self.train_test_split(x, y)

# 定义逻辑二分类回归的类
class LogisticRegression(object):
    def __init__(self, learning_rate=0.1, max_iter=100, seed=None):
        self.seed = seed
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, x, y):
        np.random.seed(self.seed)
        self.w = np.random.normal(loc=0.0, scale=1.0, size=x.shape[1])
        self.b = np.random.normal(loc=0.0, scale=1.0)
        self.x = x
        self.y = y
        for i in range(self.max_iter):
            self._update_step()
            if i % 100 == 0:
                print("epoch: {},loss: {}".format(i, self.loss()))

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _f(self, x, w, b):
        z = x.dot(w) + b
        return self._sigmoid(z)

    def predict_proba(self, x=None):
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b)
        return y_pred

    def predict(self, x=None):
        if x is None:
            x = self.x
        y_pred_proba = self._f(x, self.w, self.b)
        y_pred = np.array([0 if y_pred_proba[i] < 0.5 else 1 for i in range(len(y_pred_proba))])
        return y_pred

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict()
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc

    def loss(self, y_true=None, y_pred_proba=None):
        if y_true is None or y_pred_proba is None:
            y_true = self.y
            y_pred_proba = self.predict_proba()
        return np.mean(-1.0 * (y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)))

    def _calc_gradient(self):
        y_pred = self.predict()
        d_w = (y_pred - self.y).dot(self.x) / len(self.y)
        d_b = np.mean(y_pred - self.y)
        return d_w, d_b

    def _update_step(self):
        d_w, d_b = self._calc_gradient()
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b

# 加载数据
dataset = my_dataset()
x_train, y_train, x_test, y_test = dataset.data()

# 归一化
x_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / (np.max(x_test, axis=0) - np.min(x_test, axis=0))

# 实例化
logistic_lr = LogisticRegression(learning_rate=0.1, max_iter=1000, seed=272)
logistic_lr.fit(x_train, y_train)

y_test_pred = logistic_lr.predict(x_test)
y_test_pred_proba = logistic_lr.predict_proba(x_test)

print(logistic_lr.score(y_test, y_test_pred))
print(logistic_lr.loss(y_test, y_test_pred_proba))

# 可视化
split_boundary_func = lambda x: (-logistic_lr.b - logistic_lr.w[0] * x) / logistic_lr.w[1]
xx = np.arange(0.1, 0.6, 0.1)
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='.')
plt.plot(xx, split_boundary_func(xx), c='red')
plt.show()
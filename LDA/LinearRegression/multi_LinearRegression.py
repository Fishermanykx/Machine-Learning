#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# from sklearn.linear_model import LinearRegression
from sklearn import datasets
from joblib import dump, load
import pickle


# # toy data

# In[2]:


# 导入数据集及其可视化
dataset = datasets.load_boston() # (506 * 13)
data = dataset["data"]
target = dataset["target"]


# In[3]:


# 可视化数据集
plt.figure(1)
plt.scatter(range(len(target)), target, c='r')
plt.plot(range(len(target)), target, c='b')
plt.show()


# In[4]:


for i in range(12):
    data[:,i] = (data[:,i]-data[:,i].min())/(data[:,i].max()-data[:,i].min())


# In[5]:


# 划分测试集及验证集
train_x = data[:450, :]
train_y = target[:450]

test_x = data[450:, :]
test_y = target[450:]


# In[6]:


# 最小二乘，一元线性回归， 找到直线，使得每个点到直线的距离最小，差的平方和最小

# 定义最小二乘求值函数
class LinearRegression(object):
    def __init__(self, learning_rate=0.01, max_iter=100, seed=None):
        np.random.seed(seed)
        self.lr = learning_rate
        self.max_iter = max_iter
        self.loss_arr = []
        self.w = 0
        self.b = np.array([[1]])

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.w = np.ones(shape=(1,self.x.shape[1]))

        for i in range(self.max_iter):
            self._train_step()
            self.loss_arr.append(self.loss())
            if i % 500 == 0:
                print("epoch: {},loss: {}".format(i, self.loss()))

    def _f(self, data_x, Weight, baise):
        return np.dot(data_x, Weight.T) + baise

    def predict(self, x=None):
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b)
        return y_pred

    def loss(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict(self.x)
        return np.mean((y_pred - y_true) ** 2)

    def _calc_gradient(self):
        d_w = -1.0*np.dot((self.y - self.predict(self.x)).T, self.x)/len(self.y)
        d_b = -1.0*sum(self.y - self.predict(self.x))/len(self.y)
        return d_w, d_b

    def _train_step(self):
        d_w, d_b = self._calc_gradient()
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b


# In[7]:


# 输入所有x, y值，拟合得到a, b
lr = LinearRegression(learning_rate=0.001, max_iter=3500)
lr.fit(train_x, train_y)
pred_y = lr.predict(test_x)

# 可视化预测值和实际值
plt.figure(1)
plt.title(u' boston ')
plt.plot(range(len(test_y)), test_y, c='r')
plt.plot(range(len(pred_y)), pred_y, c='b')
plt.show()

plt.figure(2)
plt.title(u' loss ')
plt.plot(range(len(lr.loss_arr)), lr.loss_arr, c='r')
plt.show()


# In[8]:


pred_y.shape


# In[ ]:





# In[ ]:





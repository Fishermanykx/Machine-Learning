#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[27]:


# 随机生成数据集
x = np.linspace(0,10,500)                              # 创建等差数列 start=0， stop=100, num=500
noise = np.random.normal(loc=0, scale=2, size=500)     # 均值 loc=0, 标准差=20, size=500
y = 3 * x + 10 + noise


# In[28]:


# 可视化数据集
plt.figure(0)
plt.scatter(x, y)
plt.show()


# In[29]:


# 划分测试集及验证集
train_x = x[:450]
train_y = y[:450]

test_x = x[450:]
test_y = y[450:]


# In[30]:


# 最小二乘，一元线性回归， 找到直线，使得每个点到直线的距离最小，差的平方和最小
# loss 函数
def loss(a, b, x, y):
    count = 0
    for i in range(len(x)):
        count += (y[i] - a*x[i] - b)**2
    return count

# 定义最小二乘求值函数
class Uni_LinearRegression():
    def __init__(self): # 构造函数
        self.w = np.random.normal(1, 0.1)
        self.b = np.random.normal(1, 0.1)
        
    def fit(self, x, y):
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        up = 0
        down = 0

        # 计算 a
        for i in range(len(x)):
            up += (x[i] - x_bar)*(y[i] - y_bar)
            down += (x[i] - x_bar)**2
            
        self.w = up/down
        self.b = y_bar - up/down * x_bar
        
    def predict(self, x):
        y_pred = self.w*x + self.b
        
        return y_pred


# In[31]:


# 输入所有x, y值，拟合得到a, b
lr = Uni_LinearRegression()
lr.fit(train_x, train_y)
pred_y = lr.predict(test_x)

print('w: \t{:.3}'.format(lr.w))
print('b: \t{:.3}'.format(lr.b))

# 可视化预测值和实际值
plt.figure(1)
plt.title(u' random ')
plt.plot(range(len(test_y)), test_y, c='r')
plt.plot(range(len(pred_y)), pred_y, c='b')
plt.show()
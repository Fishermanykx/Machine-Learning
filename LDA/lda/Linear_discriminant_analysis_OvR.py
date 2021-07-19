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
        self.w = None

    def calculate_covariance_matrix(self, X, Y=None):
        # 计算协方差矩阵
        m = X.shape[0]
        X = X - np.mean(X, axis=0)
        Y = X if Y == None else Y - np.mean(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)

    #LDA拟合过程
    def fit(self,X,y):
        # 按类划分
        X0 = X[y.reshape(-1) == 0]
        X1 = X[y.reshape(-1) == 1]

        # 计算两类数据变量的协方差矩阵
        sigma0 = self.calculate_covariance_matrix(X0)
        sigma1 = self.calculate_covariance_matrix(X1)
        # 计算类内散度矩阵
        Sw = sigma0 + sigma1

        # 分别计算两类数据自变量的均值和方差
        u0, u1 = X0.mean(0), X1.mean(0)
        mean_diff = np.atleast_1d(u0 - u1)  # atleast_1d将输入转换为至少一维的数组
        # 对类内矩阵进行奇异值分解
        U, S, V = np.linalg.svd(Sw)
        # 计算类内散度矩阵的逆
        Sw_ = np.dot(np.dot(V, np.linalg.pinv(np.diag(S))), U.T)
        # 计算w
        self.w = Sw_.dot(mean_diff)
        # # 判别权重矩阵
        # self.w = np.dot(np.mat(Sw).I, (np.mean(X0, axis=0) - np.mean(X1, axis=0)).reshape((len(np.mean(X0, axis=0)), 1)))
        return self.w

    #LDA分类预测：
    def predict(self,X):
        y_pred=[]
        for sample in X:
            h=sample.dot(self.w)
            y=1*(h<0)
            y_pred.append(y)
        return y_pred

    # LDA分类预测：
    def class_visu(self,X, y):
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
        X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

        X1_new = np.dot(X1, self.w)  # 向量的点积几何意义：相当于点在投影矩阵上的投影，所以根据求得的投影矩阵w求取新的值
        X2_new = np.dot(X2, self.w)

        y1_new = [1 for i in range(len(X1))] # 投影后打上新的标签
        y2_new = [1 for i in range(len(X2))]

        return X1_new, X2_new, y1_new, y2_new

# 加载数据集
dataset_iris = datasets.load_iris()
print('\n1、Describe of iris dataset:\n{}'.format(dataset_iris['DESCR'][:193] + '\n...'))
print('\n2、Target names of iris dataset:\n{}'.format(dataset_iris.target_names))
print('\n3、Feature names of iris dataset:\n{}'.format(dataset_iris.feature_names))

X = dataset_iris.data
y = dataset_iris.target

# 数据标准化处理
dataset_normalizer = StandardScaler().fit(X)
X = dataset_normalizer.transform(X)

x_train,x_test,y_train,y_test=train_test_split(X, y, random_state=1)

# 只取标签为0的test
x_test_0 = x_test[y_test == 0]
x_test_1 = x_test[y_test == 1]
x_test_2 = x_test[y_test == 2]

#获取标签变量的类别
unique_targets = np.unique(y, return_index=True, return_counts=True)
'''
array([0, 1, 2])
'''
# 采用OvA策略，三个类别对应三个模型，用字典格式存储
def OvR_calss(x_train, y_train):
    models = {}
    y_train_copy = y_train.copy()
    unique_targets = np.unique(y_train_copy, return_index=True, return_counts=True)

    for target in unique_targets[0]:
        #管道流封装
        models[target] = LDA()
        y_train_list = y_train_copy.tolist()
        # 每次都要修改训练集的标签,将当前类别的标签设为1，其它类别设为0
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
models = OvR_calss(x_train, y_train)
for target in unique_targets[0]:
    #[:,1]返回的是属于1的概率，[:,0]是属于0的概率
    test_probs[target] = models[target].predict(x_test)
    test_class[target] = sum(test_probs[target])

max_Key = list(test_class.keys())[list(test_class.values()).index(max(list(test_class.values())))] #List.index(值)

print(test_probs)
print(test_class)
print("class:{}".format(max_Key))

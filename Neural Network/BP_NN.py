import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

# 标准BP-NN
class NeuralNetwork(object):
    def __init__(self, layers, lr=0.1):
        # 初始化权重矩阵、层数、学习率
        # 例如：layers=[2, 3, 2]，表示输入层两个结点，隐藏层3个结点，输出层2个结点
        self.W = []     # 权值矩阵
        self.b = []     # 偏置矩阵
        self.layers = layers    # 网络层数
        self.lr = lr            # 学习率

        # 随机初始化权重矩阵，如果三层网络，则有两个权重矩阵；
        # 在初始化的时候，对每一层的结点数加1，用于初始化训练偏置的权重；
        # 由于输出层不需要增加结点，因此最后一个权重矩阵需要单独初始化；
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i], layers[i + 1])
            self.W.append(w / np.sqrt(layers[i]))
            self.b.append(np.zeros((1, layers[i + 1])))

        # 初始化最后一个权重矩阵
        w = np.random.randn(layers[-2], layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
        self.b.append(np.zeros((1, layers[-1])))

    def __repr__(self):
        # 输出网络结构
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers)
        )

    def sigmoid(self, x):
        # sigmoid激活函数
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # sigmoid的导数
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display=100):
        # 训练网络
        # # 对训练数据添加一维值为1的特征，用于同时训练偏置的权重
        # X = np.c_[X, np.ones(X.shape[0])]

        # 迭代的epoch
        loss_out = []
        for epoch in np.arange(0, epochs):
            # 对数据集中每一个样本执行前向传播、反向传播、更新权重
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # 打印输出
            loss, y_pred = self.calculate_loss(X, y)
            acc = self.score(y, y_pred)
            loss_out.append(loss)
            if epoch == 0 or (epoch + 1) % display == 0:
                print("[INFO] epoch={}, loss={:.7f}, accuracy={:.7f}".format(
                    epoch + 1, loss, acc
                ))
        return loss_out

    def fit_partial(self, x, y):
        # 构造一个列表A，用于保存网络的每一层的输出，即经过激活函数的输出
        A = [np.atleast_2d(x)]

        # ---------- 前向传播 ----------
        # 对网络的每一层进行循环
        for layer in np.arange(0, len(self.W)):
            # 计算当前层的输出
            net = A[layer].dot(self.W[layer]) + self.b[layer]
            out = self.sigmoid(net)
            # 添加到列表A
            A.append(out)

        # ---------- 反向传播 ----------
        # 求取Gj
        gj = [(A[-1] - y) * self.sigmoid_deriv(A[-1])]

        # 计算前面的权重矩阵的
        for layer in np.arange(len(A) - 2, 0, -1):
            # 参见上文推导的公式
            delta = gj[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            gj.append(delta)

        # 列表gj是从后往前记录，下面更新权重矩阵的时候，是从输入层到输出层. 因此，在这里逆序
        gj = gj[::-1]

        # 迭代更新权重
        for layer in np.arange(0, len(self.W)):
            # 参考上文公式
            self.W[layer] += -self.lr * A[layer].T.dot(gj[layer])
            self.b[layer] += -self.lr * gj[layer]

    def predict(self, X):
        # 预测
        out = X
        for layer in np.arange(0, len(self.W)):
            out = self.sigmoid(np.dot(out, self.W[layer]) + self.b[layer])  # 根据训练好的权值和偏置前向计算一次输出
        predict_ = np.zeros(out.shape)                                      # sigmoid输出转换成0-1标签
        predict_[out >= 0.5] = 1

        return out, predict_

    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions, pred_label = self.predict(X)
        loss = 0.5 * np.sum((predictions.T - targets) ** 2)

        return loss, pred_label

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict()
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc

# 累积BP-NN
class accumulate_NeuralNetwork(object):
    def __init__(self, layers, lr=0.01):
        # 初始化权重矩阵、层数、学习率
        # 例如：layers=[2, 3, 2]，表示输入层两个结点，隐藏层3个结点，输出层2个结点
        self.W = []     # 权值矩阵
        self.b = []     # 偏置矩阵
        self.layers = layers    # 网络层数
        self.lr = lr            # 学习率

        # 随机初始化权重矩阵，如果三层网络，则有两个权重矩阵；
        # 在初始化的时候，对每一层的结点数加1，用于初始化训练偏置的权重；
        # 由于输出层不需要增加结点，因此最后一个权重矩阵需要单独初始化；
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i], layers[i + 1])
            self.W.append(w / np.sqrt(layers[i]))
            self.b.append(np.zeros((1, layers[i + 1])))

        # 初始化最后一个权重矩阵
        w = np.random.randn(layers[-2], layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
        self.b.append(np.zeros((1, layers[-1])))

    def __repr__(self):
        # 输出网络结构
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers)
        )

    def sigmoid(self, x):
        # sigmoid激活函数
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # sigmoid的导数
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display=100):
        # 训练网络
        # # 对训练数据添加一维值为1的特征，用于同时训练偏置的权重
        # X = np.c_[X, np.ones(X.shape[0])]

        # 迭代的epoch
        loss_out = []
        for epoch in np.arange(0, epochs):
            # 对数据集中每一个样本执行前向传播、反向传播、更新权重
            self.fit_partial(X, y)

            # 打印输出
            loss, y_pred = self.calculate_loss(X, y)
            acc = self.score(y, y_pred)
            loss_out.append(loss)
            if epoch == 0 or (epoch + 1) % display == 0:
                print("[INFO] epoch={}, loss={:.7f}, accuracy={:.7f}".format(
                    epoch + 1, loss, acc
                ))
        return loss_out

    def fit_partial(self, x, y):
        # 构造一个列表A，用于保存网络的每一层的输出，即经过激活函数的输出
        A = [np.atleast_2d(x)]

        # ---------- 前向传播 ----------
        # 对网络的每一层进行循环
        for layer in np.arange(0, len(self.W)):
            # 计算当前层的输出
            net = A[layer].dot(self.W[layer]) + self.b[layer]
            out = self.sigmoid(net)
            # 添加到列表A
            A.append(out)

        # ---------- 反向传播 ----------
        # 求取Gj
        gj = [(A[-1] - np.atleast_2d(y).T) * self.sigmoid_deriv(A[-1])]

        # 计算前面的权重矩阵的
        for layer in np.arange(len(A) - 2, 0, -1):
            # 参见上文推导的公式
            delta = gj[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            gj.append(delta)

        # 列表gj是从后往前记录，下面更新权重矩阵的时候，是从输入层到输出层. 因此，在这里逆序
        gj = gj[::-1]

        # 迭代更新权重
        for layer in np.arange(0, len(self.W)):
            # 参考上文公式
            self.W[layer] += -self.lr * A[layer].T.dot(gj[layer])
            self.b[layer] += -self.lr * np.atleast_2d(gj[layer].mean(axis=0)) # 取平均

    def predict(self, X):
        # 预测
        out = X
        for layer in np.arange(0, len(self.W)):
            out = self.sigmoid(np.dot(out, self.W[layer]) + self.b[layer]) # 根据训练好的权值和偏置前向计算一次输出
        predict_ = np.zeros(out.shape)                                   # sigmoid输出转换成0-1标签
        predict_[out >= 0.5] = 1

        return out, predict_

    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions, pred_label = self.predict(X)
        loss = np.sum(abs(predictions.T - targets)) / X.shape[0]

        return loss, pred_label

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict()
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc


# 加载数据集
dataset = my_dataset()
x_train, y_train, x_test, y_test = dataset.data()

# 归一化
x_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / (np.max(x_test, axis=0) - np.min(x_test, axis=0))

# 实例化标准BP
nn = NeuralNetwork([x_train.shape[1], 3, 1])
print(nn.__repr__())

loss = nn.fit(x_train, y_train)
pro, y_test_pred = nn.predict(x_test)

# 实例化累计BP
acc_nn = accumulate_NeuralNetwork([x_train.shape[1], 3, 1])
print(acc_nn.__repr__())

acc_loss = acc_nn.fit(x_train, y_train)
acc_pro, acc_y_test_pred = acc_nn.predict(x_test)

# 计算预测精度
acc = nn.score(y_test_pred, y_test)
acc_acc = acc_nn.score(acc_y_test_pred, y_test)
print("standard BP： ", acc, "\n", "accumulate BP： ", acc_acc)

# 输出混淆矩阵
maxtrix = confusion_matrix(y_test, y_test_pred)
acc_maxtrix = confusion_matrix(y_test, acc_y_test_pred)
print("standard BP： \n", maxtrix, "\n", "accumulate BP： \n", acc_maxtrix)
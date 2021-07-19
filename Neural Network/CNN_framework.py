import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data

# MNIST数据集在torvision.datasets里面，可以自行加载，其中训练集有6W张，测试集有1W，都为灰度图，即channel为1，图片的大小都是28x28

"""
pyper parameter
"""
# 每次在训练集中提取64张图像进行批量化训练，目的是提高训练速度。
# 就好比搬砖，一次搬一块砖头的效率肯定要比一次能搬64块要低得多
BATCH_SIZE = 64

# 学习率，学习率一般为0.01，0.1等等较小的数，为了在梯度下降求解时避免错过最优解
LR = 0.001

# EPOCH 假如现在我有1000张训练图像，因为每次训练是64张，
# 每当我1000张图像训练完就是一个EPOCH，训练多少个SEPOCH自己决定
EPOCH = 1

# 现在我要训练的训练集是系统自带的，需要先下载数据集，
# 当DOWNLOAD_MNIST为True是表示学要下载数据集，一但下载完，保存
# 然后这个参数就可以改为False，表示不用再次下载
DOWNLOAD_MNIST = True

"""
导入数据及预处理
"""
# root表示下载到哪个目录下
# train表示下载的是训练集，而不是测试集
# tranform格式转换为tensor
# download是否要下载

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)
"""
数据预处理
"""
# 训练集迭代器

# 测试集处理

'''
定义模型
'''
# 定义网络结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #前向传播过程
    def forward(self, x):
        output = x
        return output

# 实例化
cnn = CNN()

# 定义优化器

# 定义损失函数，交叉熵

# 训练阶段
for epoch in range(EPOCH):
    # step,代表现在第几个batch_size
    # batch_x 训练集的图像
    # batch_y 训练集的标签
    for step, (batch_x, batch_y) in enumerate(train_loader):
        # model只接受Variable的数据，因此需要转化

        # 将b_x输入到model得到返回值

        # 计算误差

        # 将梯度变为0

        # 反向传播

        # 优化参数

        # 打印操作，用测试集检验是否预测准确
        if step % 50 == 0:
            test_output = cnn(test_x)
            # squeeze将维度值为1的除去，例如[64, 1, 28, 28]，变为[64, 28, 28]
            pre_y = torch.max(test_output, 1)[1].data.squeeze()
            # 总预测对的数除总数就是对的概率
            accuracy = float((pre_y == test_y).sum()) / float(test_y.size(0))
            print("epoch:", epoch, "| train loss:%.4f" % loss.data, "|test accuracy：%.4f" %accuracy)
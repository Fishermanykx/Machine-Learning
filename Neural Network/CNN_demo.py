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
    download=False
)

# loader将训练集划分为n个64的集合，方便每次取64个，当最后不足64时，将剩下的归为一个集合里

# datase需要划分的数据集
# batch_size按多少划分
# shuffle是否要打乱数据，一般打乱对结果有一个更好的影响
# num_workers表示需要多少个核进行操作
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)
test_x = torch.unsqueeze(test_data.data, dim=1)/255.
test_y = test_data.targets

'''
定义模型
'''
# 定义网络结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 前面都是规定结构
        # 第一个卷积层，这里使用快速搭建发搭建网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      #灰度图，channel为一
                out_channels=16,    #输出channels自己设定
                kernel_size=3,      #卷积核大小
                stride=1,           #步长
                padding=1           #padding=（kernel_size-stride）/2   往下取整
            ),
            nn.ReLU(),                  #激活函数，线性转意识到非线性空间
            nn.MaxPool2d(kernel_size=2) #池化操作，降维，取其2x2窗口最大值代表此窗口，因此宽、高减半，channel不变
        )
        # 此时shape为[16, 14, 14]
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 此时shape为[32, 7, 7]
        # 定义全连接层，十分类，并且全连接接受两个参数，因此为[32*7*7, 10]
        self.prediction = nn.Linear(32*7*7, 10)
        #前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.prediction(x)
        return output

# 实例化
cnn = CNN()

# 大数据常用Adam优化器，参数需要model的参数，以及学习率
optimizer = torch.optim.Adam(cnn.parameters(), LR)
# 定义损失函数，交叉熵
loss_func = nn.CrossEntropyLoss()
# 训练阶段
for epoch in range(EPOCH):
    # step,代表现在第几个batch_size
    # batch_x 训练集的图像
    # batch_y 训练集的标签
    for step, (batch_x, batch_y) in enumerate(train_loader):
        # model只接受Variable的数据，因此需要转化
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        # 将b_x输入到model得到返回值
        output = cnn(b_x)
        # 计算误差
        loss = loss_func(output, b_y)
        # 将梯度变为0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()
        # 打印操作，用测试集检验是否预测准确
        if step%50 == 0:
            test_output = cnn(test_x)
            # squeeze将维度值为1的除去，例如[64, 1, 28, 28]，变为[64, 28, 28]
            pre_y = torch.max(test_output, 1)[1].data.squeeze()
            # 总预测对的数除总数就是对的概率
            accuracy = float((pre_y == test_y).sum()) / float(test_y.size(0))
            print("epoch:", epoch, "| train loss:%.4f" % loss.data, "|test accuracy：%.4f" %accuracy)
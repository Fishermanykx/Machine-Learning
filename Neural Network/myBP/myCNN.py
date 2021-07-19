# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:11:09 2021

@author: illusory
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torch.utils.data as Data
"""
pyper parameter
"""
BATCH_SIZE = 64
LR = 0.001
EPOCH = 1
DOWNLOAD_MNIST = False
"""
导入数据及预处理
"""
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)
"""
数据预处理
"""
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = torch.unsqueeze(test_data.data, dim=1)/255.
test_y = test_data.targets
'''
定义模型
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      
                out_channels=16,    
                kernel_size=3,      
                stride=1,           
                padding=1          
            ),
            nn.ReLU(),                  
            nn.MaxPool2d(kernel_size=2)
        )
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
        self.prediction = nn.Linear(32*7*7, 10)
        #前向传播过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.prediction(x)
        return output

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), LR)
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = cnn(test_x)
            pre_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = float((pre_y == test_y).sum()) / float(test_y.size(0))
            print("epoch:", epoch, "| train loss:%.4f" % loss.data, "|test accuracy：%.4f" %accuracy)

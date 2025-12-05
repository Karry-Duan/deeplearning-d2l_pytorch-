import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#d2l.load_data_fashion_mnist(batch_size) 加载 FashionMNIST 数据集，并返回训练集和测试集的迭代器

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

#nn.Sequential 用于定义一个顺序模型。
#nn.Flatten() 将输入的多维张量展平为二维张量。
#nn.Linear(784, 10) 定义一个全连接层，输入维度为 784（28x28 的图像展平后的维度），输出维度为 10（分类任务的类别数）。
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
#定义一个函数 init_weights，用于初始化全连接层的权重为均值为 0、标准差为 0.01 的正态分布。
net.apply(init_weights)
#使用 net.apply(init_weights) 将此初始化函数应用于网络中的所有层。
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr = 0.1)
num_epochs = 10

d2l.train_ch3(net, train_iter,test_iter,loss,num_epochs,trainer)


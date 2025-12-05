import numpy as np
import torch
from torch.utils import data #含有处理数据的模块
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w,true_b,1000)
#synthetic_data合成人工数据集

'''构造一个Pytorch数据迭代器'''
def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)  #构建数据集
    return data.DataLoader(dataset,batch_size,shuffle=is_train) #将数据集类型的东西分为batch
#data.TensorDataset 是 PyTorch 的一个类，用于将多个张量包装成一个数据集
#*data_arrays：解包后的输入特征和标签张量。data_arrays是元组，由features与labels构成
#shuffle：洗牌，打乱

batch_size = 10
data_iter = load_array((features,labels),batch_size)
#iter：Iterator：迭代器

next(iter(data_iter))
#iter(data_iter):创建一个迭代器，该迭代器可以逐个生成 data_iter 中的批次数据。
#data_iter 是一个 DataLoader 对象，它本身就是一个可迭代的对象，但调用 iter 函数可以显式地创建一个迭代器。
#next(iter(data_iter)):
#调用 next 函数，获取迭代器的下一个元素。在这个上下文中，next(iter(data_iter)) 获取 data_iter 的第一个批次的数据。

net = nn.Sequential(nn.Linear(2,1))
#Sequential:按层数排起来的所有，nn.linear这一层是线性回归（因为用的只有一层，所以只有一个

net[0].weight.data.normal_(0,0.01) #要迭代的w（
net[0].bias.data.fill_(0)  #要迭代的b

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)
#net.paraments():是一个方法，返回模型中所有需要优化的参数。这些参数通常包括权重（weights）和偏置（biases）。在代码中，net 是一个 nn.Sequential 模型，包含一个 nn.Linear 层。net.parameters() 会返回这个线性层的权重和偏置。

num_epochs = 3
for epoch in range(num_epochs):
    for X , y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch{epoch + 1},loss {l: f}')

'''数据读取，模型定义，参数初始化，损失函数，训练模块'''
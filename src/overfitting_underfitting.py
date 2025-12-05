import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
max_degree = 20 #特征20
n_train,n_test = 100,100 #样本数量
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])

features = np.random.normal(size=(n_train + n_test , 1))
np.random.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
#计算多项式特征poly_features。features的每个元素被提升到0到max_degree-1的幂次，
# 形成一个多项式特征矩阵。np.arange(max_degree).reshape(1, -1)生成了一个形状为(1, max_degree)的数组，表示幂次。

for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i + 1)

#对多项式特征进行归一化处理，以避免高次项的数值过大。具体来说，代码的作用是将每个特征值除以其对应的阶乘值。
#math.gamma是伽马函数，对于正整数n，math.gamma(n)等于(n-1)!，即n-1的阶乘。

labels = np.dot(poly_features,true_w)
#矩阵乘法np.do
#计算标签labels。通过矩阵乘法np.dot，将多项式特征矩阵poly_features与真实权重true_w相乘，得到每个样本的标签值
labels += np.random.normal(scale = 0.1,size = labels.shape)
#在标签值上添加噪声。

def evaluate_loss(net, data_iter, loss):
    '''评估给定数据集上模型的损失'''
    metric = d2l.Accumulator(2) #d2l.Accumulator(2)：创建一个累加器
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum(),l.numel())#l.numel() 张量（Tensor）中所有元素的总数
    return metric[0] /metric[1]

def train(train_features, test_features, train_labels, test_labels,num_epochs=400):
    loss = nn.MSELoss()#损失函数，用于计算均方误差（Mean Squared Error, MSE）
    input_shape = train_features.shape[-1]#获取张量形状的最后一个维度的大小。在机器学习中，这通常表示每个样本的特征数量。
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))#nn.Sequential：这是一个容器，用于按顺序堆叠多个神经网络层。它会自动将输入数据依次通过这些层，最终得到输出。
    #input_shape：输入特征的维度，即每个样本的特征数量。1：输出特征的维度，这里设置为 1，表示模型的输出是一个标量值（例如，回归任务中的预测值）。
    #bias=False：表示不使用偏置项 b。在某些情况下，为了简化模型，可以不使用偏置项。
    batch_size = min(10,train_labels.shape[0])
   
    '''
    train_iter = d2l.load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features,test_labels.reshape(-1,1)),batch_size, is_train=False)
    '''
     #d2l.load_array 是一个函数，用于将特征和标签数组转换为 PyTorch 的 DataLoader 对象。DataLoader 是 PyTorch 中用于加载和迭代数据的工具，它支持批量加载、数据打乱、多线程加载等功能。
    #data_arrays：一个元组，包含特征数组和标签数组。例如，(train_features, train_labels)。
    #batch_size：每个批次的样本数量。
    #is_train：一个布尔值，表示是否为训练数据。如果为 True，则在加载数据时会打乱数据；如果为 False，则不会打乱数据。
    # 关键修正：显式转成 Tensor（float32）
    train_X = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels.reshape(-1, 1), dtype=torch.float32)
    test_X  = torch.tensor(test_features, dtype=torch.float32)
    test_y  = torch.tensor(test_labels.reshape(-1, 1), dtype=torch.float32)

    # 用 d2l.load_array（此时已是 Tensor，就不会在里面再触发 numpy 的 .size）
    train_iter = d2l.load_array((train_X, train_y), batch_size)
    test_iter  = d2l.load_array((test_X,  test_y),  batch_size, is_train=False)
    
    trainer = torch.optim.SGD(net.parameters(),lr=0.01)
    anamator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[1,num_epochs],ylim=[1e-3,1e2],legend=['train','test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net,train_iter,loss,trainer)
        if epoch ==0 or (epoch + 1) %20 == 0:
            anamator.add(epoch + 1, (evaluate_loss(net,train_iter,loss),evaluate_loss(net,test_iter,loss)))
            plt.pause(0.01) # 暂停以刷新图像，参数为刷新间隔时间（秒）           

    print('weight:',net[0].weight.data.numpy())
'''三阶多项式拟合'''
train(poly_features[:n_train, :4],poly_features[n_train:, :4],labels[:n_train],labels[n_train:])
plt.show()
'''一次线性欠拟合'''
train(poly_features[:n_train, :2],poly_features[n_train:, :2],labels[:n_train],labels[n_train:])
plt.show()
'''高阶多项式过拟合'''
train(poly_features[:n_train,:],poly_features[n_train:, :],labels[:n_train],labels[n_train:],num_epochs=1500)
plt.show()
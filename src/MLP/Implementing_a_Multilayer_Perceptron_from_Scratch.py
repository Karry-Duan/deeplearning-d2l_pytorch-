import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
#初始化参数
w1 = nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True) * 0.01) #没乘0.01，也是无语了...
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))

w2 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)

params = [w1, b1, w2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1,num_inputs))
    H = relu(X @ w1 +b1)
    return (H @ w2 +b2)

loss = nn.CrossEntropyLoss(reduction= 'none')
#reduction='none'行为：不进行任何归约操作，返回每个样本的损失值。
#reduction='mean'行为：对所有样本的损失值取平均。
#reduction='sum'行为：对所有样本的损失值求和。
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params,lr = lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,updater)


d2l.plt.show()
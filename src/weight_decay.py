import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

n_train, n_test, num_imputs, batch_size = 20, 100, 200, 5 
true_w, true_b = torch.ones((num_imputs,1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w,true_b,n_train)
train_iter = d2l.load_array(train_data,batch_size)
test_data = d2l.synthetic_data(true_w,true_b,n_test)
test_iter = d2l.load_array(test_data,batch_size,is_train=False)

def init_params():
    '''初始化模型参数'''
    w = torch.normal(0, 1, size=(num_imputs, 1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w, b]

def l1_penalty(w):
    '''定义L1（曼哈顿）范数乘法'''
    return torch.sum(torch.abs(w))

def l2_penalty(w):
    '''定义L2的范数乘法'''
    return torch.sum(w.pow(2)) / 2

def train(lambd):  #ambd：L2 正则化参数，用于控制正则化的强度
    '''定义训练函数'''
    w, b = init_params()
    net, loss = lambda X:d2l.linreg(X, w, b),d2l.squared_loss #这部分代码定义了一个线性模型，使用了 lambda 表达式。lambda 表达式lambda 表达式是一种简洁的匿名函数定义方式。在这个例子中，lambda X: d2l.linerg(X, w, b) 定义了一个函数，它接受一个输入 X，并返回 d2l.linerg(X, w, b) 的结果。
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs',ylabel='loss',yscale='log',xlim=[5,num_epochs],legend=['train','test'])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            #with torch.enable_grad():
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w,b],lr,batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(d2l.evaluate_loss(net,train_iter,loss),
                                    d2l.evaluate_loss(net,test_iter,loss)))
            plt.pause(0.01) # 暂停以刷新图像，参数为刷新间隔时间（秒）
    print('w的L2范数是：', torch.norm(w).item())

train(lambd=3)
plt.show()


'''
train(lambd=0)
plt.show()#测试集上的loss没有减小，且训练集与测试集的gap很大，典型的过拟合

train(lambd=6)
plt.show()#后续的gap没有继续扩大，还减小了，说明过拟合由于罚的加入变小了
'''
def train_concise(wd):
    '''权重衰减的简洁实现'''
    net = nn.Sequential(nn.Linear(num_imputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([{"params":net[0].weight,'weight_decay':wd},{"params":net[0].bias}],lr=lr)
    animator = d2l.Animator(xlabel='epochs',ylabel='loss',yscale='log',xlim=[5,num_epochs],legend=['train','test'])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch+1,(d2l.evaluate_loss(net, train_iter,loss),d2l.evaluate_loss(net,test_iter,loss)))
            plt.pause(0.01)
    print('w的L2范数：',net[0].weight.norm().item())
'''
train_concise(0)
plt.show()

train_concise(3)
plt.show()

train_concise(6)
plt.show()
'''

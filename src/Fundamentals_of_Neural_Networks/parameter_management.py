import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size = (2, 4))
#print(net(X))

'''参数访问'''
#print(net[2].state_dict())
#print(type(net[2].bias))
#print(net[2].bias,'***')
#print(net[2].weight.shape)
#print(net[2].bias.data)

#print(net[2].weight.grad == None)

'''一次性访问所有参数'''
#print(*[(name, param.shape) for name, param in net[0].named_parameters()])
#print('******************************************')
#print(*[(name, param.shape) for name, param in net.named_parameters()])

#print(net.state_dict()['2.bias'].data)

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net
rgnet = nn.Sequential(block2(),nn.Linear(4, 1))
#print(rgnet(X))

#print(rgnet)

'''内置初始化'''
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)
#print(net[0].weight.data,'**********',net[0].bias.data,sep='\n')

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
#print(net[0].weight.data,'**********',net[0].bias.data,sep='\n')
'''对不同层进行不同的初始化'''
def xvier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        #Xavier 初始化方法的核心思想是保持输入和输出的方差相同。具体来说，对于一个神经网络层，如果输入的方差为 σ 2，那么输出的方差也应该为 σ 2。这样可以避免梯度消失或梯度爆炸的问题。
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(xvier)
net[2].apply(init_42)
print(net[0].weight.data)
print('***************************')
print(net[2].weight.data)
print(net)

'''自定义初始化'''
def my_init(m):
    if type(m) == nn.Linear:
        print('Init',*[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= (m.weight.data.abs() >= 5)

net.apply(my_init)
print(net[0].weight[:2])

net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

'''参数绑定'''
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[4].weight.data[0, 0])
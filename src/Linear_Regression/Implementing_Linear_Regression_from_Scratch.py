import matplotlib.pyplot as plt
import random
import torch
import sys
#import os
sys.path.append('C:\\Users\\Lenovo\\anaconda3\\envs\\d2l-zh\\lib\\site-packages\\d2l')
from d2l import torch as d2l

'''建立人造数据集'''
def synthetic_data(w,b,num_examples):# synthetic 人造的、合成的
    #生成y = Xw + b + 噪声
    
    X = torch.normal(0,1,(num_examples,len(w)))
    #(num_examples, len(w)) 是矩阵的形状，表示生成 num_examples 
    # 个样本，每个样本有 len(w) 个特征。
    
    y = torch.matmul(X,w) + b #目标值y
    #torch.matmul 函数计算矩阵 X 和向量 w 的矩阵乘法
    
    y +=torch.normal(0,0.01,y.shape)
    #为了模拟真实数据中的噪声，向目标值 y 添加了均值为 0、
    # 标准差为 0.01 的正态分布随机噪声。
    return X,y.reshape((-1,1))#列向量返回，y是一个列向量

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000)


'''看一下模拟的数据集'''
print('features:',features[0],'\nlable:',labels[0])

d2l.set_figsize()
plt.scatter(features[:,1].detach().numpy(),
            labels.detach().numpy(),1)
plt.show()


'''定义一个data_iter 函数， 该函数接收批量大小、特征矩阵和标签向
量作为输,生成大小为batch_size的小批量'''
def data_iter(batch_size,features,labels):
    num_examples = len(features) #获取特征矩阵 features 中样本的数量,len获取的是最外层的数量，所以前面要转化为列矩阵
    indices = list(range(num_examples))
    #这些样本随机读取，没有特定的顺序
    random.shuffle(indices)
    #random.shuffle 函数随机打乱这个索引列表，以实现随机读取样本。
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i + 
                                                   batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]
    
batch_size = 10
#看一下其中的一个batch
for X,y in data_iter(batch_size,features,labels):
    print(X,y,sep='\n')
    break

'''定义初始化模型参数'''
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

'''定义模型'''
def linreg(X, w, b):
    return torch.matmul(X,w) + b #得到y_hat,torch.matmul()执行矩阵乘法的

'''定义损失函数,采用均方损失'''
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2 #reshape增强健壮性

'''定义优化算法,小批量随机下降'''
'''params:参数,是一个list,包含w,b,某一点(特定w与b)下y_hat的梯度?每尝试一次param都相当于在地图上走了一步
   lr: 学习率'''
def sgd(params,lr,batch_size): #Stochastic Gradient Descent 随机梯度下降
    with torch.no_grad(): #因为下面在计算了，之前又没有重置，不然梯度会乱
        for param in params:
            param -= lr * param.grad /batch_size
            param.grad.zero_() #重置梯度

lr = 0.5
num_epochs = 3 #数据扫三遍
net = linreg #如果后续可以的话可以快速切换模型
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size,features,labels): #划分batch
        l = loss(net(X,w,b),y) #X,y的小批量损失
        #l 的形状是(batch_size，1)而不是一个标量，l中的所有元素被加到一起
        #并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w,b],lr,batch_size) #使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch{epoch +1},loss{float(train_l.mean()):f}')

print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b - b}')

#终于把十五分钟的视频看完了，救赎感！！！！
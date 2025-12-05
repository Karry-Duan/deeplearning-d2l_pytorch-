'''从零开始实现softmax回归'''
'''softmax是线性结果+softmax的过程'''
import torch
from IPython import display
import matplotlib.pyplot as plt
from d2l import torch as d2l
import matplotlib

matplotlib.use('TkAgg')
# 设置matplotlib后端（Windows系统推荐）
plt.ion()  # 打开交互模式

batch_size = 256
train_iter , test_iter = d2l.load_data_fashion_mnist(batch_size)


num_inputs = 784 #28*28 softmax输入需要是向量
num_outputs = 10  #输出十个类，即要分的类

w = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
# 权重 X（batchsize，784） x w（784 ， 10） 最后为（batchsize，10）行数为每一个batchsize的输出结果，列数代表每一个类别
b = torch.zeros(num_outputs,requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True) #每一个batchsize里的所有总值
    return X_exp / partition 

def net(X):
    return softmax(torch.matmul(X.reshape(-1,w.shape[0]),w) + b)
#matmul 实现矩阵乘法
'''
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
'''

def cross_entropy(y_hat,y):
    '''损失函数'''
    return -torch.log(y_hat[range(len(y_hat)),y])

#print(cross_entropy(y_hat,y))

def accuracy(y_hat,y):
    '''计算预测正确的数量'''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) #每一行元素中最大值的下标存储到
    cmp = y_hat.type(y.dtype) == y #把y的数据类别转化为和y一样的作比较,CMP是一个张量
    return float(cmp.type(y.dtype).sum())

#print(accuracy(y_hat,y) / len(y))

def evaluate_accuracy(net,data_iter):
    '''计算在指定数据集上的模型的精度'''
    if isinstance(net,torch.nn.Module): #这一行用于检查 net 是否是一个 torch.nn.Module 的实例
        net.eval() #转为评估模式，只计算前向传播
    metric = d2l.Accumulator(2) #这里初始化了一个累加器，用于累加两个指标。d2l.Accumulator(2) 表示这个累加器可以存储两个累加值，分别为正确预测数，预测总数
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0] / metric[1] #分类正确样本数/总样本数

'''
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()  
    print(evaluate_accuracy(net, test_iter))
'''

def train_epoch_ch3(net, train_iter, loss, updater): #一个 epoch 表示在整个训练数据集上完成一次前向和后向传播的过程。
    if isinstance(net, torch.nn.Module): #torch.nn.Module 是 PyTorch 中所有神经网络模块的基类。它提供了一个框架，用于构建和管理神经网络模型。通过继承 torch.nn.Module，可以定义自己的神经网络结构，并利用 PyTorch 提供的自动求导、参数管理等功能。
        net.train() #net.train() 是一个方法，用于将模型设置为训练模式。这与 net.eval() 相对，后者用于将模型设置为评估模式。
    metric = d2l.Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step() #updater.step() 会根据计算得到的梯度来更新模型的参数。
            metric.add(
            float(l) * len(y), accuracy(y_hat,y),y.size().numel() #y的维度数
            )#训练损失  精确度 y的维度（几个类别
        else:
            l.sum().backward()
            updater(X.shape[0]) #X.shape[0] 是当前批次的样本数（批量大小）。
            metric.add(float(l.sum().detach()), accuracy(y_hat,y), y.numel())
    return metric[0] / metric[2] , metric[1] / metric[2]
        #metric[0]：累加的总损失值。metric[1]：累加的正确预测样本数。metric[2]：累加的总样本数

class Animator:  
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda: d2l.set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:  # 改为 self.X 或 self.Y 为 None 的情况
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        '''
         for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        '''
        for x_val, y_val, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_val, y_val, fmt)
        self.config_axes()

        plt.pause(0.01)

    def display(self):
        """在绘制完成后显示图像"""
        #plt.show()
        plt.ioff()  # 关闭交互模式
        plt.show()  # 让窗口暂停，等待用户关闭
       


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    animator.display()
    #train_loss, train_acc = train_metrics

lr = 0.1

def updater(batch_size):
    return d2l.sgd([w,b],lr, batch_size)

num_epochs = 10

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    plt.ioff()

def predic_ch3(net,test_iter, n = 6):
    for X,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis = 1))
    titles = [true +'\n' +pred for true ,pred in zip(trues,preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)),1,n,titles= titles[0:n])

predic_ch3(net,test_iter)    
plt.show()
'''导数'''
import torch

x = torch.arange((4.0))
print(x)
x.requires_grad_(True)#执行后，可以在x.grad中存放梯度
x.grad

y = 2 * torch.dot(x,x)

y.backward()#触发反向传播，计算y关于x的梯度
print(x.grad,x.grad == 4*x,sep='\n')

x.grad.zero_()#在默认情况下，pytorch会劣迹梯度，我们需要清楚之前的值
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
y.sum().backward()#很少对向量求导（y为向量），y.sum()为标量
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach() #把y不看成关于x的变量
z = u * x #u可认为是常数，那么对zbackward时，相当于常数+x对x求导，即为常数

z.sum().backward()
print(x.grad == u) #所以为true

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(),requires_grad=True)
d = f(a)
print(d.backward())
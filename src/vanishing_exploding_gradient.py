import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(),[y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
plt.show()
# 将张量（Tensor）从计算图中分离出来，使其不再参与梯度计算。
# numpy() 是一个方法，用于将框架中的张量（Tensor）转换为 NumPy 数组。
M = torch.normal(0, 1 , size=(4,4))
print('一个矩阵\n', M)
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4,4)))
    #torch.mm(input, mat2, out=None) → Tensor
    #input：第一个矩阵，形状为 (n, m)。 
    #mat2：第二个矩阵，形状为 (m, p)。
    #out：（可选）输出张量，用于存储结果。
print('乘以100个矩阵后\n', M)

'''线性代数'''
import torch
'''
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x+y,x*y,x/y,x**y,sep='\n')

x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

A = torch.arange(20).reshape(5,4)
print(A)
print(A.T)

B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B,B.T,B==B.T,sep='\n')
'''
'''
X = torch.arange(24).reshape(2,3,4)
print(X)

A = torch.arange(20,dtype=torch.float32).reshape(5,4)
B = A.clone()
print(A,A+B,A+B==A+A,sep='\n')

print(A*B)

a = 2
X = torch.arange(24).reshape(2,3,4)
print(a+X,a*X,(a*X).shape,sep='\n')

x = torch.arange(4,dtype=torch.float32)
print(x,x.sum())

print(A.shape,A.sum())
'''
'''
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
A_sum_axis0 = A.sum(axis=0)#沿列的方向求和
A_sum_axis1 = A.sum(axis=1)#沿行的方向求和
A.sum(axis=[0,1])
#print(A_sum_axis0,A_sum_axis1,A.sum(axis=[0,1]),sep='\n')
#对哪一个求和就是消掉对应的层级

#print(A.mean(dtype=torch.float32),A.sum()/A.numel(),sep='\n')#注意括号的添加sum（），numel（）
#print(A,A.shape,A.mean(axis=0,dtype=torch.float32),A.sum(axis=0)/A.shape[0],sep='\n')
sum_A = A.sum(axis=1,keepdims=True)
print(sum_A)

print(A/sum_A)

print(A.cumsum(axis=0))
'''
'''
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
x = torch.arange(4,dtype=torch.float32)
y = torch.ones(4,dtype=torch.float32)
#print(x,y,torch.dot(x,y),sep='\n')#torch.dot是点积，是相同位置的按元素乘积的和
torch.sum(x * y)
#print(x,y,torch.dot(x,y),torch.sum(x * y)==torch.dot(x,y),sep='\n')
print(A.shape,x.shape,torch.mv(A,x),sep='\n')

B = torch.ones(4,3)
print(torch.mm(A,B))
'''
u = torch.tensor([3.0,-4.0])
print(torch.norm(u),torch.abs(u).sum(),torch.norm(torch.ones((4,9))),sep='\n')

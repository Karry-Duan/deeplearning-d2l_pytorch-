'''数据操作'''
import torch 
'''
x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
print(torch.zeros((2,3,4)))
print(torch.ones((2,3,4)))
print(torch.randn(3,4))

print(torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]]))'''
'''
x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
print(x+x,x-y,x*y,x/y,x**y,sep='\n')

print(torch.exp(x))
'''
X = torch.arange(12,dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((X,Y),dim=0),torch.cat((X,Y),dim=1),sep='\n')

print(X == Y)#判断是否等于

print(X.sum())

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
print(a,b,sep='\n')
print(a + b)

print("*********",X,sep='\n')
print(X[-1],X[1:3],sep='\n')
X[1,2]=9
print(X)

X[0:2,:] = 12
print(X)

before = id(Y)
print(before,id(Y),sep='\n')
Y  = Y + X
id(Y) == before
print(id(Y) == before,before,id(Y),sep='\n')

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))

Z[:] = X + Y
print(X,Y,Z,sep='\n')
print('id(Z):', id(Z))
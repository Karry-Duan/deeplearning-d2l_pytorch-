'''数据预处理'''

import os #许多与操作系统交互的功能，比如文件和目录的操作。
import pandas as pd
import torch

os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')#存储了house_tiny的文件路径
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Prince\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))#新版本的panda有修改，与课件不同
print(inputs)

inputs = pd.get_dummies(inputs,dummy_na = True,dtype=float)#新版本会输出false与true，需要加dtype，与课件不同
print(inputs)

X,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(X,y)


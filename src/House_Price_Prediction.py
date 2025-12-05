import hashlib
import os
import tarfile
import zipfile
import requests

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

'''25.9.28记：第一次把代码基本上全懂了，感觉自己是个天才，没错，再一次的（虽然梯度爆炸梯度消失哪里仍然是狗屎...)'''
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..','data')):
    '''下载一个DATA_HUB中的文件,返回本地文件名'''
    assert name in DATA_HUB, f'{name} 不存在于 {DATA_HUB}'
    url, shal_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        shal = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                shal.update(data)
        if shal.hexdigest() == shal_hash:
            '''检查数据是否已经存在，如果存在，就不重复下载'''
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url,stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    '''下载并解压zip/tar文件'''
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    '''下载DATA_HUB中的所有文件'''
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

#print(train_data.shape)
#print(test_data.shape)

#print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
#去掉训练集中的id与labels，去掉验证集中的id，并将两个处理后的集合按行拼接在一起
#print(f'1 :{all_features.shape}')
numeric_feateres = all_features.dtypes[all_features.dtypes != 'object'].index#这会生成一个布尔型 Series，表示每列的数据类型是否不等于 'object'。在 Pandas 中，'object' 类型通常表示字符串或混合类型的数据。
all_features[numeric_feateres] = all_features[numeric_feateres].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_feateres] = all_features[numeric_feateres].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)#dummies 独热编码

#print(f'2: {all_features.shape}')

all_features = all_features.astype(np.float32)

'''将数据分离出来'''
n_train = train_data.shape[0]#有多少给数据
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
#.values 是 Pandas DataFrame 的一个属性，它将 DataFrame 转换为一个 NumPy 数组。这样可以方便地将数据转换为 PyTorch 张量
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)

'''构建训练'''
loss = nn.MSELoss()#计算均方误差（Mean Squared Error, MSE）
in_features = train_features.shape[1]#特征向量长度

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels):
    '''相对误差'''
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()#.item() 方法用于将一个包含单个元素的张量（Tensor）转换为 Python 的标量（scalar）

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    '''k折交叉验证划分训练集与验证集'''
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :],y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    '''计算损失'''
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epochs', ylabel='rmse', xlim=[1, num_epochs],legend=['train','valid'], yscale='log')
        print(f'fold{i + 1}, trian log rmse {float(train_ls[-1]):f},'
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs,lr, weight_decay, batch_size)
d2l.plt.show()
print((f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}'))
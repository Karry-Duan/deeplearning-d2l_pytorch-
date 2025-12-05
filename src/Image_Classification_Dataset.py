'''图像分类数据集的导入'''

import torch
import torchvision #关于计算机视觉实现
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\Lenovo\\anaconda3\\envs\\d2l-zh\\lib\\site-packages\\d2l')

#d2l.use_svg_display() #用svg显示图片

trans = transforms.ToTensor()#图片转换为tensor
mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,#下载训练数据集
                                                transform=trans, #表示拿出来后得到tensor而不是图片
                                                download=True) #默认从网上下载
mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,
                                              transform=trans,download=True)
#print(len(mnist_train),len(mnist_test))
#看一下图片数量（len:最外层维度
#print(mnist_train[0][0].shape) 第一个是一个包含标签与图片的元组，所以第二个0才代表了第一个图片
#1表示为黑白，只有一个通道
'''两个可书画数据集的函数'''

'''返回Fashion—MNIST数据集的文本标签'''
def get_fashion_mnist_labels(labels):
    text_labels = [
         't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

'''
imgs:要显示的图像列表，每个元素可以是一个 torch.Tensor 或其他可以直接用于 imshow 的图像格式。
num_rows:显示图像的行数。
num_cols:显示图像的列数。
titles:可选参数，为每张图像添加的标题列表。如果提供，其长度应与 imgs 相同。
scale:可选参数,用于调整整个图像网格的大小,默认值为1.5。
'''
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy().squeeze())  # 使用 squeeze() 调整形状
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
'''
X,y = next(iter(data.DataLoader(mnist_train,batch_size = 18)))
#X是图片的张量，y是对应标签
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
d2l.plt.show()
'''

if __name__ == '__main__':
    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # X是图片的张量，y是对应标签
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    d2l.plt.show()

#batch_size = 256

def get_dataloader_workers():
    '''使用四个进程读取数据'''
    return 0
'''
train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,
                             num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X,y in train_iter:
    continue

print(f'{timer.stop():.2f} sec')
'''
def main():
    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')
    
if __name__ == '__main__': 
    main()

'''
def load_data_fashion_mnist(batch_size, resize=None):  
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break'''
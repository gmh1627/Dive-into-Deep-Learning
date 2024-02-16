import torch
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def get_fashion_mnist_labels(labels):#@save
    """返回数据集的文本标签"""
    text_labels=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):#@save
    """绘制图像列表"""
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)# The _ is a convention in Python to indicate a variable that is not going to be used. In this case, it is used to capture the first return value of subplots, which is the entire figure.
    axes=axes.flatten()
    #enumerate意为"枚举"
    for i ,(ax,img) in enumerate(zip(axes,imgs)):#The enumerate() function is used to get both the index i and the paired values.
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

batch_size=256

def get_dataloader_workers():#@save
    """使用4个子进程来读取数据"""
    """每个子进程会预加载一批数据，并将数据放入一个共享内存区域。当主进程需要数据时，它可以直接从共享内存区域中获取，而不需要等待数据的读取和预处理。这样，主进程可以在处理当前批次的数据时，子进程已经在后台加载下一批数据，从而提高数据加载的效率。"""
    return 4

#定义一个计时器
class Timer:#@save
    def __init__(self) :
        self.times=[]
        self.start()
        
    def start(self):
        """启动计时器"""
        self.tik=time.time()
        
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time()-self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times)/len(self.times)
    
    def sum(self):
        """返回总时间"""
        return sum(self.times)
    
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
    
def load_fashion_mnist(batch_size,resize=None):#@save
    """下载Fashion-MNIST数据集到内存中"""
    trans = [transforms.ToTensor()]#将PIL图像转为tensor格式(32位浮点数)，并除以255使得所有像素的数值均为0-1
    #the "transforms" module in PyTorch's torchvision library is used to define a sequence of image transformations
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    #"Compose" transformation allows you to apply a sequence of transformations to an input. The resulting trans object can be applied to images or datasets.
    mnist_train=torchvision.datasets.FashionMNIST(root="data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="data",train=False,transform=trans,download=True)
    #"train=True"表示加载训练集，"train=False"表示加载测试集。
    #print(len(mnist_train),len(mnist_test))
    #每个输入的图像高度和宽度均为28像素，并且是灰度图像，通道数为1，下文将高度为h像素，宽为w像素的图像的的图像的形状记为(h,w)
    #print(mnist_train[0][0].shape)#mnist_train[0][0]指的是第一个图像数据的张量，mnist_train[0][1]指的是第一个图像的标签
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,batch_size,shuffle=False,
                            num_workers=get_dataloader_workers()))
    # The DataLoader is responsible for loading batches of data, shuffling the data (for training), and using multiple workers for data loading ("num_workers" parameter).
    # "shuffle=False" for mnist_test ensures that the test data remains in its original order during evaluation.

#X,y=next(iter(data.DataLoader(mnist_train,batch_size=18)))
#show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
#plt.show()

if __name__ == '__main__':
    train_iter, test_iter = load_fashion_mnist(32, resize=64)
    #timer = Timer()
    #for X, y in train_iter:
    #    continue
    #print(f'{timer.stop():.2f} sec')
    for X,y in train_iter:
        print(X.shape,X.dtype,y.shape,y.dtype)#dtype即data type
        break
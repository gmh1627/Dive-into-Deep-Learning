import torch
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display

#初始化参数
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
num_inputs=784#图像有28*28像素，本节将其看作长度为784的向量
num_outputs=10#softmax回归中输出与类别一样多，数据集有10个类别
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

#定义softmax操作
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition#结果每行和为1

#定义softmax回归模型
#在将数据传递到模型之前，使用reshape将每个原始图像展开为向量
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

y = torch.tensor([0, 2])#有了y，我们知道在第一个样本中第一类是正确的预测；在第二个样本中第三类是正确的预测
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])#2个样本在3个类别上的预测概率
#print(y_hat[[0, 1], y])#然后使用y作为y_hat中概率的索引，我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率，即输出[y[0],y[1]]

#定义交叉熵损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

#定义一个用于对多个变量累加的的类
class Accumulator:#@save
    def __init__(self,n):
        self.data=[0.0]*n
    
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data=[0.0]*len(self.data)
        
    def __getitem__(self,idx):
        return self.data[idx]
    
#计算分类精度
def accuracy(y_hat,y):#@save
    """计算预测正确的数量"""
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:#如果`y_hat`是矩阵，那么假定第二个维度存储每个类的预测分数
        y_hat=y_hat.argmax(axis=1)#使用`argmax`获得每行中最大元素的索引来获得预测类别
    cmp=y_hat.type(y.dtype)==y#由于等式运算符“`==`”对数据类型很敏感，因此我们将`y_hat`的数据类型转换为与`y`的数据类型一致。
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):#@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net,torch.nn.Module):
        net.eval()#将模型设为评估模式
    metric=Accumulator(2)#Initializes an Accumulator with two variables: the number of correct predictions and the total number of predictions.
    with torch.no_grad():#disables gradient computation
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())#y.numel() returns the total number of elements in y.
    return metric[0]/metric[1]

#if __name__ == '__main__':
    #print(evaluate_accuracy(net,test_iter))#由于使用随机权重初始化net模型,因此该模型的精度接近于随机猜测,如在有10个类别情况下的精度接近0.1 

#定义一个在动画中绘制数据的类
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:#lengend:图例
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        #"plt.subplots()" is called to create a figure (self.fig) and one or more subplots (self.axes).
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        #A lambda function is used to create an anonymous function that is then assigned to the "self.config_axes" attribute. 
        #This is a common pattern in Python, especially when a short, simple function is needed, and there's no intention to reuse it elsewhere in the code.
        # It provides a more compact and inline way to express the behavior.
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):#Adds data points to the plot.
        if not hasattr(y, "__len__"):#If y is not iterable (doesn't have a length), it is converted to a list to ensure it can be processed as a collection of values.
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):#If x is not iterable, it is repeated n times to match the length of y.
            x = [x] * n
        if not self.X:#If "self.X" is not initialized, it is initialized as a list of empty lists, with one list for each element in y.
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()# clears the current axis to prepare for the new data
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()#configures the axis using specified parameters.
        display.display(self.fig)
        display.clear_output(wait=True)
        
#训练
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    #updater是更新模型参数的常用函数，在后文定义
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):#checks if the object referred to by the variable net is an instance of the "torch.nn.Module" class
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            #使用PyTorch内置的优化器和损失函数
            updater.zero_grad()# Clear previously calculated gradients
            l.mean().backward()
            updater.step()
        else:
            #使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    # 创建一个用于动画绘制的实例
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        # 训练模型一个迭代周期，并获取训练损失和准确度
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 在测试集上评估模型精度
        test_acc = evaluate_accuracy(net, test_iter)
        # 将训练损失、训练准确度和测试准确度添加到动画中
        animator.add(epoch + 1, train_metrics + (test_acc,))

    # 获取最后一个周期的训练损失和训练准确度
    train_loss, train_acc = train_metrics
    # 检查训练损失、训练准确度和测试准确度的合理性
    assert train_loss < 0.5, train_loss#If the condition is False, it raises an AssertionError exception.
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def updater(batch_size):#@save
    return d2l.sgd([W, b], lr, batch_size)

if __name__ == '__main__':
    lr = 0.1
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

#预测
#给定一系列图像，我们将比较它们的实际标签（文本输出的第一行）和模型预测（文本输出的第二行）
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    # Iterate over the test dataset to get a batch of images and their true labels
    for X, y in test_iter:
        break
    
    # Get the true labels in text format
    trues = d2l.get_fashion_mnist_labels(y)
    # Use the trained model to make predictions on the test batch and convert predictions to text labels
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1)) 
    # Create titles for the images by combining true labels and predicted labels
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    # Display a subset (n) of the images along with their true and predicted labels
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

    
if __name__ == '__main__':
    predict_ch3(net, test_iter)
    
plt.show()#将折线图和预测结果的图像统一显示
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    #generates a random matrix X with dimensions (num_examples, len(w)) using a normal distribution with a mean of 0 and standard deviation of 1.
    X = torch.normal(0, 1, (num_examples, len(w)),dtype=torch.float32) 
    #calculates the target values y by multiplying the input matrix X with the weight vector w and adding the bias term b. 
    y = torch.matmul(X, w) + b  
    #And then adds some random noise to the target values y. The noise is generated from a normal distribution with mean 0 and standard deviation 0.01.                    
    y += torch.normal(0, 0.01, y.shape)             
    return X, y.reshape((-1, 1)) #The -1 in the first dimension means that PyTorch should automatically infer the size of that dimension based on the total number of elements. In other words, it is used to ensure that the reshaped tensor has the same total number of elements as the original tensor.

true_w=torch.tensor([2,-3.4],dtype=torch.float32)
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)
print('features:',features[0],'\nlabel:',labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1)
#plt.show()#显示散点图
#"features[:, 1]" selects the second column of the features tensor. 
#The detach() method is used to create a new tensor that shares no memory with the original tensor, and numpy() is then called to convert it to a NumPy array.
#"1" is the size of the markers in the scatter plot.

def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    #随机读取文本
    random.shuffle(indices)#"Shuffle the indices"意为打乱索引 
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])#"min(i + batch_size, num_examples)" is used to handle the last batch, which might have fewer examples than batch_size.
        yield features[batch_indices],labels[batch_indices]

#初始化参数。从均值为0，标准差为0.01的正态分布中抽取随机数来初始化权重，并将偏置量置为0       
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

#定义线性回归模型
def linreg(X,w,b): #@save
    return torch.matmul(X,w)+b #广播机制：用一个向量加一个标量时，标量会加到向量的每一个分量上

#定义均方损失函数
def squared_loss(y_hat,y): #@save
    return (y_hat-y.reshape(y_hat.shape))**2/2

#定义优化算法：小批量随机梯度下降
def sgd(params,lr,batch_size): #@save
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

#轮数num_epochs和学习率lr都是超参数，先分别设为3和0.03，具体方法后续讲解
lr=0.03
num_epochs=3
batch_size=10

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=squared_loss(linreg(X,w,b),y)
        l.sum().backward()#因为l是一个向量而不是标量，因此需要把l的所有元素加到一起来计算关于(w,b)的梯度
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l=squared_loss(linreg(features,w,b),labels)
        print(f'epoch {epoch+1}:squared_loss {float(train_l.mean()):f}')
print(f'w的估计误差:{true_w-w.reshape(true_w.shape)}')#结果中的grad_fn=<SubBackward0>表示这个tensor是由一个正向减法操作生成的
print(f'b的估计误差:{true_b-b}')#<RsubBackward1>表示由一个反向减法操作生成
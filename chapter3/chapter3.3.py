import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)

#构造一个pytorch数据迭代器
def load_array(data_arrays,batch_size,is_train=True): #@save
    dataset=data.TensorDataset(*data_arrays)
    #"TensorDataset" is a class provided by the torch.utils.data module which is a dataset wrapper that allows you to create a dataset from a sequence of tensors. 
    #"*data_arrays" is used to unpack the tuple into individual tensors.
    #The '*' operator is used for iterable unpacking.
    #Here, data_arrays is expected to be a tuple containing the input features and corresponding labels. The "*data_arrays" syntax is used to unpack the elements of the tuple and pass them as separate arguments.
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
    #Constructs a PyTorch DataLoader object which is an iterator that provides batches of data during training or testing.
batch_size=10
data_iter=load_array([features,labels],batch_size)
print(next(iter(data_iter)))#调用next()函数时会返回迭代器的下一个项目，并更新迭代器的内部状态以便下次调用

#定义模型变量，nn是神经网络的缩写
from torch import nn
net=nn.Sequential(nn.Linear(2,1))
#Creates a sequential neural network with one linear layer.
#Input size (in_features) is 2, indicating the network expects input with 2 features.
#Output size (out_features) is 1, indicating the network produces 1 output.

#初始化模型参数
net[0].weight.data.normal_(0,0.01)#The underscore at the end (normal_) indicates that this operation is performed in-place, modifying the existing tensor in memory.
net[0].bias.data.fill_(0)

#定义均方误差损失函数，也称平方L2范数，返回所有样本损失的平均值
loss=nn.MSELoss()#MSE:mean squared error 

#定义优化算法（仍是小批量随机梯度下降）
#update the parameters of the neural network (net.parameters()) using gradients computed during backpropagation. 
trainer=torch.optim.SGD(net.parameters(),lr=0.03)#SGD:stochastic gradient descent(随机梯度下降)

#训练
num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()#Updates the model parameters using the computed gradients and the optimization algorithm.
    l=loss(net(features),labels)
    print(f'epoch {epoch+1},loss {l:.6f}')#{l:.f}表示将变量l格式化为小数点后有6位的浮点数。
    
w=net[0].weight.data
print('w的估计误差：',true_w-w.reshape(true_w.shape))
b=net[0].bias.data
print('b的估计误差：',true_b-b)
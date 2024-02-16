import matplotlib.pyplot as plt
from torch import nn
import torch
from d2l import torch as d2l

#我们将实现一个具有单隐藏层的多层感知机，它包含256个隐藏单元，我们可以将层数和隐藏单元数都视为超参数
#我们通常选择2的若干次幂作为层的宽度，因为内存在硬件中的分配和寻址方式，这么做往往可以在计算上更高效。
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#我们用几个张量来表示我们的参数。注意，对于每一层我们都要记录一个权重矩阵和一个偏置向量。
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

#定义激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

#实现模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # '@'代表矩阵乘法
    return (H@W2 + b2)

#损失函数
loss = nn.CrossEntropyLoss(reduction='none')

#训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

#评估
d2l.predict_ch3(net, test_iter)

plt.show()
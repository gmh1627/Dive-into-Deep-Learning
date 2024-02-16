import torch
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l

#初始化
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(32)

# PyTorch不会隐式地调整输入的形状，因此我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)#仍然以均值0和标准差0.01随机初始化权重

net.apply(init_weights)#对神经网络net的每一层执行权重初始化
loss = nn.CrossEntropyLoss(reduction='none')
#如果reduction='none'，那么会返回一个损失向量，每个元素是对应样本的损失。如果reduction='mean'（默认值），那么会返回损失的平均值。如果reduction='sum'，那么会返回损失的总和。
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()
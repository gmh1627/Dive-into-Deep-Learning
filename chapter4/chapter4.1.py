import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from d2l import torch as d2l

#绘制ReLU函数图像
x=torch.arange(-8,8,0.1,requires_grad=True)
y=torch.relu(x)
d2l.plot(x.detach(),y.detach(),'x','relu(x)',figsize=(5,2.5))
#"detach()" is used to create a new tensor that shares the same data with x but doesn't have a computation graph
#plt.show()

#绘制ReLU函数的导数图像
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(),x.grad,'x','grad of relu(x)',figsize=(5,2.5))
#torch.ones_like(x): creates a tensor of the same shape as x but filled with ones. This tensor is used as the gradient of the output y with respect to x during backpropagation
#retain_graph=True: retains the computational graph after performing the backward pass
#plt.show()

#绘制sigmoid函数图像
y=torch.sigmoid(x)
d2l.plot(x.detach(),y.detach(),'x','sigmoid(x)',figsize=(5,2.5))
#plt.show()

#绘制sigmoid函数的导数图像
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(),x.grad,'x','grad of sigmoid(x)',figsize=(5,2.5))
#plt.show()

#绘制tanh函数图像
y=torch.tanh(x)
d2l.plot(x.detach(),y.detach(),'x','tanh(x)',figsize=(5,2.5))
#plt.show()

#绘制tanh函数的导数图像
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(),x.grad,'x','grad of tanh(x)',figsize=(5,2.5))
#plt.show()
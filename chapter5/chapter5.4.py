import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print(x2)

y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')#保存模型参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
#clone.eval()的目的是切换到评估模式，以确保在加载完模型参数后，模型的行为与推断时一致。
#在训练模式下，某些层的行为可能会导致不同的输出，因此通过切换到评估模式来避免这种不一致性。
Y_clone = clone(X)
print(Y_clone)
print(Y_clone == Y)
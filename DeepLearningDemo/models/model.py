import torch
from torch import nn

# 一般都是喜欢在model.py中单独来定义模型，然后在train.py中import
# 搭建一个名叫Jingxin的神经网络
class Jingxin(nn.Module):
    def __init__(self):
        super(Jingxin, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.model(x)
        return x

if __name__ == '__main__':
    jingxin=Jingxin()
    input=torch.ones((64,3,32,32))
    output=jingxin(input)
    print(output.shape)


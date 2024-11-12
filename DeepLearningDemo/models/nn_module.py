import torch
from torch import nn

class Jingxin(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self,input):
        output=input+1
        return output

Jingxin=Jingxin()
x=torch.tensor(1.0)
output=Jingxin(x)
print(output)
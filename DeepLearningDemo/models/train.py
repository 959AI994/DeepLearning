import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn

# 准备训练数据集
train_data=torchvision.datasets.CIFAR10(root="../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
# 准备测试数据集
test_data=torchvision.datasets.CIFAR10(root="../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 获取数据集长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用Dataloader来加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

# 搭建一个名叫Jingxin的神经网络
class Jingxin(nn.Moudle):
    def __init__(self):
        super(Jingxin, self).__init__()
    def forward(self,x):

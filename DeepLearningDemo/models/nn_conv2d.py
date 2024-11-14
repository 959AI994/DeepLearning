import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader
from torchvision import  datasets,transforms
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

# 这段代码基本上是一个简单的卷积神经网络的前向传播过程，并将输入图像写入 TensorBoard，用于监控模型的输入和输出情况。

# 定义图像转换，例如将图像转换为Tensor并进行归一化
process = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一调整所有图像为 256x256
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化为 [-1, 1] 范围
])

# 使用 ImageFolder 加载本地数据集
test_data = datasets.ImageFolder(
    root="E:/pythonProject/DeepLearningDemo/DeepLearningDemo/hymenoptera_data/train/",  # 本地数据集路径
    transform=process  # 指定转换操作
)
dataloader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)

class Jingxin(nn.Module):
    def __init__(self):
        super(Jingxin,self).__init__()
        self.conv1= Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return  x
writer=SummaryWriter("../logs")

step=0

model=Jingxin()

for data in dataloader:
    imgs,targets =data
    output = model(imgs)
    print(imgs.shape)
    print(output.shape)

    writer.add_image("input",imgs[0],step)

    # output= torch.reshape(output,(-1,-1,30,30))

    step=step+1
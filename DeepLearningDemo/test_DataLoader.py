import torchvision
from torch.utils.data import DataLoader
from torchvision import  datasets,transforms
from torch.utils.tensorboard import SummaryWriter

# 主要是看看dataloader是怎么用的
# 定义图像转换，例如将图像转换为Tensor并进行归一化
process = transforms.Compose([
    transforms.Resize((256, 256)),  # 统一调整所有图像为 256x256
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化为 [-1, 1] 范围
])

# 使用 ImageFolder 加载本地数据集
test_data = datasets.ImageFolder(
    root="E:/pythonProject/DeepLearningDemo/DeepLearningDemo/hymenoptera_data/train",  # 本地数据集路径
    transform=process  # 指定转换操作
)
test_loder=DataLoader(dataset=test_data,batch_size=8,shuffle=True,num_workers=0,drop_last=False)

# 测试数据集中的第一张图片及target

img, target=test_data[0]
print("Single image shape:", img.shape)
print("Single image target:", target)

writer=SummaryWriter("logs")
# 这里可以设置选两轮
for epoch in range(2):
    step=0
    for data in test_loder:
        imgs,targets=data
        print("Batch of images shape:", imgs.shape)  # 图像批次的形状
        print("Batch of targets:", targets)
        writer.add_images("test_data",imgs,step)
        step=step+1

writer.close()
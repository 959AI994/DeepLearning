import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from model import *
from torch.utils.tensorboard import SummaryWriter

writer =SummaryWriter("logs")

# 当前比较常见的调用gpu的方式是用.to(device)
device=torch.device("cuda")

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

# 从model.py中引入模型
jingxin=Jingxin()
# 将网络转移到设备中去
jingxin=jingxin.to(device)

# 损失函数,定义一个交叉熵损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)

# 定义一个优化器
learningRate=1e-2
optimizer=torch.optim.SGD(jingxin.parameters(),lr=learningRate)

# 设置训练次数
total_train_step=0
# 设置测试的次数
total_test_step=0
# 设置训练的次数
total_train_loss=0
epoch = 10

for i in range(epoch):
    print("-----------第{}轮训练开始-----------".format(i+1))
    jingxin.train()  # 设置成训练状态，也不是必须的，只会对dropout之类的起作用
    for data in train_dataloader:
        imgs,targets = data
        imgs=imgs.to(device)
        targets=targets.to(device)

        # 用模型训练并获得输出
        outputs=jingxin(imgs)
        # 计算损失
        loss=loss_fn(outputs,targets)
        # 先进行梯度清零
        optimizer.zero_grad()
        # 然后反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 输出
        total_train_step=total_train_step+1
        # 训练步骤是整百的时候才打印
        if total_train_step%100==0:
            print("训练次数:{},Loss:{}".format(total_train_step,loss.item()))
            # 绘制损失的图像
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    jingxin.eval() # 这一步不是必须的，作用很小
    total_test_loss=0 # 记录一个整个数据集上的loss
    total_acc=0
    # 设置无梯度，防止被更新
    with torch.no_grad():
        # 从测试集中取数据
        for data in test_dataloader:
            imgs,targets =data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs=jingxin(imgs)
            loss=loss_fn(outputs,targets)
            # 计算整体损失
            total_test_loss=total_test_loss+loss.item()
            # 计算整体准确率(分类)
            acc=(outputs.argmax(1)==targets).sum()
            total_acc=total_acc+acc

    print("整体测试集上的loss为：{}".format(total_test_loss))
    print("整体测试集上的acc为：{}".format(total_acc/test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_acc", total_acc/test_data_size, total_test_step)
    total_test_step=total_test_step+1

    # 保存每一轮模型的权重
    torch.save(jingxin,"jingxin_{}.pth".format(i))
    print("模型已保存")

writer.close()
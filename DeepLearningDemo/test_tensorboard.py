# 在当前py文件的根目录（DeepLearningDemo对应的终端运行：tensorboard --logdir=logs） 即可。
from torch.utils.tensorboard import SummaryWriter

writer= SummaryWriter("logs")
# 用tensorboard画一个一元二次函数
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()
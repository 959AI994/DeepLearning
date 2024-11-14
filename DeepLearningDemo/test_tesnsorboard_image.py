# 主要是希望通过TensorFlowboard看下我们给model提供了哪些数据
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer= SummaryWriter("logs")
# 用tensorboard显示图片
image_path="E:\\pythonProject\\DeepLearningDemo\\DeepLearningDemo\\hymenoptera_data\\train\\ants\\0013035.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train",img_array,2,dataformats='HWC')

# 终端运行tensorboard --logdir=logs  查看效果
writer.close()
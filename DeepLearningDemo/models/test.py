import torch
import torchvision
from PIL import Image
from torch import nn
import torchvision.transforms as transforms

# 该文件主要用于测试前面写好的model
imge_path ="./img_1.png"
image=Image.open(imge_path)
image = Image.open(imge_path).convert("RGB")  # 确保图像为 RGB 格式

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转换图像为 32x32 大小并转换为 Tensor
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像尺寸
    transforms.ToTensor()
])
image = transform(image)  # 使用 transform 对象来转换图像

# 添加批处理维度，使图像形状为 (1, 3, 32, 32)
# image = image.unsqueeze(0).to(device)
image = image.to(device)

# 将待评估的model加载进来
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

# 加载模型并将其移动到设备上
model = torch.load("jingxin_9.pth").to(device)
print(model)

image=torch.reshape(image,(1,3,32,32))

model.eval()
with torch.no_grad():
    output=model(image)
print(output)

print(output.argmax(1))
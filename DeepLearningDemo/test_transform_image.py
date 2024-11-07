from torchvision import  transforms
from PIL import  Image
from torch.utils.tensorboard import SummaryWriter

# 这里主要是将一张图片转换为一个tensor张量，本质就是第一个函数：transforms.ToTensor()
image_path="E:\\pythonProject\\DeepLearningDemo\\DeepLearningDemo\\hymenoptera_data\\train\\bees\\16838648_415acd9e3f.jpg"
img=Image.open(image_path)
# 1.将图片转换成tensor格式
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

print(tensor_img)

# 也可以用tensorboard来显示tensor
writer=SummaryWriter("logs")
writer.add_image("tensor_img",tensor_img)

# 2.归一化tensor，其实就是概率论里中心化处理(均值和方差)
print(tensor_img[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

# 3.Resize()函数来改变图片的尺寸
print(img.size)
# 定义函数
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
# 输出转换后图像大小
print(img_resize.size)
img_resize=tensor_trans(img_resize)
print(img_resize)
writer.add_image("Resize",img_resize)

# 4.Compose()函数，将多个操作组合在一起
# 创建一个图像预处理管道,其实就是把几个操作放到一起了
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
img_process=preprocess(img)
writer.add_image("Compose",img_process)

# 5.随机裁剪函数RandomCrop
trans_random=transforms.RandomCrop((100,200))
preprocess2=transforms.Compose([
    trans_random,
    tensor_trans
])
for i in range (10):
    img_crop=preprocess2(img)
    writer.add_image("RandomCrop",img_crop,i)


writer.close()

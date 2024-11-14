from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        # 把这些变量变成全局变量，主要是方便后续的调用和操作
        self.root_dir = root_dir
        self.label_dir = label_dir
        # 拼接地址，将train根目录与ants子母录进行拼接，就可以获得ants完整的路径
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获取图片列表,用listdir函数，传ants文件夹的路径即可
        self.image_path = os.listdir(self.path)

    # 操作列表下具体的每一张图片
    def __getitem__(self, idx):
        # 获取列表中idx索引对应的图片名称
        image_name = self.image_path[idx]
        # 获取idx的具体路径
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        # 获取图片
        image = Image.open(image_item_path)
        # 获取ants文件夹
        label = self.label_dir

        return image, label

    # 获取列表长度（数据集中数据的个数）

    def __len__(self):
        # 输入ants文件夹的路径即可
        return len(self.image_path)


# print(os.getcwd())
root_dir = os.path.join(os.getcwd(), "hymenoptera_data", "train")
ants_label_dir = "ants"
# 用MyData类来创建一个实体
ants_dataset = MyData(root_dir, ants_label_dir)
# 操作该实体类中的具体对象，比如访问ants文件夹下的第一个图片的内容
print(ants_dataset[0])
# 返回对象，注意，这里一定要返回两个参数，因为init中有两个参数
image,label=ants_dataset[1]
print(label)
image.show()
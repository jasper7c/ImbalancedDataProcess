
from PIL import Image  # 图片处理模块
import os  # 文件夹读取
from tqdm import tqdm  # 迭代进度显示化 放入迭代器即可

path = 'D:/srpdata/g/'


print("-" * 50)
print("训练集读取")
'''读取路径下所有子文件夹中的图片并存入list'''
train = []
dir_counter = 0
x = 0
i = 0
h = -1  # 标签初始值 迭代第一次为0

for child_dir in os.listdir(path):  # 两个for循环实现文件夹下所有子文件夹的遍历
    child_path = os.path.join(path, child_dir)
    h += 1  # 用于不同子文件夹定义标签
    for dir_image in tqdm(os.listdir(child_path)):
        im = Image.open(child_path + "\\" + dir_image)

        im = im.resize((28, 28))  # 原图压缩为60x60  以下带井号的语句用于测试
        im = im.convert("L")  # 灰度处理
        im = im.point(lambda x: 0 if x > 70 else 255)  # 二值化


        outpath = 'D:/srpdata/g_prepron/'

        im.save(os.path.join(outpath, str(h) + dir_image))  # 可以不创建文件，但一定要创建文件夹


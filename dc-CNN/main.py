from PIL import Image  # 图片处理模块
import numpy as np  # 数学处理
import os  # 文件夹读取
from tqdm import tqdm  # 迭代进度显示化 放入迭代器即可
from sklearn.model_selection import train_test_split  # 数据集的测试和训练样本划分
import tensorflow as tf
import prettytable
from tensorflow.keras import datasets, layers, models, metrics
from matplotlib import pyplot as plt  # 导入matplotlib进行绘图


file = 'E:\srpdata\g_data'
label = np.zeros(550)
data_ = np.zeros((550, 28, 28, 1))


def image(path):  # 读取所有子文件夹中图片处理为可用于分类的数据
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

            im = im.convert("L")  # 灰度处理
            im = im.point(lambda x: 0 if x > 120 else 1)  # 二值化
            # im.show()
            features_array = np.asarray(im)
            # np.set_printoptions(threshold=np.inf)
            # print(features_array)
            # data = np.sum(img_array, axis=0)
            # print(features_array)
            data = features_array.reshape((28, 28, 1))  # 向量化

            # print(data)
            data_[i, :, :, :] = data[:, :, :]  # 将当前图片处理后数据写入数据集
            label[x] = h  # 写入标签

            i += 1
            x += 1

    dir_counter += 1
    train.append(label)
    train.append(data_)

    return train


image(file)
# print('data_')  # 查看可用数据情况
data_ = np.array(data_)
# print(data_)
print(data_.shape)
print('label')
print(label)
print(label.shape)

train_images, test_images, train_labels, test_labels = train_test_split(data_, label, test_size=0.25, random_state=0)

print("train_images shape:{}".format(train_images.shape))  # 查看数据切分情况
print("test_images shape:{}".format(test_images.shape))
print("train_labels shape:{}".format(train_labels.shape))
print("test_labels shape:{}".format(test_labels.shape))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)


# 评价矩阵
class_names = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']

predictions = model.predict(test_images)

train_result = np.zeros((10, 10), dtype=int)
for i in range(138):
    train_result[int(test_labels[i])][int(np.argmax(predictions[i]))] += 1

result_table = prettytable.PrettyTable()
result_table.field_names = ['Type', 'Precision', 'Recall', 'Specificity', 'F1', 'G-mean']

for i in range(10):
    ps = train_result[i][i] / sum(train_result.T[i])
    rc = train_result[i][i] / sum(train_result[i])
    sf = (np.sum(train_result) + train_result[i][i] - sum(train_result[i]) - sum(train_result.T[i])) / (
                np.sum(train_result) - sum(train_result[i]))
    f1 = ps * rc * 2 / (ps + rc)
    gm = (rc * sf) ** 0.5
    ps = np.round(ps, 3)
    rc = np.round(rc, 3)
    sf = np.round(sf, 3)
    f1 = np.round(f1, 3)
    gm = np.round(gm, 3)

    result_table.add_row([class_names[i], ps, rc, sf, f1, gm])

print(result_table)
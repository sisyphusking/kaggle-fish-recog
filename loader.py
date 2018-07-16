import os
import numpy as np
from PIL import Image

# 数据集来源：https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data

path = "./data/train/"
labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


# 读取图片
def load_data(path):

    image_labels = []
    for index, label in enumerate(labels):
        for _ in os.listdir(path):
            if _ == label:
                label_dir = os.path.join(path, _)
                for k in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, k)
                    image_labels.append([image_path, index])
    return image_labels


# 数据预处理
def preprocess(dataset):

    np.random.shuffle(dataset)
    X = []
    Y = []
    for data in dataset:
        image = Image.open(data[0])
        image_vec = np.array(image)
        label = np.zeros(len(labels))
        label[data[1]] = 1
        X.append(image_vec)
        Y.append(label)
    return X, Y


def split_dataset(x, y, train_test_prop=0.8):

    data_num = len(x)
    train_data_num = int(data_num*train_test_prop)
    x_train = x[:train_data_num]
    y_train = y[:train_data_num]

    x_test = x[train_data_num:]
    y_test = y[train_data_num:]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    data = load_data(path)
    x, y = preprocess(data)
    x_train, y_train, x_test, y_test = split_dataset(x, y)
    print(x_train)


# todo
# 图片的尺寸怎样保持一致

# todo
# 怎样使用迭代器

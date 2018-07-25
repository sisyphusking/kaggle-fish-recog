import os
import numpy as np
from PIL import Image
import pickle
import glob


labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# 读取图片
# def load_images(path):
#
#     image_labels = []
#     for index, label in enumerate(labels):
#         for _ in os.listdir(path):
#             if _ == label:
#                 label_dir = os.path.join(path, _)
#                 for k in os.listdir(label_dir):
#                     image_path = os.path.join(label_dir, k)
#                     image_labels.append([image_path, index])
#     return image_labels


def load_images(path):
    images_labels = []
    for index, label in enumerate(labels):
        image_path = os.path.join(path, label, '*.jpg')
        files = glob.glob(image_path)

        for file in files:
            images_labels.append([file, index])
    return images_labels


# 数据预处理
def preprocess(dataset, width, height):

    np.random.shuffle(dataset)
    X = []
    Y = []
    for data in dataset:
        # image = Image.open(data[0])
        # image_vec = np.array(image)
        image_vec = resize_image(data[0], width, height)
        label = np.zeros(len(labels))
        label[data[1]] = 1
        X.append(image_vec)
        Y.append(label)
    return X, Y


# 切分数据集
def split_dataset(x, y, train_test_prop=0.8):

    data_num = len(x)
    train_data_num = int(data_num*train_test_prop)
    x_train = x[:train_data_num]
    y_train = y[:train_data_num]

    x_test = x[train_data_num:]
    y_test = y[train_data_num:]

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    # return x_train, y_train, x_test, y_test


# reshape图片
def resize_image(path, width, height):

    image = Image.open(path)

    # 等比缩放
    # (x, y) = image.size
    # x_s = width
    # y_s = int(y * x_s / x)
    # resize_image = image.resize((x_s, y_s), Image.ANTIALIAS)

    # 固定长度和宽度
    resize_image = image.resize((width, height), Image.ANTIALIAS)
    # resize_image.save('data/reshape.jpg')
    resize_image_vec = np.array(resize_image)
    return resize_image_vec


def plot_image(path):

    if isinstance(path, str):

        image = Image.open(path)
        print(path)
        (x, y) = image.size
        print((x, y))
        image.show()
    else:
        new_image = Image.fromarray(path.astype(np.uint8), 'RGB')
        new_image.show()


def pickle_data(obj, file):

    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def reload_pickle(file):

    with open(file, 'rb')as f:
        data = pickle.load(f)
    return data


def load_data(path, data_set_x, data_set_y, width, height):

    if not (data_set_x and data_set_y):
        data = load_images(path)
        x, y = preprocess(data, width, height)
        return split_dataset(x, y)

    if not (os.path.exists(data_set_x) or os.path.exists(data_set_y)):
        data = load_images(path)
        x, y = preprocess(data, width, height)
        pickle_data(x, data_set_x)
        pickle_data(y, data_set_y)

    x_load = reload_pickle(data_set_x)
    y_load = reload_pickle(data_set_y)

    return split_dataset(x_load, y_load)


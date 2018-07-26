import loader
import config
from scipy.misc import imread
import matplotlib.pyplot as plt
import seaborn as sns

base_config = config.BaseConfig
data_set = loader.load_images(base_config.PATH)


def plt_images_size():
    images_size = {}
    for data in data_set:
        image_array = imread(data[0])
        size = "*".join(map(str, list(image_array.shape)))
        images_size[size] = images_size.get(size, 0) + 1
    plt.figure(figsize=(12, 4))
    sns.barplot(list(images_size.keys()), list(images_size.values()), alpha=0.8)
    plt.xlabel("image size", fontsize=12)
    plt.ylabel("number of images", fontsize=12)
    plt.title("images size present in dataset")
    plt.savefig('./data/image/images-size-proportion.jpg')


plt_images_size()

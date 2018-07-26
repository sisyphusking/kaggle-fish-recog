import loader
import augment
import tensorflow as tf
from keras.preprocessing import image
import  matplotlib.pyplot as plt
import numpy as np

data_set = loader.load_images("../data/train/")


class TestDataAug(tf.test.TestCase):

    def test_data_aug(self):

        img = image.load_img(data_set[0][0])
        plt.imshow(img)
        plt.savefig("../data/image/origin.jpg")
        plt.show()
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        train_set = augment.DataGen(x, np.array([1]), batch_size=1)
        plt.figure()
        for i in range(3):
            for j in range(3):
                _x, y = train_set.next()
                idx = (3 * i) + j
                # plt.subplot(3, 3, idx + 1)
                plt.imshow(_x[0]/256)
                plt.savefig('../data/image/{}-{}.jpg'.format(i, j))
                plt.show()


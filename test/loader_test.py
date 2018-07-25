import glob
import os
import loader
import random
import config
import tensorflow as tf
labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
path = "../data/train/"
param_config = config.InceptionModel


class TestLoader(tf.test.TestCase):

    def test_image_load(self):

        images_path = []
        for index, label in enumerate(labels):

            image_path = os.path.join(path, label, '*.jpg')
            files = glob.glob(image_path)

            for file in files:

                images_path.append([file, index])
        print(len(images_path))

    def test_plot_image(self):
        x_train, y_train, x_test, y_test = loader.load_data(
            param_config.PATH,
            param_config.X_PKL,
            param_config.Y_PKL,
            param_config.INCEPT_WIDTH,
            param_config.INCEPT_HEIGHT
        )
        loader.plot_image(x_train[random.randint(0, len(x_train))])
        # print(x_train[0])

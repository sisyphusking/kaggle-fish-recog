import glob
import os
import loader
from PIL import Image
import random

labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
path = "../data/train/"


class TestLoader:

    def test_image_load(self):

        images_path = []
        for index, label in enumerate(labels):

            image_path = os.path.join(path, label, '*.jpg')
            files = glob.glob(image_path)

            for file in files:

                images_path.append([file, index])

        # print(images_path)
        print(len(images_path))

    def test_plot_image(self):

        x_train, y_train, x_test, y_test = loader.load_data(loader.path)
        loader.plot_image(random.choice(x_train))

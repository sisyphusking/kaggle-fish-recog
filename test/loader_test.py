import glob
import os

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

        print(images_path)
        print(len(images_path))

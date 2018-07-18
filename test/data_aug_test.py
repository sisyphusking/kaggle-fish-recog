import loader
import data_augment
import tensorflow as tf


class TestDataAug(tf.test.TestCase):

    def test_data_aug(self):
        x_load = loader.reload_pickle('../data/data_set_x.pkl')
        y_load = loader.reload_pickle('../data/data_set_y.pkl')
        x_train, y_train, x_test, y_test = loader.split_dataset(x_load, y_load)
        for i in range(3):
            _x_train, _y_train = data_augment.image_data_augment(x_train, y_train)
            print("*"*20)
            print(_y_train)
        print(_x_train.shape)


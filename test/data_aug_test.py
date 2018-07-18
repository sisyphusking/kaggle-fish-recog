import loader
import data_augment
import tensorflow as tf


class TestDataAug(tf.test.TestCase):

    def test_data_aug(self):
        x_load = loader.reload_pickle('../data/data_set_x.pkl')
        y_load = loader.reload_pickle('../data/data_set_y.pkl')
        x_train, y_train, x_test, y_test = loader.split_dataset(x_load, y_load)
        train_set = data_augment.DataGen(x_train, y_train)
        for i in range(3):
            x, y = train_set.next()
            print("*"*20)
            print(y)
        print(x.shape)


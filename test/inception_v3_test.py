import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib import slim


# 参考链接：https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3_test.py
class InceptionV3Test(tf.test.TestCase):
    def testBuildClassificationNetwork(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = inception.inception_v3(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith(
            'InceptionV3/Logits/SpatialSqueeze'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])
        self.assertTrue('Predictions' in end_points)
        self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                             [batch_size, num_classes])

    def testBuildPreLogitsNetwork(self):
        batch_size = 5
        height, width = 299, 299
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        net, end_points = inception.inception_v3(inputs, num_classes)
        final = end_points['PreLogits']
        print(final.get_shape().as_list())
        # self.assertTrue(net.op.name.startswith('InceptionV3/Logits/AvgPool'))
        self.assertListEqual(final.get_shape().as_list(), [batch_size, 1, 1, 2048])

    def testBuildBaseNetwork(self):
        batch_size = 5
        height, width = 299, 299

        inputs = tf.random_uniform((batch_size, height, width, 3))
        final_endpoint, end_points = inception.inception_v3_base(inputs)
        self.assertTrue(final_endpoint.op.name.startswith(
            'InceptionV3/Mixed_7c'))
        self.assertListEqual(final_endpoint.get_shape().as_list(),
                             [batch_size, 8, 8, 2048])
        expected_endpoints = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
                              'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3',
                              'MaxPool_5a_3x3', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
                              'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d',
                              'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c']
        self.assertItemsEqual(end_points.keys(), expected_endpoints)


if __name__ == '__main__':
    tf.test.main()

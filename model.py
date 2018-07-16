import tensorflow as tf
import loader
import tensorflow.contrib.slim as slim
from tensorflow.keras.applications import inception_v3  # 加载keras中保存的模型
import tensorflow.contrib.slim.nets as nets

x_train = loader.reload_pickle('data/x_train.pkl')
y_train = loader.reload_pickle('data/y_train.pkl')

x_test = loader.reload_pickle('data/x_test.pkl')
y_test = loader.reload_pickle('data/y_test.pkl')


x = tf.placeholder(tf.float32, [None, loader.width, loader.height, 3])
y = tf.placeholder(tf.float32, [None, 8])

# 读取训练好的模型
with slim.arg_scope(nets.inception.inception_v1_arg_scope()):
    logits, end_points = nets.inception.inception_v3(inputs=x, num_classes=8)

probabilities = tf.nn.softmax(logits)
accuracy = end_points['Predictions']


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    load_train_set = loader.DataSet(x_train, y_train)
    for i in range(1):
        _x, _y = load_train_set.next_batch()
        sess.run([logits, end_points], feed_dict={x: _x, y: _y})


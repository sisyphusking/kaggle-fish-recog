import tensorflow as tf
import loader
from data_augment import DataGen
import tensorflow.contrib.slim as slim
from tensorflow.keras.applications import inception_v3  # 加载keras中保存的模型
import tensorflow.contrib.slim.nets as nets
from config import InceptionModel

param_config = InceptionModel()
x_train, y_train, x_test, y_test = loader.load_data(
                                                    param_config.PATH,
                                                    param_config.X_PKL,
                                                    param_config.Y_PKL,
                                                    param_config.INCEPT_WIDTH,
                                                    param_config.INCEPT_HEIGHT
                                                )

# 加载数据集的生成器
# load_train_set = loader.generator(x_train, y_train)
# load_test_set = loader.generator(x_test, y_test)

train_set = DataGen(x_train, y_train, batch_size=param_config.BATCH_SIZE)
test_set = DataGen(x_test, y_test, batch_size=param_config.BATCH_SIZE)

n_batch = x_train.shape[0]//param_config.BATCH_SIZE  # “//”运算符是执行除法后，取整

x = tf.placeholder(tf.float32, [None, param_config.INCEPT_WIDTH, param_config.INCEPT_HEIGHT, 3])
y = tf.placeholder(tf.float32, [None, 8])

# 读取inception_v3模型
with slim.arg_scope(nets.inception.inception_v3_arg_scope()):

    # 这个输出的是[batch_size, num_classes]
    logits, end_points = nets.inception.inception_v3(inputs=x, is_training=True, reuse=tf.AUTO_REUSE)
    # final_endpoint, end_points = nets.inception.inception_v3_base(inputs=x)   # 这个的输出是8*8*2048

final = end_points['PreLogits']  # final: [batch_size, 1, 1, 2048]

# probabilities = tf.nn.softmax(logits)  # 输出的y值 [batch_size, num_classes]
# predicted_labels = tf.argmax(end_points['Predictions'], 1)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# flattern层
h_flat = tf.reshape(final, [-1, 1*1*2048])

W_fc1 = weight_variable([1*1*2048, 8])  # 上一层有1*1*2048个神经元，全连接层有8个神经元
b_fc1 = bias_variable([8])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_flat, W_fc1)+b_fc1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))

# 使用AdamOptimizer优化
# train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

# 使用梯度下降优化
train_step = tf.train.GradientDescentOptimizer(7e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:

    # 允许复用参数，一般在使用adam优化器时可能会出现这种问题
    # tf.get_variable_scope().reuse_variables()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        for batch in range(n_batch):
            # _x, _y = load_train_set.next_batch()
            _x, _y = train_set.next()
            print(len(_x))
            prob = sess.run(train_step, feed_dict={x: _x, y: _y})
        # _x_test, _y_test = load_train_set.next_batch()
        _x_test, _y_test = test_set.next()
        acc = sess.run(accuracy, feed_dict={x: _x_test, y: _y_test})
        print("Iter " + str(i) + ", Testing Accuracy= " + str(acc))

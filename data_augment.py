from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 参考文档：https://keras-cn.readthedocs.io/en/latest/preprocessing/image/

# def image_data_augment(x, y):
#     data_gen = ImageDataGenerator(
#             rescale=None,  # 1./255,
#             shear_range=0.1,
#             zoom_range=0.1,
#             rotation_range=10.,
#             width_shift_range=0.1,
#             height_shift_range=0.1,
#             horizontal_flip=True)
#     data_gen.fit(x)
#
#     generator = data_gen.flow(x, y, batch_size=64)
#     return generator


class DataGen:
    def __init__(self, x, y, batch_size=64):
        self.data_gen = ImageDataGenerator(
            rescale=None,  # 1./255,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=10.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

        self.data_gen.fit(x)
        self.generator = self.data_gen.flow(x, y, batch_size=batch_size)

    def next(self):
        return self.generator.next()


# 第一种迭代器
# def next_batch(x, y, batch_size=20):
#
#     x_batch = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
#     y_batch = np.zeros((batch_size, len(labels)), dtype=np.uint8)
#     data_num = len(x)
#     while True:
#         for i in range(batch_size):
#             index = random.randint(0, data_num-1)
#             x_batch[i] = x[index]
#             y_batch[i] = y[index]
#         yield x_batch, y_batch


# 第二种迭代器
class generator:

    def __init__(self, x, y):
        self._index_in_epoch = 0
        self._x = x
        self._y = y
        self._num_examples = x.shape[0]   # len(x)
        self._epochs_completed = 0

    def next_batch(self, batch_size=32):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # 完成一轮
            self._epochs_completed += 1
            # 打乱数据
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._x = self._x[perm]
            self._y = self._y[perm]
            # 开始新的迭代
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._x[start:end], self._y[start:end]



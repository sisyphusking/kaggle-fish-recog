from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 参考文档：https://keras-cn.readthedocs.io/en/latest/preprocessing/image/
def image_data_augment(x, y):
    data_gen = ImageDataGenerator(
        rescale=None,  # 1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    data_gen.fit(x)

    generator = data_gen.flow(x, y, batch_size=64)
    return generator.next()


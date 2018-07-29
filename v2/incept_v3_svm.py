from config import InceptionModel
import loader
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import augment
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm

# https://www.kaggle.com/craigglastonbury/using-inceptionv3-features-svm-classifier/comments
# https://becominghuman.ai/transfer-learning-retraining-inception-v3-for-custom-image-classification-2820f653c557

# todo svm
param_config = InceptionModel()

if not os.listdir(param_config.DEST_PATH):
    augment.ImageGen(param_config.PATH, param_config.DEST_PATH).gen_image()
print("generate images end...")
data_set = loader.load_images(param_config.DEST_PATH)
X = [data[0] for data in data_set]
Y = [data[1] for data in data_set]


def extract_features(model_path, X, pickle_file=None):

    if os.path.exists(pickle_file):
        return loader.reload_pickle(pickle_file)

    nb_features = 2048
    features = np.empty((len(X), nb_features))
    with tf.Session() as sess:
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for index, x in enumerate(X):
            image = tf.gfile.FastGFile(x, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image})
            features[index, :] = np.squeeze(predictions)
    loader.pickle_data(features, pickle_file)
    return features


def extract_lables(y, pickle_file=None):

    if os.path.exists(pickle_file):
        return loader.reload_pickle(pickle_file)

    labels = []
    for i in y:
        label = np.zeros(8)
        label[i] = 1
        labels.append(label)
    labels = np.array(labels)
    loader.pickle_data(labels, pickle_file)
    return labels


if __name__ == '__main__':

    features = extract_features(param_config.MODEL_PATH, X, param_config.PICKLE_X_FILE)
    # labels = extract_lables(Y, param_config.PICKLE_Y_FILE)   # 使用label会报错,svm中y值不能是one-hot形式

    x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.1, random_state=0)

    clf = SVC(kernel='linear', C=0.1).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            count+=1
    print(count)
    print("accuracy: ", count/len(y_test))



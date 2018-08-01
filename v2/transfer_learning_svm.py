from config import InceptionModel
import loader
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import augment
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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


def train(x_train, y_train, x_test, y_test, kernel='linear', C=1):
    clf = SVC(kernel=kernel, C=1)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    # y_pred = clf.predict(x_test)
    # accuracy = sum([y_pred[i] == y_test[i] for i in range(len(y_test))]) / len(y_test)
    print('the score of {}--{} is {}'.format(kernel, C, str(score)))


if __name__ == '__main__':

    param_config = InceptionModel()

    if not os.listdir(param_config.DEST_PATH):
        augment.ImageGen(param_config.PATH, param_config.DEST_PATH).gen_image()
    print("generate images end...")

    data_set = loader.load_images(param_config.DEST_PATH)
    X = [data[0] for data in data_set]
    Y = [data[1] for data in data_set]

    features = extract_features(param_config.MODEL_PATH, X, param_config.PICKLE_X_FILE)
    # 使用label会报错,svm中y值不能是one-hot形式
    x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.1, random_state=0)

    # 从2048维度降到200
    n_components = 200
    pca = PCA(n_components=n_components).fit(x_train)

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    param_grid = {
        "C": [1e3, 5e3, 1e4, 1e5],
        "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                       param_grid=param_grid, n_jobs=-1).fit(x_train_pca, y_train)

    accuracy = clf.score(x_test_pca, y_test)
    print("accuracy: ", accuracy)
    # 打印出最优的分类器以及参数
    print("the best estimator: ", clf.best_estimator_)
    y_pred = clf.predict(x_test_pca)
    labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    print(classification_report(y_test, y_pred, target_names=labels))
    # 对角线数字越多，就表示准确率越高
    print(confusion_matrix(y_test, y_pred, labels=range(8)))


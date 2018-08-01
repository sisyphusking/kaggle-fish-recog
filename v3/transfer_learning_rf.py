from v2.transfer_learning_svm import extract_features, extract_lables
from config import InceptionModel
import loader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import augment
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

    param_config = InceptionModel()

    if not os.listdir(param_config.DEST_PATH):
        augment.ImageGen(param_config.PATH, param_config.DEST_PATH).gen_image()
    print("generate images end...")
    data_set = loader.load_images(param_config.DEST_PATH)
    X = [data[0] for data in data_set]
    Y = [data[1] for data in data_set]

    features = extract_features(param_config.MODEL_PATH, X, param_config.PICKLE_X_FILE)
    labels = extract_lables(Y, param_config.PICKLE_Y_FILE)

    x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.1, random_state=0)

    # 特征缩放
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(x_train)
    # X_test = sc.transform(x_test)

    # pca降维
    # n_components = 500
    # pca = PCA(n_components=n_components).fit(x_train)
    # x_train_pca = pca.transform(x_train)
    # x_test_pca = pca.transform(x_test)

    parameters = {
                    'n_estimators': [200],
                    'max_depth': [5],
                    'min_samples_split': [3],
                    'min_samples_leaf': [3]
                }

    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=6)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("current score : ", str(score))
    print("best_score: ", clf.best_score_)
    print("best_params: ", clf.best_params_)

    y_pred = clf.predict(x_test)
    labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    print(classification_report(y_test, y_pred, target_names=labels))
    # 对角线数字越多，就表示准确率越高
    cfm = confusion_matrix(y_test, y_pred, labels=range(8))
    print("confusion matrix: ", cfm)


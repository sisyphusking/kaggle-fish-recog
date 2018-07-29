# global config
class BaseConfig:
    PATH = "../data/train/"
    DEST_PATH = '../data/train2/'
    LABELS = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    X_PKL = '../data/data_set_x.pkl'
    Y_PKL = '../data/data_set_y.pkl'


# inception_v3 model config
class InceptionModel(BaseConfig):
    INCEPT_WIDTH = 299
    INCEPT_HEIGHT = 299
    BATCH_SIZE = 32
    MODEL_PATH = '../data/model/classify_image_graph_def.pb'
    PICKLE_X_FILE = '../data/features.pkl'
    PICKLE_Y_FILE = '../data/labels.pkl'


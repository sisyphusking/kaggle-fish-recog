# global config
class BaseConfig:
    PATH = "./data/train/"
    LABELS = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    X_PKL = '../data/data_set_x.pkl'
    Y_PKL = '../data/data_set_y.pkl'


# inception_v3 model config
class InceptionModel(BaseConfig):
    INCEPT_WIDTH = 299
    INCEPT_HEIGHT = 299
    BATCH_SIZE = 32

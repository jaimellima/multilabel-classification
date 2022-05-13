
from importlib.resources import path
from os.path import dirname, join, abspath
DIR_PATH = '/home/jolima/Documentos/multilabel-classification/multi-label-classification/dataset'

current_dir = dirname(DIR_PATH)

KAGGLE_DATASET = "dataset/kaggle_dataset.csv"
FILE_PATH = join(current_dir, KAGGLE_DATASET)

TERM_SIZE_STD = 8
RAM_STD = 16
IGNORE_ZERO_WSD = False

N_COMPONENTES_SVD = 100
K_BEST_FS = 10

SAMPLE = 100

MAX_FEATURES_TFIDF = 500

TAGS_COLUMNS = {
        'CS':['Computer Science', 0],
        'PH':['Physics', 1],
        'MA':['Mathematics', 2],
        'ST':['Statistics', 3],
        'QB':['Quantitative Biology', 4],
        'QF':['Quantitative Finance', 5],
    }

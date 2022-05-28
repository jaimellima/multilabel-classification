
#from importlib.resources import path
from os.path import dirname, join, abspath
DIR_PATH = '/home/jolima/Documentos/multilabel-classification/multi-label-classification/'
#DIR_PATH = '/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/multilabel_code/'

CURRENT_DIR = dirname(DIR_PATH)

KAGGLE_DATASET = "dataset/kaggle_dataset.csv"
JSON_BINARY = "dataset/data_bin.json"
BINARIES_DIR = "dataset/binaries"
FILE_PATH = join(CURRENT_DIR, KAGGLE_DATASET)
BINARIES_PATH = join(CURRENT_DIR, BINARIES_DIR)

TERM_SIZE_STD = 8
RAM_STD = 16

MIN_TERM_SIZE = 3
MAX_TERM_SIZE = 64
IGNORE_ZERO_WSD = False

N_COMPONENTES_SVD = 100
K_BEST_FS = 10

SAMPLE = 10000

MAX_FEATURES_TFIDF = 500

TAGS_COLUMNS = {
        'CS':['Computer Science', 0],
        'PH':['Physics', 1],
        'MA':['Mathematics', 2],
        'ST':['Statistics', 3],
        'QB':['Quantitative Biology', 4],
        'QF':['Quantitative Finance', 5],
    }

Y_COLUMNS = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
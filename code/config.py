
#from importlib.resources import path
from os.path import dirname, join, abspath
DIR_PATH = '/home/jolima/Documentos/multilabel-classification/'
#DIR_PATH = '/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/multilabel_code/'

CURRENT_DIR = dirname(DIR_PATH)

KAGGLE_DATASET = "/media/jolima/DADOS/kaggle_multilabel/kaggle_dataset.csv"
BINARY_CSV = "/media/jolima/DADOS/kaggle_multilabel/binaries/kaggle_10/3.csv"
X_JSON_BINARY = "/media/jolima/DADOS/kaggle_multilabel/binaries/kaggle_100/3.json"
Y_JSON_BINARY = "/media/jolima/DADOS/kaggle_multilabel/binaries/kaggle_100/3_ps_labels.json"
BINARIES_DIR = "/media/jolima/DADOS/kaggle_multilabel/binaries"
FILE_PATH = join(CURRENT_DIR, KAGGLE_DATASET)
BINARIES_PATH = join(CURRENT_DIR, BINARIES_DIR)
KAGGLE_100 = "/media/jolima/DADOS/kaggle_multilabel/binaries/kaggle_100/"
KAGGLE_1000 = "/media/jolima/DADOS/kaggle_multilabel/binaries/kaggle_1000/"

TERM_SIZE_STD = 8
RAM_STD = 16

MIN_TERM_SIZE = 3
MAX_TERM_SIZE = 64

MIN_RAM_SIZE = 3
MAX_RAM_SIZE = 64

SEEDS = 5

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

Y_COLUMNS = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
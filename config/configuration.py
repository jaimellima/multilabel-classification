#MOVER PARA ARQUIVO DE CONFIGURAÇÃO, CRIANDO FUNÇÃO PARA CARREGAR ARQUIVO
conf = {
    #raw file to be preprocessed
    "ORIGIN_CSV_FILE": "./dataset/train.csv",

    #directory to save preprocessed files.
    "DEST_DIR_PREPROCESSED": "/home/jolima/Documentos/multilabel-classification/",

    #standard thermometer value
    "THERM_SIZE_STD": 8,

    #sample size
    "SAMPLE": 100,

    #columns to be processed
    "X_COLUMNS": ["TITLE", "ABSTRACT"],

    #label columns
    "Y_COLUMNS":["Computer Science"],

    #vectorization method
    #"TF" - Term Frequency
    #"TFIDF" - Term Frequency Inverse Document Frequency
    "VECTORIZATION_METHOD": "TF",

    "SPACY_MODEL": "en_core_web_lg",

    "RAM_STD": 16,
    "MIN_TERM_SIZE": 3,
    "MAX_TERM_SIZE": 32,
    "MIN_RAM_SIZE": 8,
    "MAX_RAM_SIZE": 16,
    "SEEDS": 2,
    "IGNORE_ZERO_WSD": False,
    
}


class Configuration():
    def __init__(self):
        self._config = conf

    def get_property(self, property_name):
        return self._config.get(property_name)

    def set_property(self, property_name, value):
        self._config[property_name] = value


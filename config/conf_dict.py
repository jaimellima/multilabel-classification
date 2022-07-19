#TODO: MOVER PARA ARQUIVO DE CONFIGURAÇÃO, CRIANDO FUNÇÃO PARA CARREGAR ARQUIVO
conf_dict = {
    #raw file to be preprocessed
    "ORIGIN_CSV_FILE": "/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/kaggle/train.csv",

    #directory to save preprocessed files.
    "DIR_TOKENIZED_FILES": "/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/multilabel_code/corpus/preprocessed/",

    "DIR_BINARIZED_FILES": "/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/multilabel_code/corpus/binarized/",

    "DIR_VECTORIZED_FILES": "/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/multilabel_code/corpus/vectorized/",

    "DIR_TEXT_FILES": "/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/multilabel_code/corpus/raw/",

    #standard thermometer value
    "THERM_SIZE_STD": 8,

    #sample size
    "SAMPLE": 100,

    #columns to be processed
    "X_COLUMNS": ["TITLE", "ABSTRACT"],

    #label columns
    "Y_COLUMNS":["Computer Science", "Physics"],

    #vectorization method
    #"TF" - Term Frequency
    #"TFIDF" - Term Frequency Inverse Document Frequency
    "VECTORIZATION_STRATEGY": "TF",

    "SPACY_MODEL": "en_core_web_lg",

    "RAM_STD": 16,
    "MIN_TERM_SIZE": 3,
    "MAX_TERM_SIZE": 32,
    "MIN_RAM_SIZE": 8,
    "MAX_RAM_SIZE": 16,
    "SEEDS": 2,
    "IGNORE_ZERO_WSD": False,
    
}
from dataset import Dataset
from preprocessing import Preprocessing

#importação do arquivo de configurações
import config as cfg
import en_core_web_lg

nlp = en_core_web_lg.load()

data = Dataset(cfg.KAGGLE_DATASET, ["ABSTRACT"], ["Computer Science"], "ABSTRACT", 10)
preprocessing = Preprocessing()
print(data.X_term_frequency)
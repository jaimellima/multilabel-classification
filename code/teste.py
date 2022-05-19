from dataset import Dataset

#importação do arquivo de configurações
import config as cfg

data = Dataset(cfg.KAGGLE_DATASET, ["ABSTRACT"], ["Computer Science"])

print(data.dataset)
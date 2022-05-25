import pandas as pd
import config as cfg
import en_core_web_lg

from preprocessing import Preprocessing
from spacy.lang.en.stop_words import STOP_WORDS


dataset = pd.read_csv(cfg.KAGGLE_DATASET)
#print(dataset)

# preprocessing = Preprocessing(path_dataset=cfg.KAGGLE_DATASET, n_sample=cfg.SAMPLE,
#      X_columns=["TITLE", "ABSTRACT"], y_columns=['Computer Science'],
#      column_text_name="TEXT", nlp_model=en_core_web_lg.load(), stop_words=STOP_WORDS)

dataset['LABELS'] = dataset[["Computer Science", "Physics"]].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
print([label[0] for label in dataset['LABELS']])
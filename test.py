import sys
sys.path.append('./')

from config.configurationFactory import ConfigurationFactory
from models.document import Document
from models.documentFactory import DocumentFactory


ds_configs = ConfigurationFactory.get_configuration(type="dataset")
print(ds_configs.y_columns)

import pandas as pd
import json

df = pd.read_csv("/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/kaggle/train.csv")
print(df.head())

df = df.sample(n=2)

documents = []

for index, row in df.iterrows():
    text = row["TITLE"] + row["ABSTRACT"]
    documentFactory = DocumentFactory(index, text=text)
    textDocument = documentFactory.get_document(type="text")
    textDocument.to_json(ds_configs.dest_dir_preprocessed)
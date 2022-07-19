import os, json
import sys
sys.path.append('./')
import pandas as pd


class Json:
    @staticmethod
    def save_json(dictionary, path, file_name, file_type=".json"):
        #TODO: USAR BIBLIOTECA PATH. 
        #PASSAR CAMINHO COM "/"
        fname = '{}{}{}'.format(path, file_name, file_type)
        with open(fname, 'w') as f:
            json.dump(dictionary, f)
    
    @staticmethod
    def list_jsons(path, file_type=".json"):
        json_files = [ps_json for ps_json in os.listdir(path) if ps_json.endswith(file_type)]
        return json_files

    @staticmethod
    def load_json(path, file_name):
        with open('{}{}'.format(path, file_name)) as json_file:
            data = json.load(json_file)
            return data

class Csv:
    def __init__(self, fname, x_columns, y_columns):
        self._fname = fname
        self._x_columns = x_columns
        self._y_columns = y_columns
        self._dataframe = pd.read_csv(self._fname)

    def get_text_from_csv(self, n_sample):
        self._dataframe = self._dataframe.sample(n=n_sample)
        self._dataframe["TEXT"] = self._dataframe[self._x_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        self._dataframe["LABELS"] = self._dataframe[self._y_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        print("Saving text file...")
        documents = []
        for index, row in self._dataframe.iterrows():
            doc = {"document_id": None, "text":"", "labels":""}
            doc["document_id"] = index
            doc["text"] = row["TEXT"]
            doc["labels"] = [row["LABELS"]]
            documents.append(doc)
        return documents
            

if __name__=="__main__":
    from config.configuration import ConfigurationSettings, Configuration
    from dao import Json
    cfg = ConfigurationSettings()
    #valor passado para teste
    cfg.sample = 10
    print(cfg.origin_csv_file)
    csv = Csv(cfg.origin_csv_file, cfg.x_columns, cfg.y_columns)
    documents = csv.get_text_from_csv(n_sample=5)

    for doc in documents:
        Json.save_json(doc, cfg.dest_dir_text,str(doc["document_id"]),".text")

    
        
    

#    path = "/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/kaggle/"
#    json_files = Json.list_jsons(path, file_type=".text")
#    for file in json_files:
#        data = Json.load_json(path, file)
#        print(data["document_id"])


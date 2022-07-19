import sys
sys.path.append('./')

from dataset import Dataset
from dao.data import Json
from config.configuration import ConfigurationSettings
from documentFactory import DocumentFactory

class DatasetFactory:
    def __init__(self, directory, file_type):
        self._directory = directory
        self._files = Json.list_jsons(directory, file_type=file_type)

    def load_from_files(self):
        dataset = Dataset()
        for file in self._files:
            data = Json.load_json(self._directory, file_name=file)
            dataset.insert_document(type="text", document_id=data["document_id"], text=data["text"], labels=data["labels"])
        return dataset


if __name__=="__main__":
    cfg = ConfigurationSettings()
    ds_fac = DatasetFactory(cfg.dest_dir_text, ".text")
    dataset = ds_fac.load_from_files()
    print(dataset.get_X())


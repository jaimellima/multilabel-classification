import sys
sys.path.append('./')

from .conf_dict import conf_dict
from abc import ABC

class Configuration():
    # this class loads the configuration file and creates 
    # the generic methods to be used by the configuration 
    # classes for dataset, preprocessing, binarization 
    # and classification.

    def __init__(self):
        self._config = conf_dict

    def get_property(self, property_name):
        return self._config.get(property_name)

    def set_property(self, property_name, value):
        self._config[property_name] = value

class ConfigurationSettings(Configuration):
    def __init__(self):
        super().__init__()
    
    @property
    def origin_csv_file(self):
        return self.get_property("ORIGIN_CSV_FILE")
    
    @origin_csv_file.setter
    def origin_csv_file(self, value):
        self.set_property("ORIGIN_CSV_FILE", value)

    @property
    def dest_dir_preprocessed(self):
        return self.get_property("DEST_DIR_PREPROCESSED")

    @property
    def dest_dir_text(self):
        return self.get_property("DIR_TEXT_FILES")

    @property
    def x_columns(self):
        return self.get_property("X_COLUMNS")
    
    @x_columns.setter
    def x_columns(self, list_columns):
        self.set_property("X_COLUMNS", list_columns)

    @property
    def y_columns(self):
        return self.get_property("Y_COLUMNS")
    
    @y_columns.setter
    def y_columns(self, list_columns):
        self.set_property("Y_COLUMNS", list_columns)

    @property
    def sample(self):
        return self.get_property("SAMPLE")
    
    @sample.setter
    def sample(self, value):
        self.set_property("SAMPLE", value)

    @property
    def vectorization_strategy(self):
        return self.get_property("VECTORIZATION_STRATEGY")

    @vectorization_strategy.setter
    def vectorization_strategy(self, value):
        self.set_property("VECTORIZATION_STRATEGY", value)

    @property
    def spacy_method(self):
        return self.get_property("SPACY_MODEL")

    @spacy_method.setter
    def spacy_method(self, value):
        self.set_property("SPACY_MODEL", value)


if __name__=="__main__":
    config = Configuration()
    print(config.get_property("SAMPLE"))
from .configuration import Configuration

class PreprocessingConfig(Configuration):
    def __init__(self):
        super().__init__()
    
    @property
    def origin_csv_file(self):
        return self.get_property("ORIGIN_CSV_FILE")
    
    @origin_csv_file.setter
    def origin_csv_file(self, value):
        self.set_property("ORIGIN_CSV_FILE", value)

    @property
    def sample(self):
        return self.get_property("SAMPLE")
    
    @sample.setter
    def sample(self, value):
        self.set_property("SAMPLE", value)

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
    def vectorization_method(self):
        return self.get_property("VECTORIZATION_METHOD")

    @vectorization_method.setter
    def vectorization_method(self, value):
        self.set_property("VECTORIZATION_METHOD", value)

    @property
    def spacy_method(self):
        return self.get_property("SPACY_MODEL")

    @spacy_method.setter
    def spacy_method(self, value):
        self.set_property("SPACY_MODEL", value)

import sys
sys.path.append('./')

from abc import ABC, abstractclassmethod
from dao.data import Json


#para a criação dos documentos, usar a classe DocumentFactory em documentFactory.py
class Document(ABC):
    """
    Abstract class for document model. For creating the documents, 
    use the DocumentFactory class in documentFactory.py
    """
    def __init__(self, document_id, labels=None):
        self._document_id = document_id
        self._labels = labels

    @property
    def document_id(self):
        return self._document_id
    
    @property
    def labels(self):
        return self._labels

    @abstractclassmethod
    def to_dict(self):
        pass
        
    def to_json(self, dir, file_type=".json"):
        """save files in json or other chosen format (eg .text, .token, .vect, .binary)

        Args:
            dir (str): directory where the files will be saved
            file_type (str, optional): types for the files. Defaults to ".json".
        """
        Json.save_json(self.to_dict(), dir, file_name=str(self.document_id), file_type=file_type)
        print("Saving document {} file in {} format".format(self.document_id, file_type))

class TextDocument(Document):
    def __init__(self, document_id, text, labels=None):
        super().__init__(document_id, labels = labels)
        self._text = text

    @property
    def text(self):
        return self._text

    def to_dict(self):
        document_dict = {}
        document_dict["document_id"] = self.document_id
        document_dict["text"] = self.text
        document_dict["labels"] = self.labels
        return document_dict

    def to_json(self, dir, file_type=".text"):
        """save files in json or other chosen format (eg .text, .token, .vect, .binary)

        Args:
            dir (str): directory where the files will be saved
            file_type (str, optional): types for the files. Defaults to ".json".
        """
        Json.save_json(self.to_dict(), dir, file_name=str(self.document_id), file_type=file_type)
        print("Saving document {} file in {} format".format(self.document_id, file_type))


class TokenizedDocument(Document):
    def __init__(self, document_id, tokens, labels=None):
        super().__init__(document_id, labels = labels)
        self._tokens = tokens

    @property
    def tokens(self):
        return self._tokens

    def to_dict(self):
        pass

    def to_json(self, dir, file_type=".token"):
        """save files in json or other chosen format (eg .text, .token, .vect, .binary)

        Args:
            dir (str): directory where the files will be saved
            file_type (str, optional): types for the files. Defaults to ".json".
        """
        Json.save_json(self.to_dict(), dir, file_name=str(self.document_id), file_type=file_type)
        print("Saving document {} file in {} format".format(self.document_id, file_type))

class VectorizedDocument(Document):
    def __init__(self, document_id, vector, labels=None):
        super().__init__(document_id, labels = labels)
        self._vector = vector

    @property
    def vector(self):
        return self._vector

    def to_dict(self):
        pass

    def to_json(self, dir, file_type=".vect"):
        """save files in json or other chosen format (eg .text, .token, .vect, .binary)

        Args:
            dir (str): directory where the files will be saved
            file_type (str, optional): types for the files. Defaults to ".json".
        """
        Json.save_json(self.to_dict(), dir, file_name=str(self.document_id), file_type=file_type)
        print("Saving document {} file in {} format".format(self.document_id, file_type))

class BinarizedDocument(Document):
    def __init__(self, document_id, binary_vector, labels=None):
        super().__init__(document_id, labels = labels)
        self._vector = binary_vector

    @property
    def binary_vector(self):
        return self._vector

    def to_dict(self):
        pass

    def to_json(self, dir, file_type=".binary"):
        """save files in json or other chosen format (eg .text, .token, .vect, .binary)

        Args:
            dir (str): directory where the files will be saved
            file_type (str, optional): types for the files. Defaults to ".json".
        """
        Json.save_json(self.to_dict(), dir, file_name=str(self.document_id), file_type=file_type)
        print("Saving document {} file in {} format".format(self.document_id, file_type))


if __name__=="__main__":
    textDocument = TextDocument(document_id=10, text="teste")
    print(textDocument.text)
    print(textDocument.document_id)
    print(textDocument.to_dict())
    textDocument.to_json("/home/jaimel/GoogleDrive/Doutorado/experimentos/Classificacao_Multilabel_Documentos_Cientificos/kaggle/")
    

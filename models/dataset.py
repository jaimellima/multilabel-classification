import sys
sys.path.append('./')

from documentFactory import DocumentFactory
from document import TextDocument, TokenizedDocument, VectorizedDocument, BinarizedDocument
import numpy as np


class Dataset():
    def __init__(self):
        self._documents = []
        
    def get_documents(self):
        return self._documents
    
    def get_X(self):
        X = []
        for document in self.get_documents():
            if isinstance(document, TextDocument):
                X.append(document.text)
            elif isinstance(document, TokenizedDocument): 
                X.append(document.tokens)
            elif isinstance(document, VectorizedDocument):
                X.append(document.vector)
            else:
                X.append(document.binary_vector)
        return X

    def get_y(self):
        y = []
        for document in self.get_documents():
            y.append(document.labels)
        return y

    def insert_document(self, type, document_id, text=None, tokens=None, vector=None, labels=None):
        documentFactory = DocumentFactory(document_id=document_id, text=text, tokens=tokens, vector=vector, labels=labels)
        document = documentFactory.get_document(type=type)
        self._documents.append(document)
        del documentFactory
        del document
   

if __name__=="__main__":
    import numpy as np
    dataset = Dataset()
    dataset.insert_document(document_id=10, type="tokenized", tokens=[1,0,1], labels=[1,2])
    dataset.insert_document(document_id=20, type="tokenized", tokens=[1,1,1], labels=[1,1])
    X = np.array(dataset.get_X())
    print(X)
    y = np.array(dataset.get_y())
    print(y)
    print(type(dataset))


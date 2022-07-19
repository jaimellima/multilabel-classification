import sys
sys.path.append('./')

from document import TextDocument, TokenizedDocument, VectorizedDocument, BinarizedDocument

class DocumentFactory:
    def __init__(self, document_id, text=None, tokens=None, vector=None, labels=None):
        self._document_id = document_id
        self._text = text
        self._vector = vector
        self._tokens = tokens
        self._labels = labels

    def get_document(self, type):
        """creates an instance of a given document type.

        Args:
            type (_type_): text, tokenized, vectorized, binarized

        Returns:
            _type_: Document (TextDocument, TokenizedDocument, VectorizedDocument or BinarizedDocument)
        """
        document = None
        if type=="text":
            document = TextDocument(self._document_id, self._text, self._labels)
        elif type=="tokenized":
            document = TokenizedDocument(self._document_id, self._tokens, self._labels)
        elif type=="vectorized":
            document = VectorizedDocument(self._document_id, self._vector, self._labels)
        elif type=="binarized":
            document = BinarizedDocument(self._document_id, self._vector, self._labels)
        else:
            document = None
        return document

        

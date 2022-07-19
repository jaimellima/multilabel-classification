
from preprocess import Preprocess

class PreprocessFactory:
    def __init__(self):
        self._preprocess = self.get_preprocessor()

    def get_preprocessor(self):
        preprocess = Preprocess()
        return preprocess

    def execute(self):
        #executa os passos do preprocessamento do texto.
        print(self._preprocess.get_y())
    
#TESTING
if __name__=="__main__":
    prep = PreprocessFactory()
    prep.execute()
    
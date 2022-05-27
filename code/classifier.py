
import wisardpkg as wsd
import numpy as np

class Classifier:
    def __init__(self, args=None):
        self.__args = None
        
    def wisard(self, X_train, y_train, X_test, ram, ignoreZero=False):
        wisard = wsd.Wisard(ram, ignoreZero=ignoreZero)
        wisard.train(X_train, y_train)
        y_pred = np.array(wisard.classify(X_test))
        return y_pred

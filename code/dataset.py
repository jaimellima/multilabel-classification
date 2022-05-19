import pandas as pd
from sklearn.model_selection import KFold

class Dataset:
    def __init__(self, path: str, X_columns: list, y_columns: list):
        """Generate Train and Test datasets from csv file

        Args:
            path (str): _description_
            X_columns (list): _description_
            y_columns (list): _description_
        """

        self.__dataset = pd.read_csv(path)
        self.__X = self.__dataset[X_columns]
        self.__y = self.__dataset[y_columns]
        self.__X_train_folds = []
        self.__X_test_folds = []
        self.__y_train_folds = []
        self.__y_test_folds = []

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, new_dataset: pd.DataFrame):
        self.__dataset = new_dataset

    @property
    def X_train_folds(self):
        return self.__X_train_folds

    @property
    def X_test_folds(self):
        return self.__X_test_folds

    @property
    def y_train_folds(self):
        return self.__y_train_folds

    @property
    def y_test_folds(self):
        return self.__y_test_folds
 
    def kfold_split(self, n_folds:int = 5):
        kf = KFold(n_splits=n_folds)
        for train_index, test_index in kf.split(self.dataset):
            self.__X_train_folds.append(self.__X[train_index])
            self.__y_train_folds.append(self.__y[train_index])
            self.__X_test_folds.append(self.__X[test_index])
            self.__y_test_folds.append(self.__y[test_index])


            




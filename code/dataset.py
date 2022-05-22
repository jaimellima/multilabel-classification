import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from preprocessing import Preprocessing
import en_core_web_lg
from sklearn.feature_extraction.text import CountVectorizer

class Dataset:
    def __init__(self, path: str, X_columns: list, y_columns: list, text_column_name: str, sample: int):
        """Generate Train and Test datasets from csv file

        Args:
            path (str): path to CSV file.
            X_columns (list): feature set
            y_columns (list): label set
        """

        self.__dataset = pd.read_csv(path)
        self.__dataset = self.__dataset.sample(n=sample)
        self.__X = self.__dataset[X_columns]
        self.__y = self.__dataset[y_columns]
        self.__X_preprocessed = []
        self.__X_term_frequency = None
        self.__y = self.__dataset[y_columns]
        self.__X_train_folds = []
        self.__X_test_folds = []
        self.__y_train_folds = []
        self.__y_test_folds = []
        self.__text_column_name = text_column_name

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y

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

    @property
    def X_term_frequency(self):
        return self.__X_term_frequency
 
 
    def kfold_split(self, n_folds:int = 5):
        kf = KFold(n_splits=n_folds)
        for train_index, test_index in kf.split(self.X):
            self.__X_train_folds.append(self.X.loc[train_index])
            self.__y_train_folds.append(self.y.loc[train_index])
            self.__X_test_folds.append(self.X.loc[test_index])
            self.__y_test_folds.append(self.y.loc[test_index])

    def preprocessing(self):
        documents = self.__X[self.__text_column_name]
        nlp_model = en_core_web_lg.load()
        preproc = Preprocessing()
        for text in documents:
            cleaned_text = preproc.remove_punctuations(text)
            cleaned_text = preproc.remove_stop_words(cleaned_text)
            cleaned_text = " ".join([word for word in cleaned_text])
            lemma_text = preproc.lemmatize_tokenize(nlp_model, cleaned_text)
            final_text = " ".join([word for word in lemma_text])
            self.__X_preprocessed.append(final_text)

    def term_frequency(self):
        vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 3))
        self.__X_term_frequency = vectorizer2.fit_transform(self.__X_preprocessed)
        self.__X_term_frequency = self.__X_term_frequency.todense()
        #self.__X_term_frequency = pd.DataFrame(self.__X_term_frequency)




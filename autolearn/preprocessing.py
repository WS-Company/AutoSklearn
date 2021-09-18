"""
Предобработка данных для обучения
"""

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from autolearn.autonormalizetransformer import AutoNormalizeTransformer


class DataPreprocessor:
    """
    Предобработка данных
    """

    def __init__(self,
                 *,
                 auto_normalize_before_transform: bool = False,
                 auto_normalize_after_transform: bool = False,
                 transformer: str = None,
                 pca: int = None):
        """
        Инициализация предобработчика
        """
        self.auto_normalize_before_transform = auto_normalize_before_transform
        self.auto_normalize_after_transform = auto_normalize_after_transform
        self.transformer = transformer
        self.pca = pca
        self._pipeline = self.get_pipeline()

    def read_data(self, filename: str):
        """
        Считывает данные из файла CSV.

        Параметры
        ---------

        filename: str
            Имя файла, содержащего исходные данные. Это должен быть CSV
            файл без заголовка. Последний столбец считается столбцом значений
            целевой переменной, остальные - признаками

        Возвращает
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Матрица признаков для обучения

        y: numpy.ndarray, shape (n_samples, )
            Вектор значений целевой переменной для обучения
        """
        # Считываем данные из файла, без заголовка
        data = pd.read_csv(filename, header=None)
        # И объявляем последний столбец значениями целевой переменной
        # Сразу после чтения столбцы будут индексированы числами, так
        # что столбец с максимальным номером переименовываем в y,
        # а с остальными номерами N - в строку xN.
        last_column = len(data.columns) - 1
        data.rename(
            columns=lambda x: "y" if x == last_column else "x{}".format(x),
            inplace=True
        )
        # Делим данные на обучающие и тестовые
        X = data.drop('y', axis='columns').astype(float)
        y = data['y'].astype(float).values
        return (X, y)

    def fit(self, X, y):
        if self._pipeline is not None:
            self._pipeline.fit(X, y)

    def transform(self, X, y):
        """

        Параметры
        ---------
        X: numpy.ndarray, shape (n_samples, n_features)
            Исходная матрица признаков для обучения

        y: numpy.ndarray, shape (n_samples, )
            Исходный вектор значений целевой переменной для обучения

        Возвращает
        ----------
        X: numpy.ndarray, shape (n_samples, n_features)
            Преобразованная матрица признаков для обучения
        """
        if self._pipeline is not None:
            return self._pipeline.transform(X, y)
        return X

    def fit_transform(self, X, y = None):
        if self._pipeline is not None:
            return self._pipeline.fit_transform(X, y)
        return X

    def load_data(self, filename: str):
        """
        Загружает данные из файла, применяя к ним необходимые преобразования
        """
        (X, y) = self.read_data(filename)
        return (self.fit_transform(X, y), y)

    def load_train_test(self,
                        filename: str,
                        *,
                        test_size: float = 0.2):
        """
        Загружает данные из CSV-файла, разделяет их на обучающие и тестовые
        и выполняет все преобразования исходных данных
        """
        (X, y) = self.read_data(filename)
        (
            X_train, X_test, y_train, y_test
        ) = train_test_split(
            X, y, test_size = test_size
        )
        self.fit(X_train, y_train)
        X_train = self.transform(X_train, y_train)
        X_test = self.transform(X_test, y_test)
        return (X_train, X_test, y_train, y_test)

    def get_pipeline(self):
        steps = self.get_pipeline_steps()
        if steps:
            return Pipeline(steps)
        return None

    def get_pipeline_steps(self):
        steps = []
        if self.auto_normalize_before_transform:
            steps.append(("autonorm", AutoNormalizeTransformer()))
        if self.pca:
            steps.append(("pca", PCA(n_components=self.pca)))
        if self.auto_normalize_after_transform:
            steps.append(("autonorm", AutoNormalizeTransformer()))

    def wrap_model(self, model):
        """
        Упаковывает модель в конвейр вместе с несколькими препроцессорами

        Параметры
        ---------
        model: sklearn.BaseEstimator
            Модель классификации или регрессии Scikit-Learn или другого
            фреймворка с совместимым интерфейсом. Может быть как
            классификатором, так и регрессором.

        Возвращает
        ----------
        packed: sklearn.Pipeline
            Конвейр Scikit-Learn, содержащий model в качестве последней
            модели в цепочке, а перед ней - все предобработчики данных.
        """
        steps = self.get_pipeline_steps()
        steps.append(("model", model))
        return Pipeline(steps)

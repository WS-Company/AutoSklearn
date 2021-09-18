"""
Предобработка данных для обучения
"""

import pandas as pd

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

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

"""
Обучение модели регрессии
"""

import sys

import pandas as pd

from autolearn.metrics import assymmetric_mse

from autolearn.autonormalize import autonormalize


class RegressionTrainer:
    """
    Обчение модели регрессии
    """

    def __init__(self, *,
                 data_filename: str,
                 time_limit: float = None,
                 time_limit_per_model: float = None,
                 n_jobs: int = -1,
                 ensemble_size: int = 1,
                 final_model: str = "first",
                 auto_normalize: bool = False,
                 preprocessor: str = None,
                 pca: int = None,
                 verbosity: int = 9,
                 use_xgboost: bool = False,
                 metric_assymmetry: float = None):
        """
        Инициализация учителя для модели регрессии.

        Параметры
        ---------
        data_filename: str
            Имя файла с данными. Это должен быть CSV файл без заголовка.
            Последний столбец считается столбцом значений целевой переменной.

        time_limit: float
            Максимальное время обучения результирующей модели в секундах
        """
        # Запоминаем все переданные параметры
        self.data_filename = data_filename
        self.time_limit = time_limit
        self.time_limit_per_model = time_limit_per_model
        self.n_jobs = n_jobs
        self.ensemble_size = ensemble_size
        self.final_model = final_model
        self.auto_normalize = auto_normalize
        self.preprocessor = preprocessor
        self.pca = pca
        self.verbosity = verbosity
        self.use_xgboost = use_xgboost
        self.metric_assymmetry = metric_assymmetry
        # Проверяем переданные параметры на совместимость
        if self.use_xgboost and self.metric_assymmetry:
            raise ValueError(
                "Ассиметричную метрику нельзя использовать с XGBoost"
            )
        if self.pca and self.preprocessor:
            raise ValueError(
                "Метод главных компонент нельзя использовать вместе с препроцессором"
            )

    def prepare_data(self):
        """
        Подготовка данных для обучения

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
        if self.auto_normalize:
            X = np.hstack([x.values for x in autonormalize.auto_normalize(X)])
        y = data['y'].astype(float).values
        return (X, y)

    def fit_model(self, model, X, y):
        """
        Обучает модель model на матрице признаков X и векторе целевых
        переменных y. Если модель имеет тип AutoSklearnRegressor, то
        выполняет отдельно обучение моделей и обучение ансамбля.

        Параметры
        ---------
        model : sklearn.BaseEstimator
            Модель, которую будем обучать, может быть регрессором
            для Scikit-Learn, AutoSklearnRegressor или XGBRegressor.

        X : numpy.ndarray размера n_samples * n_features
            Матрица признаков для обучения модели

        y : numpy.ndarray размера n_features
            Вектор значений целевой переменной для обучения модели

        Возвращает
        ----------
        model : sklearn.BaseEstimator
            Аргумент model
        """
        if self.preprocessor or self.pca:
            if isinstance(model.steps[-1][1], AutoSklearnRegressor):
                model.fit(X, y)
                if self.ensemble_size:
                    model.steps[-1][1].fit_ensemble(
                        y,
                        ensemble_size=self.ensemble_size
                    )
            else:
                model.fit(X, y)
        elif isinstance(model, AutoSklearnRegressor):
            model.fit(X, y)
            if self.ensemble_size:
                model.fit_ensemble(
                    y,
                    ensemble_size=self.ensemble_size
                )
        else:
            model.fit(X, y)
        return model

    def display_model_score(self,
                            X,
                            y,
                            model,
                            model_name: str = "модели",
                            data_name: str = "тестовых данных"):
        """
        Выводин на экран информацию о метриках качества модели

        Параметры
        ---------
        X : numpy.ndarray, размера n_samples * n_features
            Матрица признаков

        y : numpy.ndarray, размера n_samples
            Вектор целевых значений

        model : sklearn.BaseEstimator
            Модель, уже обученная на некотором подмножестве (X, y).

        model_name : строка, по умолчанию "модели"
            Как будет названа модель в выводимых на экран сообщениях

        data_name : строка, по умолчанию "тестовых данных"
            Как будет назван набор данных в выводимых на экран
            сообщениях
        """
        y_pred = model.predict(X)
        sys.stdout.write(
            "Среднеквадратичная ошибка {} на {} равна {}\n".format(
                model_name, data_name, mean_squared_error(y, y_pred)
            )
        )
        if self.metric_assymmetry is not None:
            sys.stdout.write(
                "Ассиметричная среднеквадратичная ошибка {} на {} равна {}\n".format(
                    model_name, data_name, assymmetric_mse(
                        y, y_pred, extra_argument=metric_assymmetry
                    )
                )
            )
        sys.stdout.write(
            "Коэффициент детерминации {} на {} равен {}\n".format(
                model_name, data_name, r2_score(y, y_pred)
            )
        )

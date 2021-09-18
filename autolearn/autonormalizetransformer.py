from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from autolearn.autonormalize import autonormalize


class AutoNormalizeTransformer(BaseEstimator, TransformerMixin):
    """
    Преобразование autolearn.autonormalize.auto_normalize с интерфейсом
    трансформера Scikit-Learn
    """

    def __init__(self):
        """Конструктор, сводится к вызову конструктора родительского класса"""
        super().__init__()

    def fit(self, X, y = None):
        """
        Обучение модели. Этот метод должен быть, но в данном случае он ничего
        не делает

        Параметры
        ---------
        X: любой
            Игноорируется, нужен для совместимости с BaseEstimator.fit

        y: любой, необязателен
            Игноорируется, нужен для совместимости с BaseEstimator.fit
        """
        return self

    def transform(self, X):
        """
        Преобразует матрицу признаков

        Параметры
        ---------
        X: numpy.ndarray, shape (n_samples, n_features)
        """
        np.hstack([x.values for x in autonormalize.auto_normalize(X)])

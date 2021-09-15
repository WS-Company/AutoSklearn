"""
Дополнительные метрики для обучения моделей.
"""

import numpy as np
import sys
import warnings

from sklearn.metrics import mean_squared_error


def assymmetric_mse(y_true, y_pred, *,
                    sample_weight = None,
                    multioutput='uniform_average',
                    squared: bool = True,
                    extra_argument: float = 1):
    """
    Ассмтетричная среднеквадратичная ошибка

    Параметры
    ---------
    y_true: numpy.array, shape (n.samples, n_features)
        Настоящие значения целевой переменной

    y_pred: numpy.array, shape (n.samples, n_features)
        Предсказанные значения целевой переменной

    sample_weight: array, shape (n_samples), необязателен

    multioutput: str, необязателен

    squared: bool, по умолчанию True

    extra_argument: float, по умолчанию 1
        Любое расхождение, при котором предсказанное значение больше по
        модулю, чем целевое, умножать на это число
    """
    # Параметр y_pred здесь оказывается одномерным, а y_true
    # двумерным с одним измерением равным 1
    y_pred = y_pred.reshape(y_true.shape)
    mult = np.ones(y_true.shape)
    mult[np.abs(y_pred) > np.abs(y_true)] *= extra_argument
    value = mean_squared_error(
        y_true * mult, y_pred * mult,
        sample_weight=sample_weight,
        multioutput=multioutput,
        squared=squared
    )
    return value

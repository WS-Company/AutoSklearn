"""
Дополнительные метрики для обучения моделей.
"""

import numpy as np

from sklearn.metrics import mean_squared_error


def get_assymmetric_mse(factor: float = 4):
    """
    Создает функцию метрики для обучения моделей регрессии. Создаваемая
    функция будет аналогична среднеквадратичной ошибке, однако ошибки
    в сторону от 0 будут считаться в `factor` раз более весомыми, чем
    ошибки в сторону 0. То есть, если значение y_true = 10, то возврат
    y_pred = 11 будет считаться более существенной ошибкой модели, чем
    возврат y_pred = 9.

    Параметры
    ---------
    factor: float, по умолчанию 4
        Во сколько раз ошибки в большую по модулю сторону считаются более
        грубыми, чем в меньшую. Значение должно быть положительным, значение
        1 означает что ошибки в обе стороны будут равноценны, а меньше
        1 - что ошибка к 0 более грубая, чем от 0.

    Возвращает
    ----------
    mse: callable
        Функция, возвращающая скорректированную среднеквадратичную ошибку.
        Имеет те же параметры, что и sklearn.metrics.mean_squared_error
    """
    def mse(y_true, y_pred, *,
            sample_weight = None,
            multioutput='uniform_average',
            squared: bool = True):
        """Скорректированная среднеквадратичная ошибка"""
        mult = np.ones(y_true.shape)
        mult[np.abs(y_pred) > np.abs(y_true)] *= factor
        return mean_squared_error(
            y_true * mult, y_pred * mult,
            sample_weight=sample_weight,
            multioutput=multioutput,
            squared=squared
        )

    return mse

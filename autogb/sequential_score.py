"""
Качество модели для работы с временными рядами
"""

import pandas as pd


def sequential_r2_score(model,
                        X,
                        y,
                        *,
                        start: int = 10,
                        step: int = 1,
                        sample_weight = None,
                        assymmetry: float = None,
                        quantile: float = None):
    """
    Последовательная оценка R2 для модели регрессии. Вычисляется
    следующим образом. Сначала модель обучается на `start` первых
    примерах и вычисляется ее качество на последующих `step` примерах.
    Затем модель обучается на `start + step` примерах и вычисляется
    ее качество на последующих `step` примерах. Затем модель обучается
    на `start + step * 2` примерах и так далее. Возвращается среднее
    значени получаемых ошибок, если параметр `quantile` равен None,
    иначе возвращается соответствующее значение квантили последовательности
    ошибок.

    Параметры
    ---------
    model: sklearn.BaseEstimator
        Модель регрессии, совместимая с интерфейсом Scikit-Learn

    X: numpy.ndarray формы (n_samples, n_features)

    y: numpy.ndarray формы (n_samples, )

    start: int, по умолчанию 10
        Количество примеров, на которых изначально обучаем модель

    step: int, по умолчанию 1

    sample_weight: numpy.ndarray формы (n_samples), необязателен
        В данный момент не используется

    assymmetry: float, необязателен
        В данный момент не используется

    quantile: float, необязателен
    """
    if sample_weight is not None or assymmetry is not None:
        raise NotImplementedError(
            "Unsupported parameters: sample_weight and assymmetry"
        )
    scores = []
    (n_samples, n_features) = X.shape
    for i in range(start, n_samples, step):
        X_train = X[:i]
        X_test = X[i:i + step]
        y_train = y[:i]
        y_test = y[i:i + step]
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    if quantile is None:
        return pd.Series(scores).mean()
    else:
        return pd.Series(scores).quantile(quantile)

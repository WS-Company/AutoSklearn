#! /usr/bin/env python

"""
Сценарий для обучения регрессора на основе AutoSklearn.

Параметры

--model=ИМЯ_ФЙЛА или -m ИМЯ_ФАЙЛА
    Имя файла, в который нужно сохранить наилучшую модель (для
    сохранения используется joblib).

--file=ИМЯ_ФАЙЛА или -f ИМЯ_ФАЙЛА
    Файл в формате CSV, хранящий обучающие данные. Файл должен
    быть CSV, без строки заголовка, целевая переменная в последнем
    столбце, остальные - обучающие

--time-limit=СЕКУНДЫ
    Ограничить общее время обучения итоговой модели заданным числом
    секунд

--time-limit-per-model=СЕКУНДЫ
    Ограничить время обучения каждой из отдельных моделей заданным
    числом секунд.

--quiet или -q
    Подавлять вывод предупреждений Scikit-Learn в процессе работы.
    Разные алгоритмы в процессе обучения могут выводить разные
    предупреждения

--n-jobs ЧИСЛО
    Запускать обучение в заданное число потоков. По умолчанию
    используются все доступные процессоры. Задайте явно значение
    1, чтобы использовать один процессор.

--ensemble-size РАЗМЕР_АНСАМБЛЯ
    Размер итогового ансамбля моделей, рекомендуется 1

--final-model ТИП_МОДЕЛИ
    Определяет вид модели, которая в итоге будет сохранена в файл.
    Допускаются следующие значения:

    *   "source" - оригинальный объект AutoSklearnRegressor.
        Сценарий для выполнения прогноза при этом работает, но
        совместимость с моделями Scikit-Learn в прочих аспектах
        не гарантируется.

    *   "first" - сохраняется наилучшая модель из ансамбля. Она
        может быть несколько хуже, чем весь ансамбль

    *   "voting" - сохранить ансамбль моделей в виде объекта
        VotingRegressor. В настоящее время не работает из-за
        внутренней ошибки Auto-Sklearn. Разработчикам сообщено
        об ошибке, ждем.

--auto-normalize
    Использовать дополнительные механизмы автоматической нормализации
    данных перед обучением.

--pca ЧИСЛО
    Перед запуском обучения привести матрицу признаков к главным
    компонентам, и оствить это количество наиболее значимых из них

--preprocessor=power|quantile
    Перед передачай данных модели Auto-Sklearn преобразовать их
    в равномерное распределение (QuantileTransformer) или в нормальное
    (PowerTransformer)

--verbosity ЧИСЛО
    Выводить на экран отладочную информацию в процессе работы программы.
    Чем больше значение, тем больше отладочной информации будет выведено.

Примечания

1.  AutoSklearnRegressor имеет временные ограничения по умолчанию.
    Если нужно обучать его неограниченно долго, нужно вручную установить
    очень большие значения для time-limit и time-limit-per-model
"""

import sys
import warnings
import joblib
import datetime

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from autosklearn.regression import AutoSklearnRegressor

from autosklearn.metrics import make_scorer

#from autonormalize import autonormalize

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from xgboost import XGBRegressor

from metrics import assymmetric_mse


def prepare_data(filename: str, auto_normalize: bool = False):
    """
    Считывает данные из файла filename, отделяет последний столбец
    как столбец целевых значений, остальное - как матрицу признаков

    Параметры
    ---------
    filename : строка
        Имя файла с данными. Это должен быть файл в CSV формате
        без заголовка

    auto_normalize : Логическое, по умолчанию False
        Преобразовать данные с помощью утилиты autonormalize

    Возвращает
    ----------
    X : numpy.ndarray, размер n_samples * n_features
        Матрица признаков, состоящая из всех считанных данных,
        кроме последнего столбца

    y : numpy.ndarray, размер n_samples
        Вектор целевых значений
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
    if auto_normalize:
        X = np.hstack([x.values for x in autonormalize.auto_normalize(X)])
    y = data['y'].astype(float).values
    return (X, y)


def fit_model(model,
              X,
              y,
              *,
              ensemble_size: int = 1,
              preprocessor: str = None):
    """
    Обучает модель model на матрице признаков X и векторе целевых
    переменных y. Если модель имеет тип AutoSklearnRegressor, то
    выполняет отдельно обучение моделей и обучение ансамбля.

    Параметры
    ---------
    model : sklearn.BaseEstimator
        Модель, которую будем обучать, может быть регрессором
        для Scikit-Learn или AutoSklearnRegressor.

    X : numpy.ndarray размера n_samples * n_features
        Матрица признаков для обучения модели

    y : numpy.ndarray размера n_features
        Вектор значений целевой переменной для обучения модели

    ensemble_size : целое, по умолчанию 1
        Размер ансамбля при использовании AutoSklearnRegressor.
        Можно передать None, тогда считается, что нужный размер
        ансамбля уже передан в конструктор объекта.

    preprocessor : строка
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.

    Возвращает
    ----------
    model : sklearn.BaseEstimator
        Аргумент model
    """
    if preprocessor:
        if isinstance(model.steps[1][1], AutoSklearnRegressor):
            model.fit(X, y)
            if ensemble_size:
                model.steps[1][1].fit_ensemble(
                    y,
                    ensemble_size=ensemble_size
                )
        else:
            model.fit(X, y)
    elif isinstance(model, AutoSklearnRegressor):
        model.fit(X, y)
        if ensemble_size:
            model.fit_ensemble(
                y,
                ensemble_size=ensemble_size
            )
    else:
        model.fit(X, y)
    return model


def display_model_score(X,
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
    sys.stdout.write(
        "Коэффициент детерминации {} на {} равен {}\n".format(
            model_name, data_name, r2_score(y, y_pred)
        )
    )


def pack_model(reg, preprocessor: str = None, pca: int = None):
    """
    Упаковывает модель reg в Pipeline с препроцессором,
    определенным параметрами preprocessor и pca. В настоящее
    время допускается использование только одного из
    двух параметров preprocessor и pca.

    Параметры
    ---------

    reg : sklearn.BaseEstimator
        Исходный классификатор

    preprocessor : строка
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.

    pca: int, необязателен
        Если задан (и не None), то перед запуском обучения привести
        данные к главным компонентам и оставить только это количество
        наиболее значимых компонент.

    Возвращает
    ----------
    pipe : sklearn.BaseEstimator
        Комбинированный регрессор, сочетающий reg с указанным
        препроцессором, или reg, если препроцессор не указан
    """
    if not preprocessor and pca is None:
        return reg
    if pca:
        if not preprocessor:
            return Pipeline(
                [
                    ('pca', PCA(n_components=pca)),
                    ('reg', reg)
                ]
            )
        raise ValueError("Опции pca и preprocessor пока что взаимоисключающие")
    if preprocessor.lower() == "quantile":
        return Pipeline(
            [
                ('pre', QuantileTransformer()),
                ('reg', reg)
            ]
        )
    if preprocessor.lower() == "power":
        return Pipeline(
            [
                ('pre', PowerTransformer()),
                ('reg', reg)
            ]
        )
    raise ValueError("Препроцессор может быть 'quantile' или 'power', а не '{}'".format(preprocessor))


def fit_autosklearn_regressor(X_train,
                              X_test,
                              y_train,
                              y_test,
                              *,
                              time_limit: int = None,
                              time_limit_per_model: int = None,
                              n_jobs: int = -1,
                              ensemble_size: int = 1,
                              preprocessor: str = None,
                              pca: int = None,
                              use_xgboost: bool = False,
                              scorer=None):
    """
    Создаем Auto-SkLearn регрессор и устанавливаем временные
    ограничения для его работы при необходимости. Обучаем с
    использованием всех процессоров по умолчанию, или заданного
    числа процессоров

    Параметры
    ---------
    X_train : numpy.ndarray, размера n_samples * n_features
        Матрица признаков для обучения модели

    X_test : numpy.ndarray, размера n_samples * n_features
        Матрица признаков для тестирования модели

    y_train : numpy.ndarray размера n_samples
        Вектор целевых значений для обучения модели

    y_test : numpy.ndarray размера n_samples
        Вектор целевых значений для тестирования модели

    time_limit : целое, необязательно
        Если задан, то ограничить общее время подбора модели
        этим значением. По умолчанию используется значение, определенное
        по умолчанию для AutoSklearnRegressor.

    time_limit_per_model : целое, необязательно
        Если задан, то ограничить время обучения каждой из моделей-
        кандидатов этим значением. По умолчанию используется значение,
        определенное по умолчанию для AutoSklearnRegressor.

    n_jobs : целое, по умолчанию -1
        Если задано, выполнять перебор моделей в заданном количестве
        потоков. По умолчанию используются все доступные процессоры.

    ensemble_size : целое, по умолчанию 1
        Размер ансамбля моделей для обучения

    preprocessor : строка
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.

    pca: int, необязателен
        Если задан (и не None), то перед запуском обучения привести
        данные к главным компонентам и оставить только это количество
        наиболее значимых компонент

    use_xgboost: bool, по умолчанию False
        Если равен True, то вместо AutoSklearnRegressor обучать модель
        XGBRegressor. Эта опция поможет тестировать изменения в предобработке
        данных быстрее, так как AutoSklearnRegressor намного медленнее

    scorer: autosklearn.metrics.Scorer, необязателен
        Используемая метрика качества обучаемых моделей. По умолчанию
        используется среднеквадратичная ошибка

    Возвращает
    ----------
    reg : AutoSklearnRegressor
        Регрессор, обученный на (X_train, y_train)
    """
    kwargs = {"n_jobs": n_jobs, "ensemble_size": 0}
    if time_limit:
        kwargs['time_left_for_this_task'] = time_limit
    if time_limit_per_model:
        kwargs['per_run_time_limit'] = time_limit_per_model
    if scorer:
        kwargs['metric'] = scorer
        kwargs["n_jobs"] = -1
    if use_xgboost:
        reg = XGBRegressor(random_state=0)
    else:
        reg = AutoSklearnRegressor(**kwargs)
    reg = pack_model(reg, preprocessor, pca)
    # Обучаем модель
    fit_model(
        reg,
        X_train,
        y_train,
        ensemble_size=ensemble_size,
        preprocessor=preprocessor
    )
    display_model_score(
        X_train, y_train, reg,
        model_name="AutoSklearn",
        data_name="обучающих данных"
    )
    display_model_score(
        X_test, y_test, reg,
        model_name="AutoSklearn",
        data_name="тестовых данных"
    )
    return reg


def get_model_to_save(reg,
                      *,
                      n_jobs=-1,
                      get_what: str = "first",
                      preprocessor: str = None,
                      pca: int = None):
    """
    Получаем итоговую модель в виде объекта производного класса от
    sklearn.BaseEstimator

    Параметры
    ---------
    reg : AutoSklearnRegressor
        Уже обученный регрессор Auto-Sklearn

    n_jobs : целое, по умолчанию -1
        Используется только если get_what установлено в voting,
        возвращая ансамблевый регрессор, обучающийся в заданное
        число потоков. По умолчанию - использующий все доступные
        процессоры

    get_what : строка
        Допускаются значения

        *   "first" - значение по умолчанию - вернуть модель с
            наибольшим весом, входящую в общий регрессор.

        *   "voting" - вернуть объект VotingRegressor, включающий
            все модели исходного ансамбля. Не работеат, пока не
            исправлен баг в auto-sklearn.

        *   "source" - просто вернуть значение reg.

    preprocessor: str, необязателен
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.

    pca: int, необязателен
        Если он задан, то считать иходную модель конвейром, в котором
        регрессор вызывается после обработки методом главных компонент
        с заданным параметром n_components. Возвращаемая модель также
        будет упакована в конвейр с указанным преобразованием в начале

    Возвращает
    ----------
    model : sklearn.BaseEstimator
        Найденная наилучшая модель scikit-learn, извлеченная из
        регрессора Auto-Sklearn
    """
    if preprocessor or pca:
        model = reg.steps[1][1]
    else:
        model = reg
    if not hasattr(model, "get_models_with_weights"):
        return pack_model(model, preprocessor, pca)
    lst = model.get_models_with_weights()
    estimators = []
    weights = []
    if get_what == "first":
        # Берем наилучшую из найденных моделей
        model = lst[0][1]
    elif get_what == "voting":
        # Или берем ансамбль
        for (i, (w, m)) in enumerate(lst):
            estimators.append(("model_{}".format(i + 1), m))
            weights.append(w)
        model = VotingRegressor(estimators, weights=weights, n_jobs=n_jobs)
    elif get_what == "source":
        pass
    else:
        raise ValueError(
            "Допустимые значения get_what - это first, voting и source, а не {}".format(get_what)
        )
    model = pack_model(model, preprocessor, pca)
    return model


def fit_regressor(data_filename: str,
                  model_filename: str,
                  *,
                  time_limit: int = None,
                  time_limit_per_model: int = None,
                  quiet: bool = False,
                  n_jobs: int = -1,
                  ensemble_size: int = 1,
                  final_model: str = "first",
                  auto_normalize: bool = False,
                  preprocessor: str = None,
                  pca: int = None,
                  verbosity: int = 0,
                  use_xgboost: bool = False,
                  metric_assymmetry: float = 1):
    """
    Обучает с помощью Auto-Sklearn регрессор на массиве данных,
    считанном из data_filename и записывает в файл model_filename.

    Параметры
    ---------
    data_filename : строка
        Имя файла, содержащего обучающие данные. Это должен быть
        CSV файл без строки заголовка. Последний столбец считается
        столбцом целевой переменной, остальные - признаками

    model_filename : строка
        Имя файла, в который будет сохранена обученная модель.
        Сохраненено будет значение типа sklearn.BaseEstimator
        или производного от него, с использованием библиотеки
        joblib.

    time_limit : целое, необязательно
        Если задан, то ограничить общее время подбора модели
        этим значением. По умолчанию используется значение, определенное
        по умолчанию для AutoSklearnRegressor.

    time_limit_per_model : целое, необязательно
        Если задан, то ограничить время обучения каждой из моделей-
        кандидатов этим значением. По умолчанию используется значение,
        определенное по умолчанию для AutoSklearnRegressor.

    quiet : логическое, по умолчанию False
        Значение True подавляет вывод отладочных сообщений в процессе
        обучения. Многие из алгоритмов, входящих в Scikit-Learn,
        выдают разные отладочные сообщения

    n_jobs : целое, по умолчанию -1
        Если задано, выполнять перебор моделей в заданном количестве
        потоков. По умолчанию используются все доступные процессоры.

    final_model : строка
        Какую модель в итоге сохраняем в качестве окончательной
        Допускаются значения

        *   "first" - значение по умолчанию - вернуть модель с
            наибольшим весом, входящую в общий регрессор.

        *   "voting" - вернуть объект VotingRegressor, включающий
            все модели исходного ансамбля. Не работает, пока не
            исправлен баг в auto-sklearn.

        *   "source" - сохранить сам объект AutoSklearnRegressor.

    preprocessor: str, необязателен
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.

    pca: int, необязателен
        Если задан (и не None), то перед запуском обучения привести
        данные к главным компонентам и оставить только это количество
        наиболее значимых компонент

    verbosity: int, необязателен
        Вывод отладочных сообщений в процессе работы программы. Чем
        больше это значение, тем больше информации будет выводиться,
        значение 0 подваляет все отладочные сообщения

    use_xgboost: bool, по умолчанию False
        Если равен True, то вместо AutoSklearnRegressor обучать модель
        XGBRegressor. Эта опция поможет тестировать изменения в предобработке
        данных быстрее, так как AutoSklearnRegressor намного медленнее

    metric_assymmetry: float, по умолчанию 1
        Если отличен от 1, то штрафовать модель за предсказания, по модулю
        большие, чем истинные значения целевой переменной сильнее (в это число
        раз) чем за предсказания, оказавшиеся ближе к нулю. Если значение
        больше 1, то мы заставим модель пытаться предсказать по возможности
        меньшие по модулю значения, а если больше - то по возможности большие.
    """
    if metric_assymmetry and use_xgboost:
        raise NotImplementedError("Ассиметричная метрика не работает с XGBoost")
    # Заглушаем предупреждения, если нужно
    if quiet:
        warnings.filterwarnings('ignore')
    # Функция для оценки результатов модели
    if metric_assymmetry:
        scorer = make_scorer(
            "assymmetric_error",
            assymmetric_mse,
            greater_is_better=False,
            optimum=0,
            needs_proba=False,
            needs_threshold=False,
            extra_argument=metric_assymmetry
        )
    else:
        scorer = None
    # Считываем данные из файла, делим на обучающие и тестовые
    (X, y) = prepare_data(data_filename, auto_normalize=auto_normalize)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    # Сообщаем пользователю, на каком объеме данных обучаем модель
    if verbosity >= 1:
        sys.stderr.write(
            "Обучение модели регрессии на {} примерах по {} признаков\n".format(
                X.shape[0], X.shape[1]
            )
        )
        start = datetime.datetime.now()
    # Обучаем регрессор AutoSklearnRegressor
    reg = fit_autosklearn_regressor(
        X_train, X_test, y_train, y_test,
        time_limit=time_limit,
        time_limit_per_model=time_limit_per_model,
        n_jobs=n_jobs,
        ensemble_size=ensemble_size,
        preprocessor=preprocessor,
        pca=pca,
        use_xgboost=use_xgboost,
        scorer=scorer
    )
    if verbosity >= 1:
        sys.stderr.write("Обучение модели регрессии AutoSklearn завершено\n")
        end = datetime.datetime.now()
        if verbosity >= 2:
            sys.stderr.write(
                "Обучение длилось {:2f} секунд\n".format(
                    (end - start).total_seconds()
                )
            )
    # Получаем итоговую модель в виде произвоного класса от sklearn.BaseEstimator
    model = get_model_to_save(
        reg,
        n_jobs=n_jobs,
        get_what=final_model,
        preprocessor=preprocessor,
        pca=pca
    )
    # Еще раз обучаем
    if preprocessor or final_model != "source":
        fit_model(
            model, X_train, y_train,
            ensemble_size=ensemble_size,
            preprocessor=preprocessor
        )
    # И показываем метрики качества модели
    display_model_score(
        X_train, y_train, model,
        model_name="итоговой модели", data_name="обучающих данных"
    )
    display_model_score(
        X_test, y_test, model,
        model_name="итоговой модели", data_name="тестовых данных"
    )
    # Теперь обучаем на всем массиве данных
    fit_model(
        model, X, y,
        ensemble_size=ensemble_size,
        preprocessor=preprocessor
    )
    # И сохраняем в файл
    joblib.dump(model, model_filename)


if __name__ == "__main__":
    parser = ArgumentParser(
        usage="{} --file=<ИМЯ_ФАЙЛА> --model=<ИМЯ_ФАЙЛА> [ДОП_ПАРАМЕТРЫ]",
        description="Обучает модель Scikit-Learn на заданном файле с данными и сохраняет в заданный файл",
        add_help=True
    )
    parser.add_argument(
        "--file", "-f",
        help="Имя входного файла с данными, ожидается CSV без строки заголовка",
        required=True,
        type=str,
        metavar="ИМЯ_ФАЙЛА",
        dest="data_file"
    )

    parser.add_argument(
        "--model", "-m",
        help="Имя файла для сохранения модели Scikit-Learn, с использованием joblib.dump",
        required=True,
        type=str,
        metavar="ИМЯ_ФАЙЛА",
        dest="model_file"
    )

    parser.add_argument(
        "--time-limit",
        help="Максимальное время обучения результирующей модели в секундах",
        type=int,
        metavar="СЕКУНДЫ",
        dest="time_limit",
        default=None
    )

    parser.add_argument(
        "--time-limit-per-model",
        help="Максимальное время обучения каждой из моделей-кандидатов в секундах",
        type=int,
        metavar="СЕКУНДЫ",
        dest="time_limit_per_model",
        default=None
    )

    parser.add_argument(
        "--quiet", "-q",
        help="Подавлять предупреждения, выводимые Scikit-Learn",
        action="store_const",
        const=True,
        default=False,
        dest="quiet"
    )

    parser.add_argument(
        "--n-jobs",
        help="Запускать выполнение в заданное количество потоков",
        action="store",
        type=int,
        default=-1,
        metavar="ЧИСЛО",
        dest="n_jobs"
    )

    parser.add_argument(
        "--ensemble-size",
        help="Построить финальный ансамбль из этого количества лучших моделей",
        action="store",
        type=int,
        default=1,
        metavar="РАЗМЕР",
        dest="ensemble_size"
    )

    parser.add_argument(
        "--final-model",
        help="Сохранить в итоге модель, полученную таким способом",
        action="store",
        type=str,
        default="first",
        metavar="first_source_or_voting",
        choices=["first", "source", "voting"],
        dest="final_model"
    )

    parser.add_argument(
        "--auto-normalize",
        help="Перед обучением модели обработать фрейм с помощью autonormalize",
        action="store_const",
        const=True,
        default=False,
        dest="auto_normalize"
    )

    parser.add_argument(
        "--preprocessor",
        help="Способ предобработки признаков",
        action="store",
        type=str,
        default=None,
        metavar="quantile_or_power",
        choices=["quantile", "power"],
        dest="preprocessor"
    )

    parser.add_argument(
        "--pca",
        help="Привести к главным компонентам и оставить наиболее значимые",
        action="store",
        type=int,
        default=None,
        metavar="N_COMPONENTS",
        dest="pca"
    )

    parser.add_argument(
        "--verbosity",
        help="Выводить на экран отладочную информацию в просессе работы",
        action="store",
        type=int,
        default=0,
        metavar="VALUE",
        dest="verbosity"
    )

    parser.add_argument(
        "--use-xgboost",
        help="Использовать XGBRegressor вместо AutoSklearnRegressor",
        action="store_const",
        const=True,
        default=False,
        dest="use_xgboost"
    )

    parser.add_argument(
        "--metric-assymmetry",
        help="Штрафовать модель за большие по модулю предсказания сильнее, чем за меньшие",
        action="store",
        type=float,
        default=None,
        metavar="FACTOR",
        dest="metric_assymmetry"
    )

    args = parser.parse_args()

    fit_regressor(
        data_filename=args.data_file,
        model_filename=args.model_file,
        time_limit=args.time_limit,
        time_limit_per_model=args.time_limit_per_model,
        quiet=args.quiet,
        n_jobs=args.n_jobs,
        ensemble_size=args.ensemble_size,
        final_model=args.final_model,
        auto_normalize=args.auto_normalize,
        preprocessor=args.preprocessor,
        pca=args.pca,
        verbosity=args.verbosity,
        use_xgboost=args.use_xgboost,
        metric_assymmetry=args.metric_assymmetry
    )

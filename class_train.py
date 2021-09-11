#! /usr/bin/env python

"""
Сценарий для обучения классификатора на основе AutoSklearn.

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

--metric МЕТРИКА
    Возможные значения - precision, recall, f1, accuracy, roc-auc
    Пытаемся максимизировать именно эту оценку качества модели.

--ensemble-size РАЗМЕР_АНСАМБЛЯ
    Размер итогового ансамбля моделей, рекомендуется 1

--final-model ТИП_МОДЕЛИ
    Определяет вид модели, которая в итоге будет сохранена в файл.
    Допускаются следующие значения:

    *   "source" - оригинальный объект AutoSklearnClassifier.
        Сценарий для выполнения прогноза при этом работает, но
        совместимость с моделями Scikit-Learn в прочих аспектах
        не гарантируется.

    *   "first" - сохраняется наилучшая модель из ансамбля. Она
        может быть несколько хуже, чем весь ансамбль

    *   "voting" - сохранить ансамбль моделей в виде объекта
        VotingClassifier. В настоящее время не работает из-за
        внутренней ошибки Auto-Sklearn. Разработчикам сообщено
        об ошибке, ждем.

--auto-normalize
    Использовать дополнительные механизмы автоматической нормализации
    данных перед обучением.

--preprocessor=power|quantile
    Перед передачай данных модели Auto-Sklearn преобразовать их
    в равномерное распределение (QuantileTransformer) или в нормальное
    (PowerTransformer)

Примечания

1.  AutoSklearnClassifier имеет временные ограничения по умолчанию.
    Если нужно обучать его неограниченно долго, нужно вручную установить
    очень большие значения для time-limit и time-limit-per-model
"""

import sys
import warnings
import joblib

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from autosklearn.classification import AutoSklearnClassifier

from autosklearn.metrics import roc_auc
from autosklearn.metrics import f1
from autosklearn.metrics import precision
from autosklearn.metrics import recall
from autosklearn.metrics import accuracy

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from autonormalize import autonormalize


def prepare_data(filename, auto_normalize: bool = False):
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
    X = X.values
    y = data['y'].astype(int).values
    return (X, y)


def get_metric_by_name(metric_name: str = None):
    """
    Возвращает объетк метрики auto-sklearn по указанному имени
    метрики.

    Параметры
    ---------
    metric_name : строка
        Строка с названием метрики, может быть "accuracy", "f1",
        "precision", "recall", "roc-auc" или None, последний
        вариант эквивалентен "accuracy".

    Возвращает
    ----------
    metric : autosklearn.Scorer
        Объект оценщика качества

    Исключения
    ----------
    ValueError :
        Строка metric_name не является ни одной из допустимых
    """
    metric_name = metric_name or "accuracy"
    metric_name = metric_name.lower()
    if metric_name in ["roc_auc", "roc-auc"]:
        return roc_auc
    elif metric_name == "f1":
        return f1
    elif metric_name == "precision":
        return precision
    elif metric_name == "recall":
        return recall
    elif metric_name == "accuracy":
        return accuracy
    else:
        raise ValueError("Метрика может быть roc_auc, f1, precision, recall, accuracy, а не {}".format(metric_name))


def fit_model(model,
              X,
              y,
              *,
              metric: str = None,
              ensemble_size: int = 1,
              preprocessor: str = None):
    """
    Обучает модель model на матрице признаков X и векторе целевых
    переменных y. Если модель имеет тип AutoSklearnClassifier, то
    выполняет отдельно обучение моделей и обучение ансамбля.

    Параметры
    ---------
    model : sklearn.BaseEstimator
        Классификатор scikit-learn или AutoSklearnClassifier,
        который нужно обучить.

    X : numpy.ndarray размера n_samples * n_features
        Матрица признаков для обучения классификатора

    y : numpy.ndarray размера n_features
        Вектор целевых значений для обучения классификатора

    metric : строка
        Метрика, которую мы хотим максимизировать при обучении.
        По умолчанию accuracy, допускается также roc_auc, f1,
        precision, recall

    ensemble_size : целое, по умолчанию 1
        Размер ансамбля при использовании AutoSklearnClassifier.
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
        if isinstance(model.steps[1][1], AutoSklearnClassifier):
            model.fit(X, y, clf__metric=get_metric_by_name(metric))
            if ensemble_size:
                model.steps[1][1].fit_ensemble(
                    y,
                    ensemble_size=ensemble_size,
                    metric=get_metric_by_name(metric)
                )
        else:
            model.fit(X, y)
    else:
        if isinstance(model, AutoSklearnClassifier):
            model.fit(X, y, metric=get_metric_by_name(metric))
            if ensemble_size:
                model.fit_ensemble(
                    y,
                    ensemble_size=ensemble_size,
                    metric=get_metric_by_name(metric)
                )
        else:
            model.fit(X, y)
    return model


def display_model_score(X,
                        y,
                        model,
                        *,
                        model_name: str = "модели",
                        data_name: str = "тестовых данных"):
    """
    Выводин на экран информацию о метриках качества модели

    Параметры
    ---------
    X : numpy.ndarray, размера n_samples * n_features
        Матрица признаков

    y : numpy.ndarray размера n_samples
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
        "Правильность {} на {} равна {}\n".format(
            model_name, data_name, accuracy_score(y, y_pred)
        )
    )
    if len(set(y)) == 2:
        # Другие оценки roc_auc считаем только для бинарной классификации
        sys.stdout.write(
            "Точность {} на {} равна {}\n".format(
                model_name, data_name, precision_score(y, y_pred)
            )
        )
        sys.stdout.write(
            "Полнота {} на {} равна {}\n".format(
                model_name, data_name, recall_score(y, y_pred)
            )
        )
        sys.stdout.write(
            "F1 оценка {} на {} равна {}\n".format(
                model_name, data_name, f1_score(y, y_pred)
            )
        )
        y_prob = model.predict_proba(X)[:, 1]
        sys.stdout.write(
            "Площадь под кривой РХП {} на {} равна {}\n".format(
                model_name, data_name, roc_auc_score(y, y_prob)
            )
        )


def pack_model(clf, preprocessor: str = None):
    """
    Упаковывает модель clf в Pipeline с препроцессором,
    определенным параметром preprocessor.

    Параметры
    ---------

    clf : sklearn.BaseEstimator
        Исходный классификатор

    preprocessor : строка
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.


    Возвращает
    ----------
    pipe : sklearn.BaseEstimator
        Комбинированный классификатор, сочетающий clf с указанным
        препроцессором, или clf, если препроцессор не указан
    """
    if not preprocessor:
        return clf
    elif preprocessor.lower() == "quantile":
        return Pipeline(
            [
                ('pre', QuantileTransformer()),
                ('clf', clf)
            ]
        )
    elif preprocessor.lower() == "power":
        return Pipeline(
            [
                ('pre', PowerTransformer()),
                ('clf', clf)
            ]
        )
    raise ValueError("Препроцессор может быть 'quantile' или 'power', а не '{}'".format(preprocessor))


def fit_autosklearn(X_train,
                    X_test,
                    y_train,
                    y_test,
                    *,
                    time_limit: int = None,
                    time_limit_per_model: int = None,
                    n_jobs: int = -1,
                    metric: str = None,
                    ensemble_size: int = 1,
                    preprocessor: str = None):
    """
    Создаем Auto-SkLearn классификатор и устанавливаем временные
    ограничения для его работы при необходимости. Обучаем с
    использованием всех процессоров по умолчанию, или заданного
    числа процессоров

    Параметры
    ---------
    X_train : numpy.ndarray, размера n_samples * n_features
        Матрица признаков для обучения классификатора

    X_test : numpy.ndarray, размера n_samples * n_features
        Матрица признаков для проверки классификатора

    y_train : numpy.ndarray размера n_samples
        Вектор целевых значений для обучения классификатора

    y_test : numpy.ndarray размера n_samples
        Вектор целевых значений для проверки классификатора

    time_limit : целое, необязательно
        Если задан, то ограничить общее время подбора классификатора
        этим значением. По умолчанию используется значение, определенное
        по умолчанию для AutoSklearnClassifier.

    time_limit_per_model : целое, необязательно
        Если задан, то ограничить время обучения каждой из моделей-
        кандидатов этим значением. По умолчанию используется значение,
        определенное по умолчанию для AutoSklearnClassifier.

    n_jobs : целое, по умолчанию -1
        Если задано, выполнять перебор моделей в заданном количестве
        потоков. По умолчанию используются все доступные процессоры.

    metric : строка, необязательна
        Указать метрику качества модели, которую следует оптимизировать
        при отборе моделей. Вместо используемой по умолчанию верности
        можно указать следующие значения:

        *   "precision" - точность
        *   "recall" - полнота
        *   "f1" - мера f1
        *   "roc-auc" - площадь под кривой РХП

    ensemble_size : целое, по умолчанию 1
        Размер ансамбля моделей для обучения

    preprocessor : строка
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.

    Возвращает
    ----------
    clf : AutoSklearnClassifier
        Классификатор, обученный на (X_train, y_train)
    """
    kwargs = {"n_jobs": n_jobs, "ensemble_size": 0}
    if time_limit:
        kwargs['time_left_for_this_task'] = time_limit
    if time_limit_per_model:
        kwargs['per_run_time_limit'] = time_limit_per_model
    clf = AutoSklearnClassifier(**kwargs)
    clf = pack_model(clf, preprocessor)
    # Обучаем классификатор
    fit_model(clf, X_train, y_train, metric=metric, ensemble_size=ensemble_size, preprocessor=preprocessor)
    display_model_score(
        X_train, y_train, clf,
        model_name="AutoSklearn", data_name="обучающих данных"
    )
    display_model_score(
        X_test, y_test, clf,
        model_name="AutoSklearn", data_name="тестовых данных"
    )
    return clf


def get_model_to_save(clf,
                      *,
                      n_jobs=-1,
                      get_what: str = "first",
                      preprocessor: str = None):
    """
    Получаем итоговую модель в виде объекта производного класса от
    sklearn.BaseEstimator

    Параметры
    ---------
    clf : AutoSklearnClassifier
        Уже обученный классификатор Auto-Sklearn

    n_jobs : целое, по умолчанию -1
        Используется только если get_what установлено в voting,
        возвращая ансамблевый классификатор, обучающийся в заданное
        число потоков. По умолчанию - использующий все доступные
        процессоры

    get_what : строка
        Допускаются значения

        *   "first" - значение по умолчанию - вернуть модель с
            наибольшим весом, входящую в общий классификатор.

        *   "voting" - вернуть объект VotingClassifier, включающий
            все модели исходного ансамбля. Не работеат, пока не
            исправлен баг в auto-sklearn.

        *   "source" - просто вернуть значение clf.

    preprocessor : строка
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.

    Возвращает
    ----------
    model : sklearn.BaseEstimator
        Найденная наилучшая модель scikit-learn, извлеченная из
        классификатора Auto-Sklearn
    """
    if preprocessor:
        model = clf.steps[1][1]
    else:
        model = clf
    lst = model.get_models_with_weights()
    estimators = []
    weights = []
    if get_what == "first":
        # Берем наилучшую из найденных моделей
        model = lst[0][1]
    elif get_what == "voting":
        # В списке lst будут пары (вес модели, объект модели). Берем только
        # первую из них, так как именно она лучшая и именно тогда результаты
        # для итогового классификатора соответствуют результатам для
        # AutoSklearnClassifier.
        for (i, (w, m)) in enumerate(lst[:1]):
            estimators.append(("model_{}".format(i + 1), m))
            weights.append(w)
        model = VotingClassifier(estimators, weights=weights, n_jobs=n_jobs, voting='soft')
    elif get_what == "source":
        # Set model to model
        pass
    else:
        raise ValueError("Допустимые значения get_what - это first, voting и source, а не {}".format(get_what))
    model = pack_model(model, preprocessor)
    return model


def fit_classifier(data_filename: str,
                   model_filename: str,
                   *,
                   time_limit: int = None,
                   time_limit_per_model: int = None,
                   quiet: bool = False,
                   n_jobs: int = -1,
                   metric: str = None,
                   ensemble_size: int = 1,
                   final_model: str = "first",
                   auto_normalize: bool = False,
                   preprocessor: str = None):
    """
    Обучает с помощью Auto-Sklearn классификатор на массиве данных,
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
        Если задан, то ограничить общее время подбора классификатора
        этим значением. По умолчанию используется значение, определенное
        по умолчанию для AutoSklearnClassifier.

    time_limit_per_model : целое, необязательно
        Если задан, то ограничить время обучения каждой из моделей-
        кандидатов этим значением. По умолчанию используется значение,
        определенное по умолчанию для AutoSklearnClassifier.

    quiet : логическое, по умолчанию False
        Значение True подавляет вывод отладочных сообщений в процессе
        обучения. Многие из алгоритмов, входящих в Scikit-Learn,
        выдают разные отладочные сообщения

    n_jobs : целое, по умолчанию -1
        Если задано, выполнять перебор моделей в заданном количестве
        потоков. По умолчанию используются все доступные процессоры.

    metric : строка, необязательна
        Указать метрику качества модели, которую следует оптимизировать
        при отборе моделей. Вместо используемой по умолчанию верности
        можно указать следующие значения:

        *   "precision" - точность
        *   "recall" - полнота
        *   "f1" - мера f1
        *   "roc-auc" - площадь под кривой РХП

    ensemble_size : целое, по умолчанию 3
        Размер итогового ансамбля моделей

    final_model : строка
        Какую модель в итоге сохраняем в качестве окончательной
        Допускаются значения

        *   "first" - значение по умолчанию - вернуть модель с
            наибольшим весом, входящую в общий классификатор.

        *   "voting" - вернуть объект VotingClassifier, включающий
            все модели исходного ансамбля. Не работает, пока не
            исправлен баг в auto-sklearn.

        *   "source" - сохранить сам объект AutoSklearnClassifier.

    preprocessor : строка
        Может быть "power", "quantile" или None. Указывает способ
        предварительной обработки признаков - sklearn.PowerTransformer,
        sklearn.QuantileTransformer или ничего.
    """
    # Заглушаем предупреждения, если нужно
    if quiet:
        warnings.filterwarnings('ignore')
    # Считываем данные из файла, без заголовка
    (X, y) = prepare_data(data_filename, auto_normalize=auto_normalize)
    if len(set(y)) > 2 and metric and metric != "accuracy":
        sys.stderr.write("Метрики, отличные от верности, годятся только для бинарной классификации\n")
        sys.exit(1)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0, stratify=y)
    # Обучаем классификатор AutoSklearnClassifier
    clf = fit_autosklearn(
        X_train, X_test, y_train, y_test,
        time_limit=time_limit,
        time_limit_per_model=time_limit_per_model,
        n_jobs=n_jobs,
        metric=metric,
        ensemble_size=ensemble_size,
        preprocessor=preprocessor
    )
    # Получаем итоговую модель в виде произвоного класса от
    # sklearn.BaseEstimator
    model = get_model_to_save(
        clf,
        n_jobs=n_jobs,
        get_what=final_model,
        preprocessor=preprocessor
    )
    # Еще раз обучаем
    if preprocessor or final_model != "source":
        fit_model(model, X_train, y_train, metric=metric, ensemble_size=ensemble_size, preprocessor=preprocessor)
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
    fit_model(model, X, y, metric=metric, ensemble_size=ensemble_size, preprocessor=preprocessor)
    # И сохраняем в файл
    joblib.dump(model, model_filename)


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
    "--metric",
    help="Оптимизировать значение именно этой метрики модели",
    action="store",
    type=str,
    default=None,
    metavar="МЕТРИКА",
    choices=["accuracy", "precision", "recall", "f1", "roc-auc"],
    dest="metric"
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

args = parser.parse_args()

fit_classifier(
    data_filename=args.data_file,
    model_filename=args.model_file,
    time_limit=args.time_limit,
    time_limit_per_model=args.time_limit_per_model,
    quiet=args.quiet,
    n_jobs=args.n_jobs,
    metric=args.metric,
    ensemble_size=args.ensemble_size,
    final_model=args.final_model,
    auto_normalize=args.auto_normalize,
    preprocessor=args.preprocessor
)

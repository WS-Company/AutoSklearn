#! /usr/bin/env python

"""
Предсказание данных регрессором scikit-learn

Параметры

--model=ИМЯ_ФЙЛА или -m ИМЯ_ФАЙЛА
    Имя файла, в который была сохранена модель (предполагается, что
    для сохранения использовался joblib и сохраненный объект имеет
    тип sklearn.BaseEstimator).

--data="ЗНАЧЕНИЕ,ЗНАЧЕНИЕ,..."
    Строка признаков, по которым нужно предсказать целевое значение.
    Числа разделяются запятыми, количество должно соответствовать
    входу модели

--quiet, -q
    Подавлять предупреждающие сообщения scikit-learn

--remove-columns СПИСОК
    Удалить из исходных данных столбцы с указанными номерами. Номера
    столбцов задаются в формате списка диапазонов или отдельных чисел -
    например, "1,3,7-9,15-20" удалить первый столбец, третий, с седьмого
    по девятый и с пятнадцатого по двадцатый включительно.

--keep-columns СПИСОК
    Удалить из исходных данных все столбцы, кроме столбцов с указанными
    номерами. Номера задаются также, как и номера столбцов для опции
    --remove-columns.
"""

import sys

from argparse import ArgumentParser

import joblib
import warnings

import numpy as np


def run_predictor(model_filename: str,
                  data,
                  quiet: bool = False):
    """
    Выдает предсказание для строки признаков data на основе
    модели, записанной в файл model_filename. Выводит на
    экран посчитанное значение

    Параметры
    ---------
    model_filename : строка
        Имя файла, содержащего модель sklearn.BaseEstimator,
        записанную в формате joblib

    data : список или массив
        Одиночная строка значений признаков, для которой выполняется
        предсказание

    quite : логическое, по умолчанию False
        Значение True подавляет отладочные сообщения scikit-learn.
        Скорее всего, их все равно не будет
    """
    # Заглушаем предупреждения, если нужно
    if quiet:
        warnings.filterwarnings('ignore')
    # Считываем модель из файла
    model = joblib.load(model_filename)
    # Рассчитываем целевые значения
    X = np.array(data).reshape(1, -1)
    y = model.predict(X)
    # И выводит их на экран
    sys.stdout.write("{:.5f}\n".format(y[0]))


if __name__ == "__main__":
    parser = ArgumentParser(
        usage="--model=ИМЯ_ФАЙЛА --data=ЗНАЧЕНИЕ,ЗНАЧЕНИЕ,...",
        description="Предсказание на основе модели Scikit-Learn",
        add_help=True
    )
    parser.add_argument(
        "--model", "-m",
        help="Имя файла в который созранена модель Scikit-Learn, с использованием joblib.dump",
        required=True,
        type=str,
        metavar="ИМЯ_ФАЙЛА",
        dest="model_file"
    )
    parser.add_argument(
        "--data", "-d",
        help="Строка с разделенными запятыми входными признаками",
        required=True,
        type=str,
        metavar="ЗНАЧЕНИЕ,ЗНАЧЕНИЕ,...",
        dest="data"
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
        "--remove-columns",
        help="Удалить из данных столбцы с указанными номерами",
        action="store",
        type=str,
        default=None,
        metavar="COLUMN_LIST",
        dest="remove_columns"
    )

    parser.add_argument(
        "--keep-columns",
        help="Удалить из данных все столбцы кроме указанных",
        action="store",
        type=str,
        default=None,
        metavar="COLUMN_LIST",
        dest="keep_columns"
    )
    args = parser.parse_args()
    if args.remove_columns or args.keep_columns:
        raise NotImplementedError(
            "Опции --remove-columns и --keep-columns не реализованы"
        )
    run_predictor(
        model_filename=args.model_file,
        data=[float(x) for x in args.data.split(",")],
        quiet=args.quiet
    )

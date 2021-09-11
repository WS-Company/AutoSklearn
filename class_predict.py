#! /usr/bin/env python

"""
Предсказание данных классификатором scikit-learn

Параметры

--model=ИМЯ_ФЙЛА или -m ИМЯ_ФАЙЛА
    Имя файла, в который была сохранена модель (предполагается, что
    для сохранения использовался joblib и сохраненный объект имеет
    тип sklearn.BaseEstimator).

--data="ЗНАЧЕНИЕ,ЗНАЧЕНИЕ,..."
    Строка признаков, по которым нужно предсказать целевой класс.
    Числа разделяются запятыми, количество должно соответствовать
    входу модели

--quiet, -q
    Подавлять предупреждающие сообщения scikit-learn
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
    экран значения вероятностей классов

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
    # Рассчитываем вероятности классов
    X = np.array(data).reshape(1, -1)
    y = model.predict_proba(X)
    # И выводит их на экран
    for (i, p) in enumerate(y.ravel()):
        sys.stdout.write("{}: {}\n".format(i, p))


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

args = parser.parse_args()

run_predictor(
    model_filename=args.model_file,
    data=[float(x) for x in args.data.split(",")],
    quiet=args.quiet
)

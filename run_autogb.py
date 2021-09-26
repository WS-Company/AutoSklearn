#! /usr/bin/env python

"""
Обучение модели на основе градиентного бустинга

Опции
-----
--data-filename, -f <ИМЯ_ФАЙЛА>
    Считать обучающие данные из этого файла. Это должен быть CSV файл без
    строки заголовка. Послений столбец считается столбцом целевых значений,
    остальные - матрицей признаков. Этот аргумент обязателен.

--model-filename, -m <ИМЯ_ФАЙЛА>
    Имя файла, в который сохраняем обученную модель. Этот аргумент обязателен.

--disable-extratrees
    Не обучать регрессор на основе случайного леса. Это поведение по умолчанию

--enable-extratrees
    Обучить также регрессор на основе случайного леса.

--disable-sklearn
    Не обучать регрессор sklearn.GradientBoostingRegressor

--disable-xgboost
    Не обучать регрессор на основе XGBoost

--disable-lgbm
    Не обучать регрессор на основе LightGBM

--disable-catboost
    Не обучать регрессор на основе CatBoost

--disable-gridsearch
    Не использовать GridSearch для подбора параметров моделей, использовать
    параметры по умолчанию или заданные вручную

--disable-refit
    Отменить обучение итоговой лучшей модели на полном наборе обучающих данных.

--verbosity <ЗНАЧЕНИЕ>
    Чем больше это значение, тем больше отладочной информации будет выводиться
    на экран.

--n-estimators <ЗНАЧЕНИЕ>
    Задать количество деревьев. При использовании перебора гиперпараметров
    значение может быть числом, либо диапазоном чисел, а без перебора - только
    числом.

--max-depth <ЗНАЧЕНИЕ>
    Задать максимальную глубину деревьев. При использовании перебора
    гиперпараметров значение может быть числом, либо диапазоном чисел,
    а без перебора - только числом.
"""

import joblib

from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

import pandas as pd

from autogb import AutoGBRegressor


def prepare_data(filename: str):
    """
    Считывает данные из файла filename, отделяет последний столбец
    как столбец целевых значений, остальное - как матрицу признаков

    Параметры
    ---------
    filename : строка
        Имя файла с данными. Это должен быть файл в CSV формате
        без заголовка

    Возвращает
    ----------
    X : numpy.ndarray, размер n_samples * n_features
        Матрица признаков, состоящая из всех считанных данных,
        кроме последнего столбца

    y : numpy.ndarray, размер n_samples
        Вектор целевых значений

    Замечания
    ---------
    Эта функция частично дублирована в файле reg_train.py
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


if __name__ == "__main__":
    parser = ArgumentParser(
        usage="{} --data-filename=<ИМЯ_ФАЙЛА> --model-filename=<ИМЯ_ФАЙЛА> [ДОП_ПАРАМЕТРЫ]",
        description="Обучает модель градиентного бустинга на заданном файле с данными и сохраняет в заданный файл",
        add_help=True
    )
    parser.add_argument(
        "--data-filename", "-f",
        help="Имя входного файла с данными, ожидается CSV без строки заголовка",
        required=True,
        type=str,
        metavar="ИМЯ_ФАЙЛА",
        dest="data_filename"
    )
    parser.add_argument(
        "--model-filename", "-m",
        help="Имя файла для сохранения модели Scikit-Learn, с использованием joblib.dump",
        required=True,
        type=str,
        metavar="ИМЯ_ФАЙЛА",
        dest="model_filename"
    )
    parser.add_argument(
        "--disable-extratrees",
        help="Не использовать модель на основе случайного леса",
        action="store_const",
        const=True,
        default=True,
        dest="disable_extratrees"
    )
    parser.add_argument(
        "--enable-extratrees",
        help="Не использовать модель на основе случайного леса",
        action="store_const",
        const=False,
        default=True,
        dest="disable_extratrees"
    )
    parser.add_argument(
        "--disable-sklearn",
        help="Не использовать модель sklearn.GradientBoostingRegressor",
        action="store_const",
        const=True,
        default=False,
        dest="disable_sklearn"
    )
    parser.add_argument(
        "--disable-xgboost",
        help="Не использовать модель xgboost.XGBRegressor",
        action="store_const",
        const=True,
        default=False,
        dest="disable_xgboost"
    )
    parser.add_argument(
        "--disable-lgbm",
        help="Не использовать модель lightgbm.LGBMRegressor",
        action="store_const",
        const=True,
        default=False,
        dest="disable_lgbm"
    )
    parser.add_argument(
        "--disable-catboost",
        help="Не использовать модель catboost.CatBoostRegressor",
        action="store_const",
        const=True,
        default=False,
        dest="disable_catboost"
    )
    parser.add_argument(
        "--disable-gridsearch",
        help="Не подбирать гиперпараметры, использовать заданные вручную или по умолчанию",
        action="store_const",
        const=True,
        default=False,
        dest="disable_gridsearch"
    )
    parser.add_argument(
        "--disable-refit",
        help="Не обучать итоговую модель на полном наборе данных. Она будет обучена примерно на 64% данных",
        action="store_const",
        const=True,
        default=False,
        dest="disable_refit"
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
        "--n-estimators",
        help="Указать число деревьев для построения леса",
        action="store",
        type=str,
        default=None,
        metavar="NUMBER_OR_RANGE",
        dest="n_estimators"
    )
    parser.add_argument(
        "--max-depth",
        help="Указать число уровней деревьев",
        action="store",
        type=str,
        default=None,
        metavar="NUMBER_OR_RANGE",
        dest="max_depth"
    )
    args = parser.parse_args()
    # Считать данные и разделить на обучающие и тестовые
    (X, y) = prepare_data(args.data_filename)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    # Создать модель AutoGB с заданными параметрами
    model = AutoGBRegressor(
        use_extratrees=not args.disable_extratrees,
        use_sklearn=not args.disable_sklearn,
        use_xgboost=not args.disable_xgboost,
        use_lgbm=not args.disable_lgbm,
        use_catboost=not args.disable_catboost,
        use_gridsearch=not args.disable_gridsearch,
        refit=not args.disable_refit,
        verbosity=args.verbosity,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    # Обучить модель на обучающих данных
    model.fit(X_train, y_train)
    # Вывести на экран качество модели на обучающих данных и те нестовых данных
    print("Модель обучена, объект модели:")
    print(str(model.best_model_))
    print("Качество на обучающих данных = {}".format(model.score(X_train, y_train)))
    print("Качество на тестовых данных = {}".format(model.score(X_test, y_test)))
    print("Лучшая модель обучается на полном наборе данных...")
    # А теперь обучить уже на всех данных и сохранить в файл
    if not args.disable_refit:
        model.best_model_.fit(X, y)
    joblib.dump(model.best_model_, args.model_filename)
    print("Модель сохранена в файл {}".format(args.model_filename))

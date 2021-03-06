"""
Автоматический градиентный бустинг
"""

import datetime

from typing import Union

from sklearn.exceptions import NotFittedError

from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class AutoGBRegressor(BaseEstimator, RegressorMixin):
    """
    Автоматический градиентный бустинг
    ----------------------------------

    Обучает ряд моделей, основанных на градиентном бустинге и случайных
    лесах и выбирает лучшую.
    """

    def __init__(self,
                 *,
                 use_gridsearch: bool = True,
                 use_sklearn: bool = True,
                 use_extratrees: bool = False,
                 use_xgboost: bool = True,
                 use_lgbm: bool = True,
                 use_catboost: bool = True,
                 n_estimators: Union[int, list] = None,
                 max_depth: Union[int, list] = None,
                 colsample_bytree: Union[float, list] = None,
                 colsample_bylevel: Union[float, list] = None,
                 subsample: Union[float, list] = None,
                 refit: bool = True,
                 random_state: int = None,
                 scoring: callable = None,
                 verbosity: int = 9,
                 n_jobs: int = 1):
        """
        Инициализация автоматического градиентного бустинга

        Параметры
        ---------
        use_gridsearch: bool, по умолчанию True
            Использовать GridSearchCV для подбора гиперпараметров всех
            обучаемых моделей

        use_sklearn: bool, по умолчанию True
            Кроме прочих, попробовать обучить модель
            sklearn.ensemble.GradientBoostingRegressor

        use_extratrees: bool, по умолчанию False
            Помимо градиентного бустинга, попробовать обучить модель на
            основе случайного леса sklearn.ensemble.ExtraTreesRegressor

        use_xgboost: bool, по умолчанию True
            Кроме прочих, обучить модель xgboost.XGBRegressor

        use_lgbm: bool, по умолчанию True
            Кроме прочих, обучить модель lightgbm.LGBMRegressor

        use_catboost: bool, по умолчанию True
            Кроме прочих, обучить модель catboost.CatBoostRegressor

        n_estimators: int, list или str, необязателен
            Количество деревьев, которые будут построены при обучении модели.
            В случае, если параметр use_gridsearch равен True, должен быть
            списком и по умолчанию устанавливается в [25, 50, 100, 200].
            Если GridSearchCV не используется, должен быть числом и по
            умолчанию равен 100. В случае строки должен быть строкой,
            содержащей число или список чисел, разделенных запятыми.

        max_depth: int, list или str, необязателен
            Максимальная глубина деревьев, которые будут построены при обучении
            модели
        """
        # Запоминаем параметры
        self.use_gridsearch = use_gridsearch
        self.use_sklearn = use_sklearn
        self.use_extratrees = use_extratrees
        self.use_xgboost = use_xgboost
        self.use_lgbm = use_lgbm
        self.use_catboost = use_catboost
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.subsample = subsample
        self.random_state = random_state
        # Если не задана мера качества модели - используем коэффициент
        # детерминации R2
        self.scoring = scoring or r2_score
        self.refit = refit
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        # Лучшая из найденных моделей - будет заполнена при обучении
        self.best_model_ = None
        # Оценка лучшей из найденных моделей на тестовых данных - будет
        # заполнена при обучении
        self.best_score_ = None

    def fit(self, X, y, *, sample_weight=None):
        """
        Обучает модели и выбирает лучшую
        """
        # Сообщить пользователю, сколько у нас примеров и сколько признаков
        if self.verbosity >= 1:
            print(
                "Training AutoGB model on a set of {} samples and {} features".format(
                    X.shape[0], X.shape[1]
                )
            )
        # Количество деревьев. Если используем GridSearch, то задаем список
        # значений-кандидатов, если нет, то значение по умолчанию - 100
        if self.n_estimators is None:
            if self.use_gridsearch:
                self.n_estimators_ = [25, 50, 100, 200]
            else:
                self.n_estimators_ = 100
        else:
            self.n_estimators_ = self._str2list(self.n_estimators)
        # Максимальная глубина деревьев. Если используем GridSearch, то
        # по умолчанию перебираем значения 3, 4 и 5, в противном случае
        # используем значение 3.
        if self.max_depth is None:
            if self.use_gridsearch:
                self.max_depth_ = [3, 4, 5]
            else:
                self.max_depth_ = 3
        else:
            self.max_depth_ = self._str2list(self.max_depth)
        # Параметр colsample_bytree, используемый рядом моделей
        if self.colsample_bytree is None:
            if self.use_gridsearch:
                self.colsample_bytree_ = [0.25, 0.5, 0.75, 1.0]
            else:
                self.colsample_bytree_ = 1.0
        else:
            self.colsample_bytree_ = self._str2list(self.colsample_bytree)
        # Параметр colsample_bylevel, используемый рядом моделей
        if self.colsample_bylevel is None:
            if self.use_gridsearch:
                self.colsample_bylevel_ = [0.25, 0.5, 0.75, 1.0]
            else:
                self.colsample_bylevel_ = 1.0
        else:
            self.colsample_bylevel_ = self._str2list(self.colsample_bylevel)
        # Параметр subsamble
        if self.subsample is None:
            if self.use_gridsearch:
                self.subsample_ = [0.25, 0.5, 0.75, 1.0]
            else:
                self.subsample_ = 1.0
        else:
            self.subsample_ = self._str2list(self.subsample)
        if sample_weight is None:
            # Разделяем данные на обучающие и тестовые
            (X_train, X_test, y_train, y_test) = train_test_split(
                X, y, random_state=self.random_state
            )
            sample_weight_train = None
            sample_weight_test = None
        else:
            # Разделяем данные на обучающие и тестовые, сохранив при этом
            # веса обучающих примеров, если они заданы
            (
                X_train, X_test, y_train, y_test,
                sample_weight_train, sample_weight_test
            ) = train_test_split(
                X, y, sample_weight, random_state=self.random_state
            )
        if self.use_sklearn:
            # Обучаем регрессор sklearn.ensamble.GradientBoostingRegressor
            start = datetime.datetime.now()
            if self.use_gridsearch:
                model = GridSearchCV(
                    GradientBoostingRegressor(
                        random_state=self.random_state,
                        verbose=max(self.verbosity - 2, 0)
                    ),
                    param_grid={
                        'max_depth': self.max_depth_,
                        'n_estimators': self.n_estimators_,
                    },
                    scoring=make_scorer(self.scoring),
                    verbose=max(self.verbosity - 1, 0),
                    n_jobs=self.n_jobs
                )
            else:
                model = GradientBoostingRegressor(
                    random_state=self.random_state,
                    n_estimators=self.n_estimators_,
                    max_depth=self.max_depth_,
                    verbose=max(self.verbosity - 2, 0)
                )
            model.fit(X_train, y_train, sample_weight=sample_weight_train)
            if self.use_gridsearch:
                model = model.best_estimator_
            (score_train, score_test) = (
                self.scoring(y_train, model.predict(X_train)),
                self.scoring(y_test, model.predict(X_test))
            )
            end = datetime.datetime.now()
            if self.verbosity >= 1:
                print(
                    "sklearn.GradientBoostingRegressor score was {} on train, {} on test".format(
                        score_train, score_test
                    )
                )
            if self.verbosity >= 2:
                print(
                    "Fitting sklearn.GradientBoostingRegressor took {} seconds".format(
                        (end - start).total_seconds()
                    )
                )
            if self.best_score_ is None or score_test > self.best_score_:
                self.best_score_ = score_test
                self.best_model_ = model
        if self.use_extratrees:
            try:
                start = datetime.datetime.now()
                if self.use_gridsearch:
                    model = GridSearchCV(
                        ExtraTreesRegressor(
                            random_state=self.random_state,
                            verbose=self.verbosity - 2
                        ),
                        param_grid={
                            'max_depth': self.max_depth_,
                            'n_estimators': self.n_estimators_
                        },
                        scoring=make_scorer(self.scoring),
                        verbose=max(self.verbosity - 1, 0),
                        n_jobs=self.n_jobs
                    )
                else:
                    model = ExtraTreesRegressor(
                        random_state=self.random_state,
                        n_estimators=self.n_estimators_,
                        max_depth=self.max_depth_,
                        verbose=max(self.verbosity - 2, 0)
                    )
                model.fit(X_train, y_train, sample_weight=sample_weight_train)
                if self.use_gridsearch:
                    model = model.best_estimator_
                (score_train, score_test) = (
                    self.scoring(y_train, model.predict(X_train)),
                    self.scoring(y_test, model.predict(X_test))
                )
                if self.verbosity >= 1:
                    print(
                        "sklearn.ExtraTreesRegressor score was {} on train, {} on test".format(
                            score_train, score_test
                        )
                    )
                end = datetime.datetime.now()
                if self.verbosity >= 2:
                    print(
                        "Fitting sklearn.ExtraTreesRegressor took {} seconds".format(
                            (end - start).total_seconds()
                        )
                    )
                if self.best_score_ is None or score_test > self.best_score_:
                    self.best_score_ = score_test
                    self.best_model_ = model
            except Exception:
                # Иногда в процессе обучения случайного леса происходят
                # ошибки, пока не знаю с чем они связаны, так что просто
                # игнорируем эту модель и используем другие
                print("sklearn.ExtraTreesRegressor did not fit, continuing anyway")
        if self.use_xgboost:
            start = datetime.datetime.now()
            if self.use_gridsearch:
                model = GridSearchCV(
                    XGBRegressor(
                        random_state=self.random_state
                    ),
                    param_grid={
                        'max_depth': self.max_depth_,
                        'n_estimators': self.n_estimators_,
                        'colsample_bytree': self.colsample_bytree_,
                        'colsample_bylevel': self.colsample_bylevel_
                    },
                    scoring=make_scorer(self.scoring),
                    verbose=max(self.verbosity - 1, 0),
                    n_jobs=self.n_jobs
                )
            else:
                model = XGBRegressor(
                    random_state=self.random_state,
                    n_estimators=self.n_estimators_,
                    colsample_bytree=self.colsample_bytree_,
                    colsample_bylevel=self.colsample_bylevel_,
                    max_depth=self.max_depth_
                )
            model.fit(X_train, y_train, sample_weight=sample_weight_train)
            if self.use_gridsearch:
                model = model.best_estimator_
            (score_train, score_test) = (
                self.scoring(y_train, model.predict(X_train)),
                self.scoring(y_test, model.predict(X_test))
            )
            end = datetime.datetime.now()
            if self.verbosity >= 1:
                print(
                    "xgboost.XGBRegressor score was {} on train, {} on test".format(
                        score_train, score_test
                    )
                )
            if self.verbosity >= 2:
                print(
                    "Fitting xgboost.XGBRegressor took {} seconds".format(
                        (end - start).total_seconds()
                    )
                )
            if self.best_score_ is None or score_test > self.best_score_:
                self.best_score_ = score_test
                self.best_model_ = model
        if self.use_lgbm:
            start = datetime.datetime.now()
            if self.use_gridsearch:
                model = GridSearchCV(
                    LGBMRegressor(
                        random_state=self.random_state,
                        verbose=-1
                    ),
                    param_grid={
                        'max_depth': self.max_depth_,
                        'n_estimators': self.n_estimators_,
                        'subsample': self.subsample_,
                        'colsample_bytree': self.colsample_bytree_
                    },
                    scoring=make_scorer(self.scoring),
                    verbose=max(self.verbosity - 1, 0),
                    n_jobs=self.n_jobs
                )
            else:
                model = LGBMRegressor(
                    random_state=self.random_state,
                    n_estimators=self.n_estimators_,
                    max_depth=self.max_depth_,
                    subsample=self.subsample_,
                    colsample_bytree=self.colsample_bytree_,
                    verbose=-1
                )
            model.fit(X_train, y_train, sample_weight=sample_weight_train)
            if self.use_gridsearch:
                model = model.best_estimator_
            (score_train, score_test) = (
                self.scoring(y_train, model.predict(X_train)),
                self.scoring(y_test, model.predict(X_test))
            )
            end = datetime.datetime.now()
            if self.verbosity >= 1:
                print(
                    "lightgbm.LGBMRegressor score was {} on train, {} on test".format(
                        score_train, score_test
                    )
                )
            if self.verbosity >= 2:
                print(
                    "Fitting lightgbm.LGBMRegressor took {} seconds".format(
                        (end - start).total_seconds()
                    )
                )
            if self.best_score_ is None or score_test > self.best_score_:
                self.best_score_ = score_test
                self.best_model_ = model
        if self.use_catboost:
            start = datetime.datetime.now()
            if self.use_gridsearch:
                model = GridSearchCV(
                    CatBoostRegressor(
                        random_state=self.random_state,
                        verbose=max(self.verbosity - 2, 0)
                    ),
                    param_grid={
                        'max_depth': self.max_depth_,
                        'n_estimators': self.n_estimators_,
                        'colsample_bylevel': self.colsample_bylevel_
                    },
                    scoring=make_scorer(self.scoring),
                    verbose=max(self.verbosity - 1, 0),
                    n_jobs=self.n_jobs
                )
            else:
                model = CatBoostRegressor(
                    random_state=self.random_state,
                    n_estimators=self.n_estimators_,
                    max_depth=self.max_depth_,
                    colsample_bylevel=self.colsample_bylevel_,
                    verbose=max(self.verbosity - 2, 0)
                )
            model.fit(X_train, y_train, sample_weight=sample_weight_train)
            if self.use_gridsearch:
                model = model.best_estimator_
            (score_train, score_test) = (
                self.scoring(y_train, model.predict(X_train)),
                self.scoring(y_test, model.predict(X_test))
            )
            end = datetime.datetime.now()
            if self.verbosity >= 1:
                print(
                    "catboost.CatBoostRegressor score was {} on train, {} on test".format(
                        score_train, score_test
                    )
                )
            if self.verbosity >= 2:
                print(
                    "Fitting catboost.CatBoostRegressor took {} seconds".format(
                        (end - start).total_seconds()
                    )
                )
            if self.best_score_ is None or score_test > self.best_score_:
                self.best_score_ = score_test
                self.best_model_ = model
        if self.refit and self.best_model_ is not None:
            self.best_model_.fit(X, y)

    def predict(self, X):
        """
        Вычисляет прогнозируемые значения целевой переменной для матрицы
        признаков X
        """
        if self.best_model_:
            return self.best_model_.predict(X)
        raise NotFitterError("AutoGBRegressor not fitted yet")

    def score(self, X, y, *, sample_weight=None):
        """
        Вычисляет качество предсказания модели для матрицы признаков X и
        столбца значений целевой переменной y. Используется метрика качества
        модели self.scoring, по умолчанию соответствующая коэффициенту
        детерминации R2.
        """
        return self.scoring(y, self.predict(X), sample_weight=sample_weight)

    @staticmethod
    def _str2list(s):
        """
        Преобразует строку s, хранящую число, диапазон, список чисел или
        список диапазонов, в число или список чисел. При этом в списке не
        должно быть отрицательных чисел, а диапазоны допускаются только
        между целыми числами.

        Примеры
        -------
        AutoGBRegressor._str2list("20") = 20
        AutoGBRegressor._str2list("3.5") = 3.5
        AutoGBRegressor._str2list("1,2,4") = [1, 2, 4]
        AutoGBRegressor._str2list("1,2,4-7,10") = [1, 2, 4, 5, 6, 7, 10]
        """
        # Если s - не строка, а число или список, то возвращаем как есть
        if not isinstance(s, str):
            return s
        # Если строка - игнорируем пробелы в начале и конце
        s = s.strip()
        try:
            # Если строке содержит одно число - вернуть его
            try:
                return int(s)
            except ValueError:
                pass
            return float(s)
        except ValueError:
            # Разделяем строку на фрагменты, разделенные запятыми
            chunks = s.split(",")
            result = []
            for chk in chunks:
                if "-" in chk:
                    (a, b) = [int(chk.split("-")[0]), int(chk.split("-")[1])]
                    result += list(range(a, b + 1))
                else:
                    result.append(float(chk))
            return result

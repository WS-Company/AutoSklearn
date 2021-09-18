"""
Тесты для дополнительно введенных метрик качества моделей
"""

from unittest import TestCase

import numpy as np

from .metrics import assymmetric_mse


class TestSymetricMse(TestCase):
    """
    Вызов get_assymmetric_mse(factor=1) возвращает обычную функцию
    среднеквадратичной ошибки
    """

    def mse(self, y_true, y_pred):
        return assymmetric_mse(y_true, y_pred, extra_argument=1)

    def test_zero(self):
        self.assertEqual(
            self.mse(np.zeros(20), np.zeros(20)),
            0
        )
        self.assertEqual(
            self.mse(np.ones(20), np.ones(20)),
            0
        )

    def test_plus_one(self):
        self.assertEqual(
            assymmetric_mse(np.ones(5) * 3, np.ones(5) * 2, extra_argument=1),
            1
        )

    def test_minus_one(self):
        self.assertEqual(
            assymmetric_mse(np.ones(5) * 2, np.ones(5) * 3, extra_argument=1),
            1
        )


class TestDouble(TestCase):
    """
    Вызов get_assymetric_mse(factor=2) возвращает функцию, являющуюся
    обычной функцией среднеквадратичной ошибки, если предсказанное
    значение меньше истинного и удвоенной функцией среднеквадратичной
    ошибки, если больше
    """

    def test_zero(self):
        """Нулевая ошибка остается нулевой"""
        self.assertEqual(assymmetric_mse(np.zeros(20), np.zeros(20), extra_argument=2), 0)
        self.assertEqual(assymmetric_mse(np.ones(20), np.ones(20), extra_argument=2), 0)

    def test_minus_one(self):
        """Если y_true - y_pred = [1, 1, ... 1], то ошибка равна 1"""
        self.assertEqual(assymmetric_mse(np.ones(5) * 3, np.ones(5) * 2, extra_argument=2), 1)

    def test_plus_one(self):
        """Если y_pred - y_true = [1, 1, ... 1], то ошибка равна 4"""
        self.assertEqual(assymmetric_mse(np.ones(5) * 2, np.ones(5) * 3, extra_argument=2), 4)

    def test_negative_minus_one(self):
        """Если y_true - y_pred = [1, 1, ... 1], то ошибка равна 4"""
        self.assertEqual(assymmetric_mse(np.ones(5) * -2, np.ones(5) * -3, extra_argument=2), 4)

    def test_negative_plus_one(self):
        """Если y_pred - y_true = [1, 1, ... 1], то ошибка равна 1"""
        self.assertEqual(assymmetric_mse(np.ones(5) * -3, np.ones(5) * -2, extra_argument=2), 1)

    def test_minus_one_root(self):
        """Если y_true - y_pred = [1, 1, ... 1], то корень из ошибки равен 1"""
        self.assertEqual(assymmetric_mse(np.ones(5) * 3, np.ones(5) * 2, squared=False, extra_argument=2), 1)

    def test_plus_one_root(self):
        """Если y_pred - y_true = [1, 1, ... 1], то корень из ошибки равен 2"""
        self.assertEqual(assymmetric_mse(np.ones(5) * 2, np.ones(5) * 3, squared=False, extra_argument=2), 2)

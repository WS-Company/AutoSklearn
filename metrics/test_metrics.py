"""
Тесты для дополнительно введенных метрик качества моделей
"""

from unittest import TestCase

import numpy as np

from .metrics import get_assymmetric_mse


class TestSymetricMse(TestCase):
    """
    Вызов get_assymmetric_mse(factor=1) возвращает обычную функцию
    среднеквадратичной ошибки
    """

    def setUp(self):
        self.mse = get_assymmetric_mse(1)

    def test_zero(self):
        """Нулевая ошибка остается нулевой"""
        self.assertEqual(self.mse(np.zeros(20), np.zeros(20)), 0)
        self.assertEqual(self.mse(np.ones(20), np.ones(20)), 0)

    def test_plus_one(self):
        """Если y_pred - y_true = [1, 1, ... 1], то ошибка равна 1"""
        self.assertEqual(self.mse(np.ones(5) * 3, np.ones(5) * 2), 1)

    def test_minus_one(self):
        """Если y_true - y_pred = [1, 1, ... 1], то ошибка равна 1"""
        self.assertEqual(self.mse(np.ones(5) * 2, np.ones(5) * 3), 1)


class TestDouble(TestCase):
    """
    Вызов get_assymetric_mse(factor=2) возвращает функцию, являющуюся
    обычной функцией среднеквадратичной ошибки, если предсказанное
    значение меньше истинного и удвоенной функцией среднеквадратичной
    ошибки, если больше
    """

    def setUp(self):
        self.mse = get_assymmetric_mse(2)

    def test_zero(self):
        """Нулевая ошибка остается нулевой"""
        self.assertEqual(self.mse(np.zeros(20), np.zeros(20)), 0)
        self.assertEqual(self.mse(np.ones(20), np.ones(20)), 0)

    def test_minus_one(self):
        """Если y_true - y_pred = [1, 1, ... 1], то ошибка равна 1"""
        self.assertEqual(self.mse(np.ones(5) * 3, np.ones(5) * 2), 1)

    def test_plus_one(self):
        """Если y_pred - y_true = [1, 1, ... 1], то ошибка равна 4"""
        self.assertEqual(self.mse(np.ones(5) * 2, np.ones(5) * 3), 4)

    def test_negative_minus_one(self):
        """Если y_true - y_pred = [1, 1, ... 1], то ошибка равна 4"""
        self.assertEqual(self.mse(np.ones(5) * -2, np.ones(5) * -3), 4)

    def test_negative_plus_one(self):
        """Если y_pred - y_true = [1, 1, ... 1], то ошибка равна 1"""
        self.assertEqual(self.mse(np.ones(5) * -3, np.ones(5) * -2), 1)

    def test_minus_one_root(self):
        """Если y_true - y_pred = [1, 1, ... 1], то корень из ошибки равен 1"""
        self.assertEqual(self.mse(np.ones(5) * 3, np.ones(5) * 2, squared=False), 1)

    def test_plus_one_root(self):
        """Если y_pred - y_true = [1, 1, ... 1], то корень из ошибки равен 2"""
        self.assertEqual(self.mse(np.ones(5) * 2, np.ones(5) * 3, squared=False), 2)

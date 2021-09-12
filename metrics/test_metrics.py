"""
Тесты для дополнительно введенных метрик качества моделей
"""

from unittest import TestCase

import numpy as np

from .metrics import get_assymetric_mse


class TestSymetricMse(TestCase):
    """
    Вызов get_assymmetric_mse(factor=1) возвращает обычную функцию
    среднеквадратичной ошибки
    """

    def setUp(self):
        self.mse = get_assymetric_mse(1)

    def test_zero(self):
        """Нулевая ошибка остается нулевой"""
        self.assertEqual(self.mse(np.zeros(20), np.zeros(20)), 0)
        self.assertEqual(self.mse(np.ones(20), np.ones(20)), 0)

    def test_one(self):
        """Если y_pred - y_true = [1, 1, ... 1], то mse = 1"""
        self.assertEqual(self.mse(np.ones(5) * 2, np.ones(5) * 3), 1)

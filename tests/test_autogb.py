from unittest import TestCase

from autogb.regression import AutoGBRegressor


class TestStr2List(TestCase):
    """
    Тесты для функции AutoGBRegressor._str2list
    """

    def test_integer(self):
        self.assertEqual(AutoGBRegressor._str2list(20), 20)

    def test_float(self):
        self.assertEqual(AutoGBRegressor._str2list(1.5), 1.5)

    def test_list(self):
        self.assertEqual(
            AutoGBRegressor._str2list([1, 2.3, 2.8, 5]),
            [1, 2.3, 2.8, 5]
        )

    def test_single_value(self):
        self.assertEqual(AutoGBRegressor._str2list("20"), 20)
        self.assertEqual(AutoGBRegressor._str2list("1.5"), 1.5)

    def test_range(self):
        self.assertEqual(
            AutoGBRegressor._str2list("25-30"),
            [25, 26, 27, 28, 29, 30]
        )

    def test_valuelist(self):
        self.assertEqual(
            AutoGBRegressor._str2list("1,2.3,2.8,5"),
            [1, 2.3, 2.8, 5]
        )

    def test_rangelist(self):
        self.assertEqual(
            AutoGBRegressor._str2list("1,2.3,2.8,5-7,12-15"),
            [1, 2.3, 2.8, 5, 6, 7, 12, 13, 14, 15]
        )

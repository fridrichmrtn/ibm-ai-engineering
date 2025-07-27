import unittest
from stats import mean, median, mode, variance

class TestStatsOperations(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)
        self.assertEqual(mean([-1, 0, 1]), 0)
        self.assertEqual(mean([1.5, 2.5, 3.5]), 2.5)
        with self.assertRaises(ValueError):
            mean([])

    def test_median(self):
        self.assertEqual(median([1, 2, 3]), 2)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertEqual(median([3, 1, 2]), 2)
        with self.assertRaises(ValueError):
            median([])

    def test_mode(self):
        self.assertEqual(mode([1, 2, 2, 3]), [2])
        self.assertEqual(mode([1, 1, 2, 3]), [1])
        self.assertEqual(mode([1, 2, 3]), None)  # No mode
        with self.assertRaises(ValueError):
            mode([])

    def test_variance(self):
        self.assertAlmostEqual(variance([1, 2, 3]), 0.6666666666666666)
        self.assertAlmostEqual(variance([1, 1, 1]), 0)
        with self.assertRaises(ValueError):
            variance([])

if __name__ == '__main__':
    unittest.main()
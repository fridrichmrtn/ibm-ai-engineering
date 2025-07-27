import unittest
from geometry import area_of_circle, perimeter_of_circle, area_of_rectangle, perimeter_of_rectangle, area_of_triangle, perimeter_of_triangle

class TestGeometryOperations(unittest.TestCase):
    def test_area_of_circle(self):
        self.assertAlmostEqual(area_of_circle(1), 3.141592653589793)
        self.assertAlmostEqual(area_of_circle(0), 0)
        self.assertAlmostEqual(area_of_circle(2.5), 19.634954084936208)

    def test_perimeter_of_circle(self):
        self.assertAlmostEqual(perimeter_of_circle(1), 6.283185307179586)
        self.assertAlmostEqual(perimeter_of_circle(0), 0)
        self.assertAlmostEqual(perimeter_of_circle(2.5), 15.707963267949466)

    def test_area_of_rectangle(self):
        self.assertEqual(area_of_rectangle(2, 3), 6)
        self.assertEqual(area_of_rectangle(0, 5), 0)
        self.assertEqual(area_of_rectangle(4.5, 2.5), 11.25)

    def test_perimeter_of_rectangle(self):
        self.assertEqual(perimeter_of_rectangle(2, 3), 10)
        self.assertEqual(perimeter_of_rectangle(0, 5), 10)
        self.assertEqual(perimeter_of_rectangle(4.5, 2.5), 14)

    def test_area_of_triangle(self):
        self.assertEqual(area_of_triangle(3, 4), 6)
        self.assertEqual(area_of_triangle(0, 5), 0)
        self.assertEqual(area_of_triangle(5, 10), 25)

    def test_perimeter_of_triangle(self):
        self.assertEqual(perimeter_of_triangle(3, 4, 5), 12)
        self.assertEqual(perimeter_of_triangle(1, 1, 1), 3)
        self.assertEqual(perimeter_of_triangle(2.5, 3.5, 4.5), 10.5)

if __name__ == '__main__':
    unittest.main()        
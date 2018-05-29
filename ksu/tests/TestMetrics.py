import unittest
import sys

from ksu.Metrics import earthMoverDistance

class TestEarthMoverDistance(unittest.TestCase):

    @staticmethod
    def suite():
        test_suite = unittest.TestSuite()
        test_suite.addTest(unittest.makeSuite(TestEarthMoverDistance))
        return test_suite

    def test_no_weights(self):
        u = [0, 1, 3]
        v = [5, 6, 8]
        d = earthMoverDistance(u, v)
        self.assertEqual(d, 5.0)

    def test_with_weights_same_domain(self):
        u = [0, 1]
        v = [0, 1]
        u_weights = [3, 1]
        v_weights = [2, 2]
        d = earthMoverDistance(u, v, u_weights, v_weights)
        self.assertEqual(d, 0.25)

    def test_with_weights_different_domains(self):
        u = [3.4, 3.9, 7.5, 7.8]
        v = [4.5, 1.4]
        u_weights = [1.4, 0.9, 3.1, 7.2]
        v_weights = [3.2, 3.5]
        d = earthMoverDistance(u, v, u_weights, v_weights)
        self.assertEqual(d, 4.078133143804786)

if __name__ == "__main__":
    # So you can run tests from this module individually.
    sys.exit(unittest.main())

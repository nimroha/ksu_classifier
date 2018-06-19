import unittest
import sys
import pytest
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances

from ksu.Utils import computeLabels, optimizedComputeLabels


class TestComputeGram(unittest.TestCase):

    @staticmethod
    def suite():
        test_suite = unittest.TestSuite()
        test_suite.addTest(unittest.makeSuite(TestComputeGram))
        return test_suite

    def empty_elements(self):
        self.fail()

    def empty_metric(self):
        self.fail()

    def first(self):
        self.fail()

xs = np.array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1],
               [2, 1],
               [3, 1],
               [2, 2],
               [3, 2]])
ys = np.array([2, 0, 1, 2, 0, 1, 0, 2])

slice = [1, 6]
subXs = xs[slice]

def testComputeLabels():
    assert computeLabels(subXs, xs, ys, 'l2') == [2, 0]

def testOptComputeLabels():
    gram = pairwise_distances(xs, metric='l2')
    assert optimizedComputeLabels(subXs, slice, xs, ys, gram) == [2, 0]


if __name__ == "__main__":
    # So you can run tests from this module individually.
    sys.exit(unittest.main())

import unittest
import sys


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

if __name__ == "__main__":
    # So you can run tests from this module individually.
    sys.exit(unittest.main())

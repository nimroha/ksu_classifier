from unittest import TextTestRunner

from TestMetrics import TestEarthMoverDistance

if __name__ == "__main__":
    """
    Add unit test class names to testCases
    """
    testCases = [TestEarthMoverDistance]
    runner=TextTestRunner()

    for testCase in testCases:
        suite = testCase.suite()
        runner.run(suite)

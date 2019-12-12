import sys
import argparse
import logging
import numpy as np

from sklearn.utils     import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from time              import time
from math              import floor

from ksu.KSU import KSU
from ksu.KSU import METRICS

CLASS_COLUMN  = 54
TEST_RATIO    = 0.2
MAX_DATA_SIZE = 100000
STRIDE        = 200

def main(argv=None):

    parser = argparse.ArgumentParser(description='Run a comparison between 1-NN and KSU on CoverType Dataset')
    parser.add_argument('--data_in',    help='Path to input data file (in .npz format with 2 nodes named X and Y)',          required=True)
    parser.add_argument('--metric',     help='Metric to use',                                                                default='l2', choices=METRICS.keys())
    parser.add_argument('--stride',     help='How many gammas to skip at a time (similar gammas will produce similar nets)', default=STRIDE, type=int)
    parser.add_argument('--delta',      help='Required confidence level',                                                    default=0.05, type=float)
    parser.add_argument('--test_ratio', help='Percent of points to holdout for the test set',                                default=TEST_RATIO, type=float)
    parser.add_argument('--max_points', help='Maximum number of data points to consider',                                    default=MAX_DATA_SIZE, type=int)
    parser.add_argument('--mode',       help='which constuction mode.\n'
                                             '"G" for greedy (faster, but bigger net), "H" for hierarchical',                default="G", choices=['G', 'H'])
    parser.add_argument('--num_jobs',   help='Number of processes to use for computation (scipy semantics)',                 default=1, type=int)
    parser.add_argument('--log_level',  help='Logging level',                                                                default=logging.CRITICAL)

    args = parser.parse_args(argv)

    dataPath  = args.data_in
    metric    = args.metric
    delta     = args.delta
    mode      = args.mode
    maxPoints = args.max_points
    testRatio = args.test_ratio
    stride    = args.stride
    logLevel  = args.log_level
    numJobs  = args.num_jobs

    start = time()
    data = np.loadtxt(dataPath, dtype=np.int32, delimiter=',')
    print("Data read time: {:.3f}s".format(time() - start))

    X = data[:, :CLASS_COLUMN]
    Y = data[:,  CLASS_COLUMN]

    X, Y = shuffle(X, Y)

    X = X[:maxPoints,:]
    Y = Y[:maxPoints]

    testSize = floor(float(X.shape[0]) * testRatio)

    testX,  testY  = X[:testSize, :], Y[:testSize]
    trainX, trainY = X[testSize:, :], Y[testSize:]

    scaler = MinMaxScaler()

    trainX = scaler.fit_transform(trainX) # transform all columns to [0,1] range and save transformation parameters
    testX  = scaler.transform(testX)      # apply the transformation to the test set (with the train set parameters)
    print('Train set size: {}\ntest set size: {}'.format(trainX.shape[0], testX.shape[0]))

    # full set:
    print('Full set:')
    startAll = time()
    h = KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=numJobs)
    start = time()
    h.fit(trainX, trainY)
    print('Fit time: {:.3f}s'.format(time() - start))

    start = time()
    predictedY = h.predict(testX)
    print('Predict time: {:.3f}s'.format(time() - start))

    error = np.mean(predictedY != testY)
    print('Error: {:.5f} , total runtime: {:.3f}s'.format(error, time() - startAll))

    # KSU
    startAll = start = time()
    ksu = KSU(trainX, trainY, metric, logLevel=logLevel, n_jobs=numJobs)
    print("Init time: {:.3f}s".format(time() - start))

    start = time()
    ksu.compressData(delta=delta, stride=stride, greedy=mode == 'G', maxCompress=0.5)
    print('Compress time: {:.3f}s'.format(time() - start))

    # stats for compressed set
    print('Compressed:')
    start = time()
    ksuClassifier = ksu.getClassifier()
    print('Fit time: {:.3f}s'.format(time() - start))

    start = time()
    predictedY = ksuClassifier.predict(testX)
    print('Predict time: {:.3f}s'.format(time() - start))

    error = np.mean(predictedY != testY)
    print('Error: {:.5f} , total runtime: {:.3f}s'.format(error, time() - startAll))


if __name__ == '__main__' :
    exit(main())


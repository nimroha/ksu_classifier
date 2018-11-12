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

CLASS_COLUMN = 54
TEST_RATIO   = 0.2

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='Run a comparison between 1-NN and KSU on CoverType Dataset')
    parser.add_argument('--data_in',   help='Path to input data file (in .npz format with 2 nodes named X and Y)',          required=True)
    parser.add_argument('--metric',    help='Metric to use',                                                                default='l2', choices=METRICS.keys())
    parser.add_argument('--stride',    help='How many gammas to skip at a time (similar gammas will produce similar nets)', default=200, type=int)
    parser.add_argument('--delta',     help='Required confidence level',                                                    default=0.05, type=float)
    parser.add_argument('--mode',      help='which constuction mode.\n'
                                            '"G" for greedy (faster, but bigger net), "H" for hierarchical',                default="G", choices=['G', 'H'])
    parser.add_argument('--num_procs', help='Number of processes to use for computation',                                   default=1, type=int)
    parser.add_argument('--log_level', help='Logging level',                                                                default=logging.CRITICAL)

    args = parser.parse_args()

    dataPath = args.data_in
    metric   = args.metric
    delta    = args.delta
    mode     = args.mode
    stride   = args.stride
    logLevel = args.log_level
    numProcs = args.num_procs

    start = time()
    data = np.loadtxt(dataPath, dtype=np.int32, delimiter=',')
    print("Data read time: {:.3f}s".format(time() - start))

    X = data[:, :CLASS_COLUMN]
    Y = data[:,  CLASS_COLUMN]

    X, Y = shuffle(X, Y)

    testSize = floor(float(X.shape[0]) * TEST_RATIO)

    testX,  testY  = X[:testSize, :], Y[:testSize]
    trainX, trainY = X[testSize:, :], Y[testSize:]

    scaler = MinMaxScaler()

    trainX = scaler.fit_transform(trainX)
    testX  = scaler.transform(testX)
    print('Train set size: {}\ntest set size: {}'.format(trainX.shape[0], testX.shape[0]))

    # full set:
    print('Full set:')
    startAll = time()
    h = KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
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
    ksu = KSU(trainX, trainY, metric, logLevel=logLevel, n_jobs=-1)
    print("Init time: {:.3f}s".format(time() - start))

    start = time()
    ksu.compressData(delta=delta, stride=stride, greedy=mode == 'G', numProcs=numProcs, logLevel=logLevel, maxCompress=0.5)
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


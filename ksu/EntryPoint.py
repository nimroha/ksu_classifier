import sys
import argparse
import logging
import numpy as np

from time import time

from Utils   import parseInputData
from ksu.KSU import KSU, METRICS

def main(argv=None):

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='Generate a KSU classifier')
    parser.add_argument('--data_in',       help='Path to input data file (in .npz format with 2 nodes named X and Y)',         required=True)
    parser.add_argument('--data_out',      help='Path where output data will be saved',                                        required=True)
    parser.add_argument('--metric',        help='Metric to use (unless custom_metric is provided). {}'.format(METRICS.keys()), default='l2')
    parser.add_argument('--custom_metric', help='Absolute path to a directory (containing __init__.py) with a python file'
                                                'named Distance.py with a function named "dist(a, b)" that computes'
                                                'the distance between a and b by any metric of choice',                        default=None)
    parser.add_argument('--gram',          help='Path to a precomputed gram matrix (in .npz format)',                          default=None)
    parser.add_argument('--delta',         help='Required confidence level',                                                   default=0.05, type=float)
    parser.add_argument('--log_level',     help='Logging level',                                                               default='INFO')

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, filename='ksu.log')
    logger = logging.getLogger('KSU')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    dataInPath   = args.data_in
    dataOutPath  = args.data_out
    gramPath     = args.gram
    metric       = args.metric
    delta        = args.delta
    customMetric = args.custom_metric

    if customMetric is not None:
        sys.path.append(customMetric)
        try:
            from Distance import dist  # this only looks like an error because we're importing from unknown path
            metric = dist
            logger.debug('Loaded dist function successfully')
        except:
            raise RuntimeError(
                'Could not import dist function from {p}'
                'make sure Distance.py and __init__.py exist in {p}'
                'and that Distance.py has a function dist(a, b)'.format(p=customMetric))
    else:
        if metric not in METRICS.keys():
            raise RuntimeError(
                '"{m}" is not a built-in metric. use one of'
                '{ms}'
                'or provide a custom metric with the --custom_metric argument'.format(
                    m=metric,
                    ms=METRICS.keys()))

    logger.info('Reading data...')
    data = parseInputData(dataInPath)

    ksu = KSU(data['X'], data['Y'], metric, gramPath, logLevel=logging.INFO)
    ksu.compressData(delta)
    Xs, Ys = ksu.getCompressedSet()
    compression = ksu.getCompression()

    logger.info('Achieved {} compression, saving compressed set to {}...'.format(compression, dataOutPath))
    np.savez_compressed(dataOutPath, X=Xs, Y=Ys)

    logger.info('Done')

if __name__ == '__main__' :
    sys.exit(main())

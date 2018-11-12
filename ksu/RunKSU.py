import sys
import argparse
import logging
import numpy as np

from ksu.Utils import parseInputData
from ksu.KSU   import KSU, METRICS

def main(argv=None):

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='Generate a 1 nearest neighbors classifier fitted to a KSU compressed dataset')
    parser.add_argument('--data_in',         help='Path to input data file (in .npz format with 2 nodes named X and Y)',          required=True)
    parser.add_argument('--data_out',        help='Path where output data will be saved',                                         required=True)
    parser.add_argument('--metric',          help='Metric to use (unless custom_metric is provided)',                             default='l2', choices=METRICS.keys())
    parser.add_argument('--custom_metric',   help='Absolute path to a directory (containing __init__.py) with a python file'
                                                  'named Distance.py with a function named "dist(a, b)" that computes'
                                                  'the distance between a and b by any metric of choice',                         default=None)
    parser.add_argument('--gram',            help='Path to a precomputed gram matrix (in .npz format with a node named gram)',    default=None)
    parser.add_argument('--compress_limits', help='high,low compression ratio limits (comma separated, no spaces)',               default=None)
    parser.add_argument('--stride',          help='How many gammas to skip at a time (similar gammas will produce similar nets)', default=200, type=int)
    parser.add_argument('--delta',           help='Required confidence level',                                                    default=0.05, type=float)
    parser.add_argument('--mode',            help='which constuction mode.\n'
                                                  '"G" for greedy (faster, but bigger net), "H" for hierarchical',                default="G", choices=['G', 'H'])
    parser.add_argument('--num_procs',       help='Number of processes to use for computation',                                   default=1, type=int)
    parser.add_argument('--log_level',       help='Logging level',                                                                default=logging.CRITICAL)

    args = parser.parse_args()

    dataInPath   = args.data_in
    dataOutPath  = args.data_out
    gramPath     = args.gram
    metric       = args.metric
    delta        = args.delta
    mode         = args.mode
    customMetric = args.custom_metric
    compressLims = args.compress_limits
    stride       = args.stride
    logLevel     = args.log_level
    numProcs     = args.num_procs

    logging.basicConfig(level=logLevel, filename='ksu.log')
    logger = logging.getLogger('KSU')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    greedy = mode == 'G'

    if customMetric is not None:
        sys.path.append(customMetric)
        try:
            from Distance import dist  # this only looks like an error because we're importing from an unknown path
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
    gram = None
    if gramPath is not None:
        logger.info('Loading gram...')
        gram = np.load(gramPath)['gram']

    if compressLims is None:
        maxC = 1.0
        minC = 0.0
    else:
        ratios = compressLims.split(',')
        try:
            maxC = float(ratios[0])
            minC = float(ratios[1])
        except (IndexError, ValueError):
            raise RuntimeError('compress_limits shoud be two floats, comma separated, no spaces (e.g. "0.5,0.01").\ngiven {}'.format(compressLims))

        if maxC < minC:
            raise RuntimeError('compress_limits argument order is <high>,<low>')

    ksu = KSU(data['X'], data['Y'], metric, gram, logLevel=logLevel, n_jobs=1)
    ksu.compressData(delta=delta, maxCompress=maxC, minCompress=minC, greedy=greedy, stride=stride, numProcs=numProcs, logLevel=logLevel)
    Xs, Ys      = ksu.getCompressedSet()
    compression = ksu.getCompression()

    logger.info('Achieved {} compression, saving compressed set to {}'.format(compression, dataOutPath))
    np.savez_compressed(dataOutPath, X=Xs, Y=Ys)

    logger.info('Done')

if __name__ == '__main__' :
    sys.exit(main())

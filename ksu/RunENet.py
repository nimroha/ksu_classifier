import sys
import argparse
import logging
import numpy as np

from time                     import time
from sklearn.metrics.pairwise import pairwise_distances

from Utils   import parseInputData
from ksu.KSU import METRICS
from ksu.epsilon_net import EpsilonNet

def main(argv=None):

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='Generate an epsilon net of the dataset')
    parser.add_argument('--data_in',       help='Path to input data file (in .npz format with a node named X)\n'
                                                'If another node named Y is present, the script will save the'
                                                'respective labels from Y',                                                    required=True)
    parser.add_argument('--data_out',      help='Path where output data will be saved',                                        required=True)
    parser.add_argument('--epsilon',       help='Required epsilon of net',                                                     required=True, type=float)
    parser.add_argument('--metric',        help='Metric to use (unless custom_metric is provided). {}'.format(METRICS.keys()), default='l2')
    parser.add_argument('--custom_metric', help='Absolute path to a directory (containing __init__.py) with a python file'
                                                'named Distance.py with a function named "dist(a, b)" that computes'
                                                'the distance between a and b by any metric of choice',                        default=None)
    parser.add_argument('--mode',          help='which constuction mode.\n'
                                                '"G" for greedy (faster, but bigger net), "H" for hierarchical',               default="G")
    parser.add_argument('--gram',          help='Path to a precomputed gram matrix (in .npz format with a node named gram)',   default=None)
    parser.add_argument('--log_level',     help='Logging level',                                                               default='INFO')

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, filename='e-net.log')
    logger = logging.getLogger('E-Net')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    dataInPath   = args.data_in
    dataOutPath  = args.data_out
    gramPath     = args.gram
    metric       = args.metric
    epsilon      = args.epsilon
    mode         = args.mode
    customMetric = args.custom_metric

    if mode not in ['G', 'H']:
        raise RuntimeError('Mode {} is not supported. must be either of "G" for greedy (faster, but bigger net), "H" for hierarchical'.format(mode))

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
    data = np.load(dataInPath)
    try:
        Xs = data['X']
    except KeyError:
        raise RuntimeError('{} does not contain a node named X'.format(dataInPath))

    try:
        Ys = data['Y']
    except KeyError:
        Ys = None
        logger.debug('No labels')

    if gramPath is None:
        logger.info('Computing Gram matrix...')
        tStartGram = time()
        gram = pairwise_distances(Xs, metric=metric)
        logger.debug('Gram computation took {:.3f}s'.format(time() - tStartGram))
    else:
        logger.info('Loading gram...')
        gram = np.load(gramPath)['gram']

    gram = gram / np.max(gram)

    if mode == 'G':
        construct = EpsilonNet.greedyConstructEpsilonNetWithGram
    else:
        construct = EpsilonNet.hieracConstructEpsilonNet

    tStartNet   = time()
    net, idxs   = construct(Xs, gram, epsilon)
    compression = float(len(net)) / len(Xs)
    logger.info('Achieved {c} compression in time {t}s\n'
                'saving compressed set to {p}...'.format(
        c=compression,
        p=dataOutPath,
        t=time() - tStartNet))

    dataOut = {'X': net}
    if Ys is not None:
        dataOut = {'Y': Ys[idxs]}

    np.savez_compressed(dataOutPath, **dataOut)
    logger.info('Done')

if __name__ == '__main__' :
    sys.exit(main())

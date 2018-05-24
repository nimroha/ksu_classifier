import sys
import argparse
import logging

import Metrics
from Utils import parseInputData, getDateTime
from KSU   import KSU

METRICS = {'L1': Metrics.l1,
           'L2': Metrics.l2}
def main(argv=None):

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='Generate a KSU classifier')
    parser.add_argument('--data',           help='Path to input data file (in space separated key value format)',                required=True)
    parser.add_argument('--metric',         help='Metric to use (unless custom_metric is provided). {}'.format(METRICS.keys()),  default='L2')
    parser.add_argument('--custom_metric',  help='Absolute path to a directory (containing __init__.py) with a python file'
                                                 'named Distance.py with a function named "dist(a, b)" that computes'
                                                 'the distance between a and b by any metric of choice',                         default=None)
    parser.add_argument('--gram',           help='Path to a precomputed gram matrix (in .npz format)',                           default=None) # TODO decide which format. npz/panda/csv?
    parser.add_argument('--delta',          help='Delta parameter',                                                              default=0.1)  # TODO explain better, choose better default
    parser.add_argument('--log_level',      help='Logging level',                                                                default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, filename='ksu.log')
    logger = logging.getLogger('KSU')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    dataPath = args.data
    gramPath = args.gram
    metric   = args.metric
    delta    = args.delta
    customMetricPath = args.custom_metric

    if customMetricPath is not None:
        sys.path.append(customMetricPath)
        try:
            from Distance import dist  # this only looks like an error
            metric = dist
            logger.debug('Loaded dist function successfully')
        except:
            raise RuntimeError('Could not import dist function from {p}'
                               'make sure Distance.py and __init__.py exist in {p}'
                               'and that Distance.py has a function dist(a, b)')

    logger.info('Reading data...')
    data = parseInputData(dataPath)

    ksu = KSU(data['X'], data['Y'], gramPath, metric, logger)
    h   = ksu.makePredictor(delta)




if __name__ == '__main__' :
    sys.exit(main())

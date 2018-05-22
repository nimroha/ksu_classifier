import sys
import argparse
import logging

from Utils import parseInputData, getDateTime
from KSU   import KSU

def main(argv=None):

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='Generate a KSU classifier')
    parser.add_argument('--data',      help='Path to input data file (in space separated key value format)',           required=True)
    parser.add_argument('--dist',      help='Absolute path to a directory (containing __init__.py) with a python file'
                                            'named Distance.py with a function named "dist(a, b)" that computes'
                                            'the distance between a and b by any metric of choice',                    required=True)
    parser.add_argument('--gram',      help='Path to a precomputed gram matrix (in ... format)',                       default=None) # TODO decide which format. npz/panda/csv?
    parser.add_argument('--log_level', help='Logging level',                                                           default=logging.INFO)

    args = parser.parse_args()

    dataPath = args.data
    distPath = args.dist
    gramPath = args.gram

    logging.basicConfig(level=args.log_level, filename='ksu.log')
    logger = logging.getLogger('KSU')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info('Reading data...')
    parseInputData(dataPath)





if __name__ == '__main__' :
    sys.exit(main())

import sys
import argparse
import logging


def main(argv=None):

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='Generate a KSU classifier')
    parser.add_argument('--data',        help='Path to input data file (in space separated key value format)',                                      required=True)
    parser.add_argument('--metric_mode', help='"dist" - distances will be computed from the function provided in the file <dist_path>'
                                              '"gram" - distances will be taken from the gram matrix provided in <gram> (higher efficiency mode)',  required=True)
    parser.add_argument('--gram',        help='Path to a precomputed gram matrix (in ... format)',                                                  default=None) # TODO decide wiich format. npz/panda/csv?
    parser.add_argument('--dist_path',   help='Absolute path to a python file with a function named "dist(a, b)" that'
                                              'computes the distance between a and b by any metric of choice',                                      default=None)

    args = parser.parse_args()

    logger = logging.getLogger('KSU')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info('Reading data')



if __name__ == '__main__' :
    sys.exit(main())

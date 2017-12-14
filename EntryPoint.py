import sys
import argparse


def main(argv=None):

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='Generate a KSU classifier')
    parser.add_argument('--data',  help='Path to input data files',        required=True)
    parser.add_argument('--gram',  help='Path to precomputed gram matrix', default=None)
    args = parser.parse_args()



if __name__ == '__main__' :
    sys.exit(main())

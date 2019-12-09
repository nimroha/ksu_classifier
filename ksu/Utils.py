import time
import logging
import sys
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from collections       import Counter
from math              import sqrt
from tqdm              import tqdm
from numba             import jit, prange


LOGGER_FORMAT = '%(asctime)s [%(levelname)s] PID %(process)d: %(module)s - %(message)s'
READABLE_TIME = '%Y-%m-%d_%H.%M.%S'


class TqdmHandler(logging.Handler):
    def __init__ (self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except KeyboardInterrupt:
            pass
        except:
            self.handleError(record)

class TqdmStream(object):
    @classmethod
    def write(cls, msg):
        tqdm.write(msg, end='')

def getDateTime():
    return time.strftime(READABLE_TIME)

def parseInputData(dataPath):
    nodes = np.load(dataPath)
    try:
        data = {node: nodes[node] for node in ['X', 'Y']}
    except KeyError:
        raise RuntimeError('file at {p} does not contain the nodes "X" "Y"'.format(p=dataPath))

    return data

def computeGram(elements, dist): #unused
    n    = len(elements)
    gram = np.array((n, n))
    for i in range(n):
        for j in range(n - i):
            gram[i,j] = dist(elements[i], elements[j])

    lowTriIdxs       = np.tril_indices(n) #TODO make sure gram is upper triangular after the loop
    gram[lowTriIdxs] = gram.T[lowTriIdxs]

    return gram

@jit(nopython=True)
def computeQ(n, m, alpha, delta, w1=1, w2=1, w3=1):
    """
    Compute the parameter q that approximates an upper limit for the
    error of a 1-NN classifier based on the compressed set

    :param n: size of original set
    :param m: size of compressed set
    :param alpha: empirical error on the original set
    :param delta: level of confidence
    :param w1: first term weight
    :param w2: second term weight
    :param w3: third term weight

    :return: the approximation q
    """
    if m >= n:
        return float(np.inf)

    firstTerm  = (n * alpha) / (n - m)
    secondTerm = (m * np.log2(n) - np.log2(delta)) / (n - m)
    thirdTerm  = sqrt(((n * m * alpha * np.log2(n)) / (n - m) - np.log2(delta)) / (n - m))

    return w1 * firstTerm + w2 * secondTerm + w3 * thirdTerm

def computeLabels(gammaXs, Xs, Ys, metric, n_jobs=1): # unused
    """
    Compute the labels of the compressed set with a nearest neighbor majority vote

    :param gammaXs: the compressed set
    :param Xs: the original set
    :param Ys: the original labels
    :param metric: the distance metric
    :param n_jobs: number of cpus to employ (with scipy logic)

    :return: the labels of the compressed set
    """
    gammaN  = len(gammaXs)
    gammaYs = range(gammaN)
    h       = KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=n_jobs)
    h.fit(gammaXs, gammaYs)
    groups      = [Counter() for _ in range(gammaN)]
    predictions = h.predict(Xs) # cluster id for each x (ids form gammaYs)
    [groups[label].update(Ys[np.where(predictions == label)]) for label in gammaYs] # count all the labels in the cluster

    return np.array([c.most_common(1)[0][0] for c in groups])


def optimizedComputeLabels(gammaIdxs, gram, Ys, neighbors=3, n_jobs=1):
    """
    An optimized version of :func:computeLabels
    Compute the labels of the compressed set with a nearest neighbor majority vote

    :param gammaIdxs: the compressed set indices
    :param gram: the distance matrix
    :param Ys: the original labels
    :param neighbors: number of neighbors to consider
    :param n_jobs: number of cpus to employ (with scipy logic)

    :return: the labels of the compressed set
    """
    h = KNeighborsClassifier(n_neighbors=neighbors, metric='precomputed', algorithm='auto', n_jobs=n_jobs)
    h.fit(gram, Ys)
    nearest = np.squeeze(h.kneighbors()[1]) # take only indices
    representers = majorityLabels(Ys[nearest])

    return representers[gammaIdxs]


def computeAlpha(gammaXs, gammaYs, Xs, Ys, metric):
    """
    Compute the empirical error of the classifier fitted to the compressed set
    on the original set

    :param gammaXs: compressed set points
    :param gammaYs: compressed set labels
    :param Xs: original set points
    :param Ys: original set labels
    :param metric: distance metric

    :return: the misclassification error
    """
    classifier = KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
    classifier.fit(gammaXs, gammaYs)

    return classifier.score(Xs, Ys)


def optimizedComputeAlpha(gammaYs, Ys, gammaGram):
    """
    Optimized version of :func:computeAlpha

    :param gammaYs: compressed set labels
    :param Ys: original set labels
    :param gammaGram: rows of the original gram matrix corresponding to the compressed set

    :return: the misclassification error
    """
    nearest = np.argmin(gammaGram, axis=0) # nearest neighbors' indices in the compressed set
    missed  = Ys != gammaYs[nearest]

    return np.mean(missed)


def computeGammaSet(gram, stride=None):
    gammaSet = np.unique(gram)
    gammaSet = np.delete(gammaSet, 0)

    if stride is not None:
        gammaSet = gammaSet[::int(stride)]

    return gammaSet


def configLogger(logger, level):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOGGER_FORMAT, datefmt=READABLE_TIME))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)


@jit(nopython=True, nogil=True)
def majorityVote(arr):
    """
    if all elements are identical, return the first. otherwise return the majority
    """
    if arr[0] == arr.mean():
        return arr[0]

    elems  = np.unique(arr) # return_counts is not supported in numba version, this is still faster
    counts = np.array([np.sum(arr == i) for i in elems])
    argmax = np.argmax(counts)

    return elems[argmax]


@jit(nopython=True, nogil=True)
def majorityLabels(neighbors):
    ys = np.zeros(neighbors.shape[0], dtype=np.int32)
    for i in prange(neighbors.shape[0]):
        ys[i] = majorityVote(neighbors[i])

    return ys

import numpy as np
import time
import logging

from sklearn.neighbors import KNeighborsClassifier
from collections       import Counter
from math              import sqrt
from tqdm              import tqdm
from numba             import jit

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
    def write(_, msg):
        tqdm.write(msg, end='')

def getDateTime():
    return time.strftime('%d.%m.%y %H:%M:%S')

def parseInputData(dataPath):
    nodes = np.load(dataPath)
    try:
        data = {node: nodes[node] for node in ['X', 'Y']}
    except KeyError:
        raise RuntimeError('file at {p} does not contain the nodes "X" "Y"'.format(dataPath))

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
def computeQ(n, m, alpha, delta):
    """
    Compute the parameter q that approximates an upper limit for the
    error of a 1-NN classifier based on the compressed set

    :param n: size of original set
    :param m: size of compressed set
    :param alpha: empirical error on the original set
    :param delta: level of confidence

    :return: the approximation q
    """
    firstTerm  = (n * alpha) / (n - m)
    secondTerm = (m * np.log2(n) - np.log2(delta)) / (n - m)
    thirdTerm  = sqrt(((n * m * alpha * np.log2(n)) / (n - m) - np.log2(delta)) / (n - m))

    return firstTerm + secondTerm + thirdTerm

def computeLabels(gammaXs, Xs, Ys, metric, n_jobs=1): # TODO deprecate after testing optimizedComputeLabels
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

def optimizedComputeLabels(gammaXs, gammaIdxs, Xs, Ys, gram): #unused #TODO debug
    m                = len(gammaXs)
    n                = len(Xs)
    gammmaGram       = gram[gammaIdxs]
    flatGram         = np.reshape(gammmaGram, [-1])
    perm             = np.argsort(flatGram)
    flatYs           = np.array(Ys.tolist() * m)[perm]
    flatXIdxs        = np.array([j for l in [[e] * n for e in range(m)] for j in l])[perm]
    flatNeighborIdxs = np.array(range(n) * m)[perm]
    taken            = np.full([n], False, dtype=bool)

    numTaken = 0
    groups   = [Counter() for _ in range(m)]
    for y, xIdx, neighborIdx in zip(flatYs, flatXIdxs, flatNeighborIdxs):
        if numTaken == n:
            break
            
        if taken[neighborIdx]:
           continue

        groups[xIdx][y] += 1
        taken[neighborIdx] = True
        numTaken += 1

    return [c.most_common(1)[0][0] for c in groups]


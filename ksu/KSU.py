import os
import sys
import logging
import numpy as np

from time                     import time

from requests_toolbelt.downloadutils import stream
from tqdm                     import tqdm
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.neighbors.base   import VALID_METRICS
from sklearn.metrics.pairwise import pairwise_distances
from epsilon_net.EpsilonNet import greedyConstructEpsilonNetWithGram, hieracConstructEpsilonNet, \
    optimizedHieracConstructEpsilonNet

import Metrics

METRICS = {v:v for v in VALID_METRICS['brute'] if v != 'precomputed'}
METRICS['EditDistance'] = Metrics.editDistance
METRICS['EarthMover']   = Metrics.earthMoverDistance

from Utils import computeGammaSet, \
                  computeLabels, \
                  optimizedComputeAlpha, \
                  computeQ

def constructGammaNet(Xs, gram, gamma, prune=False, greedy=True):
    """
    Construct an epsilon net for parameter gamma

    :param Xs: points
    :param gram: gram matrix of the points
    :param gamma: net parameter
    :param prune: whether to prune the net after construction
    :param greedy: whether to build the net greedily or with an hierarchical strategy

    :return: the chosen points and their indices
    """
    if greedy:
        chosenXs, chosen = greedyConstructEpsilonNetWithGram(Xs, gram, gamma)
    else:
        chosenXs, chosen = hieracConstructEpsilonNet(Xs, gram, gamma)

    if prune:
        pass # TODO shoud we also implement this?

    return chosenXs, np.where(chosen)

def optimizedConstructGammaNet(Xs, gram, gamma, prune=False, greedy=True):
    """
    An optimized version of :func:constructGammaNet

    :param Xs: points
    :param gram: gram matrix of the points
    :param gamma: net parameter
    :param prune: whether to prune the net after construction
    :param greedy: whether to build the net greedily or with an hierarchical strategy

    :return: the chosen points and their indices
    """
    if greedy:
        chosenXs, chosen = greedyConstructEpsilonNetWithGram(Xs, gram, gamma)
    else:
        chosenXs, chosen = optimizedHieracConstructEpsilonNet(Xs, gram, gamma)

    if prune:
        pass # TODO shoud we also implement this?

    return chosenXs, np.where(chosen)

def compressDataWorker(gammaSet, delta, Xs, Ys, metric, gram, minC, maxC, greedy, logLevel=logging.CRITICAL):
    n           = len(Xs)
    numClasses  = len(np.unique(Ys))
    bestGamma   = 0.0
    qMin        = float(np.inf)
    compression = 0.0
    chosenXs    = None
    chosenYs    = None
    logger      = logging.getLogger('KSU-{}'.format(os.getpid()))

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logLevel)

    logger.debug('Choosing from {} gammas'.format(len(gammaSet)))
    for gamma in tqdm(gammaSet):
        tStart = time()
        gammaXs, gammaIdxs = constructGammaNet(Xs, gram, gamma, greedy=greedy)
        compression = float(len(gammaXs)) / n
        logger.debug('Gamma: {g}, net construction took {t:.3f}s, compression: {c}'.format(
            g=gamma,
            t=time() - tStart,
            c=compression))

        if compression > maxC:
            continue  # heuristic: don't bother compressing by less than an order of magnitude

        if compression < minC:
            break  # heuristic: gammas are increasing, so we might as well stop here

        if len(gammaXs) < numClasses:
            logger.debug(
                'Gamma: {g}, compressed set smaller than number of classes ({cc} vs {c})'
                'no use building a classifier that will never classify some classes'.format(
                    g=gamma,
                    cc=len(gammaXs),
                    c=numClasses))
            break

        tStart  = time()
        gammaYs = computeLabels(gammaXs, Xs, Ys, metric)
        logger.debug('Gamma: {g}, label voting took {t:.3f}s'.format(
            g=gamma,
            t=time() - tStart))

        tStart = time()
        alpha  = optimizedComputeAlpha(gammaYs, Ys, gram[gammaIdxs])
        logger.debug('Gamma: {g}, error approximation took {t:.3f}s, error: {a}'.format(
            g=gamma,
            t=time() - tStart,
            a=alpha))

        m = len(gammaXs)
        q = computeQ(n, alpha, 2 * m, delta)

        if q < qMin:
            logger.info(
                'Gamma: {g} achieved lowest q so far: {q}, for compression {c}, and empirical error {a}'.format(
                    g=gamma,
                    q=q,
                    c=compression,
                    a=alpha))

            qMin        = q
            bestGamma   = gamma
            chosenXs    = gammaXs
            chosenYs    = gammaYs
            compression = compression

    logger.info('Chosen best gamma: {g}, which achieved q: {q}, and compression: {c}'.format(
        g=bestGamma,
        q=qMin,
        c=compression))

    return qMin, chosenXs, chosenYs, compression, bestGamma

class KSU(object):

    def __init__(self, Xs, Ys, metric, gram=None, prune=False, logLevel=logging.CRITICAL, n_jobs=1):
        self.Xs          = Xs
        self.Ys          = Ys
        self.logger      = logging.getLogger('KSU')
        self.metric      = metric
        self.n_jobs      = n_jobs
        self.chosenXs    = None
        self.chosenYs    = None
        self.compression = None
        self.numClasses  = len(np.unique(self.Ys))
        self.prune       = prune # unused since pruning is not implemented yet

        logging.basicConfig(level=logLevel)

        if isinstance(metric, str) and metric not in METRICS.keys():
            raise RuntimeError(
                '"{m}" is not a built-in metric. use one of'
                '{ms}'
                'or provide your own custom metric as a callable'.format(
                    m=metric,
                    ms=METRICS.keys()))

        if gram is None:
            self.logger.info('Computing Gram matrix...')
            tStartGram = time()
            self.gram  = pairwise_distances(self.Xs, metric=self.metric, n_jobs=self.n_jobs)
            self.logger.debug('Gram computation took {:.3f}s'.format(time() - tStartGram))
        else:
            self.gram = gram

        self.gram     = self.gram / np.max(self.gram)

    def getCompressedSet(self):
        """
        Getter for compressed set

        :return: the compressed set and its labels

        :raise: :class:RuntimeError if :func:KSU.KSU.compressData was not run before
        """
        if self.chosenXs is None:
            raise RuntimeError('getCompressedSet - you must run KSU.compressData first')

        return self.chosenXs, self.chosenYs

    def getCompression(self):
        """
        Getter for compression ratio

        :return: the compression ratio

        :raise: :class:RuntimeError if :func:KSU.KSU.compressData was not run before
        """
        if self.compression is None:
            raise RuntimeError('getCompression - you must run KSU.compressData first')

        return self.compression

    def getClassifier(self):
        """
        Getter for 1-NN classifier fitted on the compressed set

        :return: an :mod:sklearn.KNeighborsClassifier instance

        :raise: :class:RuntimeError if :func:KSU.KSU.compressData was not run before
        """
        if self.chosenXs is None:
            raise RuntimeError('getClassifier - you must run KSU.compressData first')

        h = KNeighborsClassifier(n_neighbors=1, metric=self.metric, algorithm='auto', n_jobs=self.n_jobs)
        h.fit(self.chosenXs, self.chosenYs)

        return h

    def compressData(self, delta=0.1, minCompress=0.05, maxCompress=0.1, greedy=True, stride=1000, numProcs=1):
        """
        Run the KSU algorithm to compress the dataset

        :param delta: confidence for error upper bound
        :param minCompress: minimum compression ratio
        :param maxCompress: maximum compression ratio
        :param greedy: whether to use greedy or hierarchical strategy for net construction
        :param stride: how many gammas to skip between each iteration
        :param numProcs: number of processes to use
        """
        gammaSet = computeGammaSet(self.gram, stride=stride)

        if numProcs == 1:
            qMin, self.chosenXs, self.chosenYs, self.compression, bestGamma = \
            compressDataWorker(gammaSet, delta, self.Xs, self.Ys, self.metric, self.gram, minCompress, maxCompress, greedy, logLevel=logging.DEBUG)
        else:
            gammaSets = np.reshape(gammaSet, [numProcs, -1])
            raise NotImplemented


        # self.logger.info('Chosen best gamma: {g}, which achieved q: {q}, and compression: {c}'.format(
        #     g=bestGamma,
        #     q=qMin,
        #     c=self.compression))

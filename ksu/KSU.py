import os
import sys
import logging
import numpy as np

from time                     import time
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.neighbors.base   import VALID_METRICS
from sklearn.metrics.pairwise import pairwise_distances

import Metrics
from epsilon_net.EpsilonNet import greedyConstructEpsilonNetWithGram

METRICS = {v:v for v in VALID_METRICS['brute'] if v != 'precomputed'}
METRICS['EditDistance'] = Metrics.editDistance
METRICS['EarthMover']   = Metrics.earthMoverDistance

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_condensing', 'Python Implementation'))
from nn_condensing import nn # this only looks like an error because the IDE doesn't understand the ugly hack above ^
from Utils         import computeGammaSet, \
                          computeLabels, \
                          optimizedComputeLabels, \
                          computeAlpha, \
                          computeQ

class NNDataObject(object):

    def __init__(self, x, y, index):

        self.x = x
        self.y = y
        self.index = index

    def getLine(self):
        return self.x

    def getTag(self):
        return self.y

    def getIndex(self):
        return self.index

class NNDataSet(object):

    def __init__(self, Xs, Ys):

        data = []
        for i in xrange(len(Xs)):
            data.append(NNDataObject(Xs[i], Ys[i], i))

        self.data = data

def constructGammaNet(Xs, Ys, metric, gram, gamma, prune):

    chosenXs, chosen = greedyConstructEpsilonNetWithGram(Xs, gram, gamma)
    if prune:
        pass # TODO shoud we also implement this?

    return chosenXs, np.where(chosen)

class KSU(object):

    def __init__(self, Xs, Ys, metric, gramPath=None, prune=False):
        self.Xs         = Xs
        self.Ys         = Ys
        self.prune      = prune
        self.logger     = logging.getLogger('KSU')
        self.metric     = metric
        self.chosenXs   = None
        self.chosenYs   = None
        self.numClasses = len(np.unique(self.Ys))

        logging.basicConfig(level=logging.DEBUG)

        if isinstance(metric, str) and metric not in METRICS.keys():
            raise RuntimeError(
                '"{m}" is not a built-in metric. use one of'
                '{ms}'
                'or provide your own custom metric as a callable'.format(
                    m=metric,
                    ms=METRICS.keys()))

        if gramPath is None:
            self.logger.info('Computing Gram matrix...')
            tStartGram = time()
            self.gram  = pairwise_distances(self.Xs, metric=self.metric, n_jobs=1)
            self.logger.debug('Gram computation took {:.3f}s'.format(time() - tStartGram))
        else:
            self.logger.info('Loading Gram matrix from file...')
            self.gram = np.load(gramPath)

    def getCompressedSet(self):
        if self.chosenXs is None:
            raise RuntimeError('getCompressedSet - you must run KSU.compressData first')

        return self.chosenXs, self.chosenYs

    def getClassifier(self):
        if self.chosenXs is None:
            raise RuntimeError('getClassifier - you must run KSU.compressData first')

        h = KNeighborsClassifier(n_neighbors=1, metric=self.metric, algorithm='auto', n_jobs=1)
        h.fit(self.chosenXs, self.chosenYs)

        return h

    def compressData(self, delta=0.05):
        gammaSet = computeGammaSet(self.gram, stride=100)
        qMin     = float(np.inf)
        n        = len(self.Xs)

        self.logger.debug('Choosing from {} gammas'.format(len(gammaSet)))
        for gamma in gammaSet:
            tStart = time()
            gammaXs, gammaIdxs = constructGammaNet(self.Xs, self.Ys, self.metric, self.gram, gamma, self.prune)
            self.logger.debug('Gamma: {g}, net construction took {t:.3f}s, compression: {c}'.format(
                g=gamma,
                t=time() - tStart,
                c=float(len(gammaXs)) / n))

            if len(gammaXs) < self.numClasses:
                continue # no use building a classifier that will never classify some classes

            tStart  = time()
            gammaYs = computeLabels(gammaXs, self.Xs, self.Ys, self.metric)
            self.logger.debug('Gamma: {g}, label voting took {t:.3f}s'.format(g=gamma, t=time() - tStart))

            tStart = time()
            alpha  = computeAlpha(gammaXs, gammaYs, self.Xs, self.Ys, self.metric)
            self.logger.debug('Gamma: {g}, error approximation took {t:.3f}s'.format(g=gamma, t=time() - tStart))

            m = len(gammaXs)
            q = computeQ(n, alpha, 2 * m, delta)

            if q < qMin:
                self.logger.debug('Gamma: {g} achieved lowest q so far: {q}'.format(g=gamma, q=q))
                qMin          = q
                bestGamma     = gamma
                self.chosenXs = gammaXs
                self.chosenYs = gammaYs

        self.logger.info('Chosen best gamma: {g}, which achieved q: {q}, and compression: {c}'.format(
            g=bestGamma,
            q=qMin,
            c=float(len(self.chosenXs)) / n))











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
    optmizedHieracConstructEpsilonNet

import Metrics

METRICS = {v:v for v in VALID_METRICS['brute'] if v != 'precomputed'}
METRICS['EditDistance'] = Metrics.editDistance
METRICS['EarthMover']   = Metrics.earthMoverDistance

from Utils import computeGammaSet, \
                  computeLabels, \
                  optimizedComputeLabels, \
                  optimizedComputeAlpha, \
                  computeQ

def constructGammaNet(Xs, gram, gamma, prune, greedy=True):
    if greedy:
        chosenXs, chosen = greedyConstructEpsilonNetWithGram(Xs, gram, gamma)
    else:
        chosenXs, chosen = hieracConstructEpsilonNet(Xs, gram, gamma)

    if prune:
        pass # TODO shoud we also implement this?

    return chosenXs, np.where(chosen)

def optimizedConstructGammaNet(Xs, gram, gamma, prune, greedy=True):
    if greedy:
        chosenXs, chosen = greedyConstructEpsilonNetWithGram(Xs, gram, gamma)
    else:
        chosenXs, chosen = optmizedHieracConstructEpsilonNet(Xs, gram, gamma)

    if prune:
        pass # TODO shoud we also implement this?

    return chosenXs, np.where(chosen)

class KSU(object):

    def __init__(self, Xs, Ys, metric, gram=None, prune=False, minCompress=0.05, maxCompress=0.1, greedy=True, logLevel=logging.CRITICAL, n_jobs=1):
        self.Xs          = Xs
        self.Ys          = Ys
        self.prune       = prune
        self.minC        = minCompress
        self.maxC        = maxCompress
        self.greedy      = greedy
        self.logger      = logging.getLogger('KSU')
        self.metric      = metric
        self.n_jobs      = n_jobs
        self.chosenXs    = None
        self.chosenYs    = None
        self.compression = None
        self.numClasses  = len(np.unique(self.Ys))

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
            self.gram  = pairwise_distances(self.Xs, metric=self.metric)
            self.logger.debug('Gram computation took {:.3f}s'.format(time() - tStartGram))
        else:
            self.gram = gram

        self.gram = self.gram / np.max(self.gram)

    def getCompressedSet(self):
        if self.chosenXs is None:
            raise RuntimeError('getCompressedSet - you must run KSU.compressData first')

        return self.chosenXs, self.chosenYs

    def getCompression(self):
        if self.compression is None:
            raise RuntimeError('getCompression - you must run KSU.compressData first')

        return self.compression

    def getClassifier(self):
        if self.chosenXs is None:
            raise RuntimeError('getClassifier - you must run KSU.compressData first')

        h = KNeighborsClassifier(n_neighbors=1, metric=self.metric, algorithm='auto', n_jobs=self.n_jobs)
        h.fit(self.chosenXs, self.chosenYs)

        return h

    def compressData(self, delta=0.1):
        gammaSet    = computeGammaSet(self.gram, stride=100)
        qMin        = float(np.inf)
        n           = len(self.Xs)

        self.logger.debug('Choosing from {} gammas'.format(len(gammaSet)))
        for gamma in tqdm(gammaSet):
            tStart = time()
            gammaXs, gammaIdxs = constructGammaNet(self.Xs, self.gram, gamma, prune=self.prune, greedy=self.greedy)
            compression = float(len(gammaXs)) / n
            self.logger.debug('Gamma: {g}, net construction took {t:.3f}s, compression: {c}'.format(
                g=gamma,
                t=time() - tStart,
                c=compression))

            if compression > self.maxC:
                continue # heuristic: don't bother compressing by less than an order of magnitude

            if compression < self.minC:
                break # heuristic: gammas are increasing, so we might as well stop here

            if len(gammaXs) < self.numClasses:
                self.logger.debug(
                    'Gamma: {g}, compressed set smaller than number of classes ({cc} vs {c})'
                    'no use building a classifier that will never classify some classes'.format(
                        g=gamma,
                        cc=len(gammaXs),
                        c=self.numClasses))
                break

            tStart  = time()
            gammaYs = computeLabels(gammaXs, self.Xs, self.Ys, self.metric, self.n_jobs)
            self.logger.debug('Gamma: {g}, label voting took {t:.3f}s'.format(g=gamma, t=time() - tStart))
            # tStart  = time()
            # gammaYs = optimizedComputeLabels(gammaXs, gammaIdxs, self.Xs, self.Ys, self.gram)
            # self.logger.debug('Gamma: {g}, label voting took {t:.3f}s'.format(g=gamma, t=time() - tStart))

            tStart = time()
            alpha = optimizedComputeAlpha(gammaYs, self.Ys, self.gram[gammaIdxs])
            self.logger.debug('Gamma: {g}, error approximation took {t:.3f}s, error: {a}'.format(g=gamma,
                                                                                                 t=time() - tStart,
                                                                                                 a=alpha))

            m = len(gammaXs)
            q = computeQ(n, alpha, 2 * m, delta)

            if q < qMin:
                self.logger.info(
                    'Gamma: {g} achieved lowest q so far: {q}, for compression {c}, and empirical error {a}'.format(
                        g=gamma,
                        q=q,
                        c=compression,
                        a=alpha))

                qMin             = q
                bestGamma        = gamma
                self.chosenXs    = gammaXs
                self.chosenYs    = gammaYs
                self.compression = compression

        self.logger.info('Chosen best gamma: {g}, which achieved q: {q}, and compression: {c}'.format(
            g=bestGamma,
            q=qMin,
            c=self.compression))

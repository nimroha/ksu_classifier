import os
import sys
import logging
import numpy as np

from time                     import time
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.neighbors.base   import VALID_METRICS
from sklearn.metrics.pairwise import pairwise_distances

import Metrics

METRICS = {v:v for v in VALID_METRICS['brute'] if v != 'precomputed'}
METRICS['EditDistance'] = Metrics.editDistance
METRICS['EarthMover']   = Metrics.earthMoverDistance

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_condensing', 'Python Implementation'))
from nn_condensing import nn # this only looks like an error because the IDE doesn't understand the ugly hack above ^
from Utils         import computeGammaSet, \
                          computeLabels, \
                          computeAlpha, \
                          computeQ


def constructGammaNet(Xs, gram, gamma, prune):
    chosenXs = nn.epsilon_net_hierarchy(data_sample=Xs,
                                        epsilon=gamma,
                                        distance_measure=None,
                                        gram_matrix=gram)

    if prune:
        chosenXs = nn.consistent_pruning(net=chosenXs,
                                         distance_measure=None,
                                         gram_matrix=gram)

    return chosenXs

class KSU(object):

    def __init__(self, Xs, Ys, metric, gramPath=None, prune=False):
        self.classifier = None
        self.Xs         = Xs
        self.Ys         = Ys
        self.prune      = prune
        self.logger     = logging.getLogger('KSU')
        self.metric     = metric

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

    def predict(self, x):
        if self.classifier is None:
            raise RuntimeError("Predictor not generated yet. you must run KSU.makePredictor() before predicting")
        else:
            return self.classifier.predict(x)

    def makePredictor(self, delta):
        gammaSet = computeGammaSet(self.gram)
        qMin     = float(np.inf)
        n        = len(self.Xs)

        self.logger.debug('Choosing from {} gammas'.format(len(gammaSet)))
        for gamma in gammaSet:
            tStartGamma = time()
            gammaXs     = constructGammaNet(self.Xs, self.gram, gamma, self.prune)
            tStartLabel = time()
            gammaYs     = computeLabels(gammaXs, self.Xs, self.Ys, self.gram, self.metric)
            alpha       = computeAlpha(gammaXs, gammaYs, self.Xs, self.Ys)
            m           = len(gammaXs)
            q           = computeQ(n, alpha, 2 * m, delta)

            self.logger.debug(
                'For gamma: {g}, net construction took {nt:.3f}s, label choosing took {lt:.3f}s, q: {q}'.format(
                    g=gamma,
                    q=q,
                    nt=tStartLabel - tStartGamma,
                    lt=time() - tStartLabel))

            if q < qMin:
                qMin      = q
                bestGamma = gamma
                chosenXs  = gammaXs
                chosenYs  = gammaYs

        self.logger.info('Chosen best gamma: {g}, which achieved q: {q}'.format(
            g=bestGamma,
            q=qMin))

        self.classifier = KNeighborsClassifier(n_neighbors=1, metric=self.metric, algorithm='auto', n_jobs=1)
        self.classifier.fit(chosenXs, chosenYs)







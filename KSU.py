import os
import sys
import numpy as np
import logging
import time
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_condensing', 'Python Implementation'))
from nn_condensing import nn # this only looks like an error because the IDE doesn't understand the ugly hack above ^
from Utils         import computeGram, \
                          computeGammaSet, \
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

    def __init__(self, prune=False):
        self.classifier = None
        self.prune      = prune

    def predict(self, x):
        if self.classifier is None:
            raise RuntimeError("Predictor not generated yet. you must run KSU.makePredictor() before predicting")
        else:
            return self.classifier.predict(x)

    def makePredictor(self, Xs, Ys, metric, delta):
        gram     = computeGram(Xs, metric)
        gammaSet = computeGammaSet(gram)
        qMin     = float(np.inf)

        for gamma in gammaSet:
            gammaXs = constructGammaNet(Xs, gram, gamma, self.prune)
            m       = len(gammaXs)
            gammaYs = computeLabels(gammaXs, Xs, Ys, metric)
            alpha   = computeAlpha(gammaXs, gammaYs, Xs, Ys)
            q       = computeQ(len(Xs), alpha, 2 * m, delta)

            if q < qMin:
                qMin      = q
                bestGamma = gamma
                chosenXs  = gammaXs
                chosenYs  = gammaYs

        self.classifier = KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
        self.classifier.fit(chosenXs, chosenYs)








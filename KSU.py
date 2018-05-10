import os
import sys
import numpy as np
import sklearn.neighbors as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_condensing', 'Python Implementation'))

from nn_condensing import distance
from Utils import computeGram, \
                  computeGammaSet, \
                  computeLabels, \
                  computeAlpha, \
                  computeQ


def constructGammaNet(Xs, gram, gamma):
    return []

class KSU(object):

    def __init__(self):
        self.classifier = None

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
            gammaXs = constructGammaNet(Xs, gram, gamma)
            m       = len(gammaXs)
            gammaYs = computeLabels(gammaXs, Xs, Ys, metric)
            alpha   = computeAlpha(gammaXs, gammaYs, Xs, Ys)
            q       = computeQ(len(Xs), alpha, 2 * m, delta)

            if q < qMin:
                qMin      = q
                bestGamma = gamma
                chosenXs  = gammaXs
                chosenYs  = gammaYs

        self.classifier = nn.KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
        self.classifier.fit(chosenXs, chosenYs)








import Utils as utils
import numpy as np
import sklearn.neighbors as nn

class KSU(object):

    def __init__(self, compressedSet):
        self.set = compressedSet

    def predict(self, example):
        raise NotImplementedError("prediction not implemented")

    def KSU(self, Xs, Ys, metric, delta):
        gram = utils.computeGram(Xs, metric)
        gammaSet = utils.computeGammaSet(gram)
        chosenXs = Xs
        chosenYs = Ys
        Qmin = float(np.inf)
        for i in xrange(len(gammaSet)):
            gammaXs = utils.createGammaNet(Xs, gammaSet(i))
            m = len(gammaXs)
            gammaYs = utils.computeLabels(gammaXs, Xs, Ys, metric)
            gammaAlpha = utils.computeAlpha(gammaXs, gammaYs, Xs, Ys)
            gammaQ = utils.computeQ(len(Xs), gammaAlpha, 2*m, delta)
            if (gammaQ < Qmin):
                Qmin = gammaQ
                chosenXs = gammaXs
                chosenYs = gammaYs

        classifier = nn.KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
        classifier.fit(chosenXs, chosenYs)
        return lambda x : classifier.predict(x)








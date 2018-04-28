import numpy as np
import sklearn.neighbors as nn

from math import sqrt, log

def log2(x):
    return log(x, 2)

def computeGram(elements, metric):
    # naive first go
    n = len(elements)
    gram = np.array((n, n))
    for i in range(n):
        for j in range(n):
            gram[i,j] = metric(elements[i], elements[j])

    return gram

def computeQ(n, m, alpha, delta):
    firstTerm = (n * alpha) / (n - m)

    secondTerm = (m * log2(n) - log2(delta)) / (n - m)

    thirdTerm = sqrt(((n * m * alpha * log2(n)) / (n - m) - log2(delta)) / (n - m))

    return firstTerm + secondTerm + thirdTerm

def  computeAlpha(gammaXs, gammaYs, Xs, Ys, metric):
    classifier = nn.KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
    classifier.fit(gammaXs, gammaYs)
    return classifier.score(Xs, Ys)

def computeGammaSet(gram):
    gammaSet = np.unique(gram)
    gammaSet = np.delete(gammaSet, 0)
    return gammaSet

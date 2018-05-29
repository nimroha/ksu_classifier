import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from collections       import Counter
from math              import sqrt, log

def getDateTime():
    return time.strftime('%d.%m.%y %H:%M:%S')

def parseInputData(dataPath):
    data = np.load(dataPath)
    return data # TODO return as dict (like in w2v-parseNPZ)

def log2(x):
    return log(x, 2)

def computeGram(elements, dist): #unused
    n    = len(elements)
    gram = np.array((n, n))
    for i in range(n):
        for j in range(n - i):
            gram[i,j] = dist(elements[i], elements[j])

    lowTriIdxs       = np.tril_indices(n) #TODO make sure gram is upper triangular after the loop
    gram[lowTriIdxs] = gram.T[lowTriIdxs]

    return gram

def computeQ(n, m, alpha, delta):
    firstTerm  = (n * alpha) / (n - m)
    secondTerm = (m * log2(n) - log2(delta)) / (n - m)
    thirdTerm  = sqrt(((n * m * alpha * log2(n)) / (n - m) - log2(delta)) / (n - m))

    return firstTerm + secondTerm + thirdTerm

def computeLabels(gammaXs, Xs, Ys, gram, metric): # TODO deprecate after testing optimizedComputeLabels
    gammaYs = range(len(gammaXs))
    h = KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
    h.fit(gammaXs, gammaYs)

    groups = {i:Counter() for i in gammaYs}
    for x, y in zip(Xs, Ys):
        groups[h.predict(x)].update(y)

    return [c.most_common(1)[0][0] for c in groups.keys()]

def computeAlpha(gammaXs, gammaYs, Xs, Ys, metric):
    classifier = KNeighborsClassifier(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
    classifier.fit(gammaXs, gammaYs)
    return classifier.score(Xs, Ys)

def computeGammaSet(gram):
    gammaSet = np.unique(gram)
    gammaSet = np.delete(gammaSet, 0)
    return gammaSet

def findIndices(array, elements):
    return filter(lambda x: x is not None, [array.index(e) if e in array else None for e in elements])

def optimizedComputeLabels(gammaXs, Xs, Ys, gram):
    m                = len(gammaXs)
    n                = len(Xs)
    slice            = findIndices(Xs, gammaXs)
    gammmaGram       = gram[:, slice]
    flatGram         = np.reshape(gammmaGram, [-1])
    perm             = np.argsort(flatGram)
    flatYs           = np.array(Ys * m)[perm]
    flatXIdxs        = np.array([j for l in [[e] * m for e in range(m)] for j in l])[perm]
    flatNeighborIdxs = np.array(range(n) * m)[perm]
    taken            = np.full([n], False, dtype=bool)

    numTaken = 0
    groups   = {i: Counter() for i in range(m)}
    for y, xIdx, neighborIdx in zip(flatYs, flatXIdxs, flatNeighborIdxs):
        if numTaken == n:
            break
            
        if taken[neighborIdx]:
           continue

        groups[xIdx].update(y)
        taken[neighborIdx] = True
        numTaken += 1

    return [c.most_common(1)[0][0] for c in groups.keys()]

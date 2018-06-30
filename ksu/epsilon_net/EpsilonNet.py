"""
Implementation of Algorithm 3 from [Near-optimal sample compression for nearest neighbors](https://www.cs.bgu.ac.il/~karyeh/condense-journal.pdf)

variable names sadly avoid normal convention to correspond to the paper notations
"""
import numpy as np
from math import log, ceil, floor


def greedyConstructEpsilonNetWithGram(points, gram, epsilon):
    idx = np.random.randint(0, len(points) - 1)

    net     = np.zeros_like(points)
    netGram = np.full_like(gram, np.inf)
    taken   = np.full(len(points), False)

    netGram[idx] = gram[idx]
    net[idx]     = points[idx]
    taken[idx]   = True

    for i, p in enumerate(points): #iterate rows
        if np.min(netGram[:,i]) >= epsilon:
            net[i]     = points[i]
            netGram[i] = gram[i]
            taken[i]   = True

    return net[taken], taken

def buildLevel(p, i, radius, gram, S, N, P, C):
    T = [e for l in [list(C[r, i]) for x in P[p, i] for r in N[x, i]] for e in l] #TODO simplify or explain

    for r in T:
        if gram[r, p] < radius:
            P[p, i - 1] = {r} #TODO take only one point or all points?
            return

    S[i - 1].add(p)
    N[(p, i - 1)].add(p)
    [C[r, i].add(p) for r in P[p, i]]

    for r in T:
        if gram[r, p] < 4 * radius:
            N[p, i - 1].add(r)
            N[r, i - 1].add(p)


def hieracConstructEpsilonNet(points, gram, epsilon):#WIP
    lowestLvl = int(ceil(log(epsilon, 2)))
    n = len(points)
    levels = range(1, lowestLvl - 1, -1)

    #arbitrary starting point
    startIdx = np.random.randint(0, n)

    # basic data structure
    D = {(p, i): set() for p in range(n) for i in levels}

    #init S - nets
    S = {i:set() for i in levels}
    S[levels[0]].add(startIdx)

    #init P - parents
    P = D.copy()
    for p in range(n):
        P[p, levels[0]] = {startIdx}

    #init N - neighbors
    N = D.copy()
    N[startIdx, levels[0]] = {startIdx}

    #init C - covered
    C = D.copy()

    for i in levels[:-1]:
        radius = pow(2, i - 1)
        for p in S[i]:
            buildLevel(p, i, radius, gram, S, N, P, C)
        for p in set(range(n)) - S[i]:
            buildLevel(p, i, radius, gram, S, N, P, C)

    # gauranteed to by an e-net of at least epsilon
    return list(S[lowestLvl])

from sklearn.metrics.pairwise import pairwise_distances
# xs = np.array([[0, 0],
#                [0, 1],
#                [1, 0],
#                [1, 1],
#                [2, 1],
#                [3, 1],
#                [2, 2],
#                [3, 2]])
ys = np.array([2, 0, 1, 2, 0, 1, 0, 2])
x0   = np.array([0, 0])
x1   = np.array([0, 0.5])
x2   = np.array([0.5, 0])
x3   = np.array([0.5, 0.5])
xs   = np.vstack((x0, x1, x2, x3))
gram = pairwise_distances(xs, metric='l2')
print(gram)
print(hieracConstructEpsilonNet(xs, gram, 0.3))

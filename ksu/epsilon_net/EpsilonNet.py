"""
Implementation of Algorithm 3 from [Near-optimal sample compression for nearest neighbors](https://www.cs.bgu.ac.il/~karyeh/condense-journal.pdf)

Epsilon Nets refer to https://en.wikipedia.org/wiki/Delone_set

variable names sadly avoid normal convention to correspond to the paper notations
"""

import numpy as np

from math import log, floor


def greedyConstructEpsilonNetWithGram(points, gram, epsilon):
    """
    Construct an Epsilon net in a greedy fashion.

    :param points: points to construct a net from
    :param gram: gram matrix of the points (for some metric)
    :param epsilon: epsilon parameter

    :return: the points chosen for the net and their indices (as an indicator vector)
    """
    idx = np.random.randint(0, len(points))

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
    """
    Builds a level in the net hierarchy.
    (content of the loop of line 6 in the algorithm 3's pseudo-code)

    :param p: starting point
    :param i: current level in the hierarchy
    :param radius: radius for epsilon net
    :param gram: gram matrix of all the points
    :param S: the entire hierarchy
    :param N: neighbors
    :param P: parents
    :param C: covers
    """

    T = [e for l in [list(C[r, i - 1]) for x in P[p, i] for r in N[x, i]] for e in l] #TODO simplify or explain

    for r in T:
        if gram[r, p] < radius:
            P[p, i - 1] = {r}
            return

    S[i - 1].add(p)
    N[(p, i - 1)].add(p)
    [C[r, i - 1].add(p) for r in P[p, i]]

    for r in T:
        if gram[r, p] < 4 * radius:
            N[p, i - 1].add(r)
            N[r, i - 1].add(p)

def optimizedBuildLevel(p, i, radius, gram, S, N, P, C):
    """
    Optimized version of :func:buildLevel

    :param p: starting point
    :param i: current level in the hierarchy
    :param radius: radius for epsilon net
    :param gram: gram matrix of all the points
    :param S: the entire hierarchy
    :param N: neighbors
    :param P: parents
    :param C: covers
    """
    def whereAndSqeeze(A):
        return np.squeeze(np.argwhere(A))

    _P = P[p,i]
    if np.any(_P):
        _N = N[whereAndSqeeze(_P), i]
        if np.any(_N):
            _C = C[whereAndSqeeze(_N), i - 1]
            if np.any(_C):
                T = whereAndSqeeze(_C)
                print('opt', list(T))

                tGram = gram[T]
                minTGram = np.min(tGram, axis=0)  # TODO ensure axis 0
                j = np.argmin(minTGram)
                if minTGram[j] < radius:
                    P[p, i + 1, j] = True
                    return

                valid = np.where(minTGram < 4 * radius)
                N[p, i + 1] |= valid
                N[valid, i + 1, p] = True

    # T = np.squeeze(np.argwhere(C[np.squeeze(N[P[p, i], i]), i - 1])) #TODO should be i-1
    # print('opt', list(T))
    #
    # if len(T) > 0:
    #     tGram = gram[T]
    #     minTGram = np.min(tGram, axis=0) #TODO ensure axis 0
    #     j = np.argmin(minTGram)
    #     if minTGram[j] < radius:
    #         P[p, i + 1, j] = True
    #         return
    #
    #     valid = np.where(minTGram < 4 * radius)
    #     N[p, i + 1] |= valid
    #     N[valid, i + 1, p] = True

    S[i + 1, p]    = True
    N[p, i + 1, p] = True
    C[P[p, i], i + 1] |= P[p, i]

def hieracConstructEpsilonNet(points, gram, epsilon):
    """
    Construct an Epsilon net in a hierarchical fashion.
    Note: the resulting net has a radius 2^floor(log_2(epsilon))

    :param points: points to construct a net from
    :param gram: gram matrix of the points (for some metric)
    :param epsilon: epsilon parameter

    :return: the points chosen for the net and their indices (as an indicator vector)
    """
    lowestLvl = int(floor(log(epsilon, 2)))
    n         = len(points)
    levels    = range(1, lowestLvl - 1, -1)

    #arbitrary starting point
    startIdx = np.random.randint(0, n)

    #init S - nets
    S = {i: set() for i in levels}
    S[levels[0]].add(startIdx)

    #init P - parents
    P = {(p, i): set() for p in range(n) for i in levels}
    for p in range(n):
        P[p, levels[0]] = {startIdx}

    #init N - neighbors
    N = {(p, i): set() for p in range(n) for i in levels}
    N[startIdx, levels[0]] = {startIdx}

    #init C - covered
    C = {(p, i): set() for p in range(n) for i in levels}

    for i in levels[:-1]:
        radius = pow(2, i - 1)
        for p in S[i]:
            buildLevel(p, i, radius, gram, S, N, P, C)
        for p in set(range(n)) - S[i]:
            buildLevel(p, i, radius, gram, S, N, P, C)

    # guaranteed to by an e-net of at least epsilon
    return points[list(S[lowestLvl])], list(S[lowestLvl])

def optimizedHieracConstructEpsilonNet(points, gram, epsilon):
    """
    An optimized version of :func:hieracConstructEpsilonNet

    :param points: points to construct a net from
    :param gram: gram matrix of the points (for some metric)
    :param epsilon: epsilon parameter

    :return: the points chosen for the net and their indices (as an indicator vector)
    """
    lowestLvl = int(floor(log(epsilon, 2)))
    levels    = range(1, lowestLvl - 1, -1)

    n = len(points)
    l = len(levels)

    #arbitrary starting point
    startIdx = np.random.randint(0, n)
    print('startIdx', startIdx)
    print('n', n)
    print('l', l)

    Svec = np.zeros([l, n], dtype=np.bool)
    Nvec = np.zeros([n, l, n], dtype=np.bool)
    Pvec = np.zeros([n, l, n], dtype=np.bool)
    Cvec = np.zeros([n, l, n], dtype=np.bool)

    Svec[0, startIdx] = True
    Pvec[:, 0, startIdx] = True
    Nvec[startIdx, 0, startIdx] = True

    for j, i in enumerate(levels[:-1]):
        radius = pow(2, i - 1)
        for p in [k for k, x in enumerate(Svec[j,:]) if x]:
            optimizedBuildLevel(p, j, radius, gram, Svec, Nvec, Pvec, Cvec)
        for p in [k for k, x in enumerate(Svec[j,:]) if not x]:
            optimizedBuildLevel(p, j, radius, gram, Svec, Nvec, Pvec, Cvec)

    # guaranteed to by an e-net of at least epsilon
    netIndices = [i for i, x in enumerate(Svec[lowestLvl,:]) if x]
    return points[netIndices], list(netIndices)

# from sklearn.metrics.pairwise import pairwise_distances
# x0   = np.array([0, 0])
# x1   = np.array([0, 0.51])
# x2   = np.array([0.51, 0])
# x3   = np.array([0.51, 0.51])
# x4   = np.array([0.5, 1])
# x5   = np.array([1, 0.5])
# x6   = np.array([1, 1])
# xs   = np.vstack((x0, x1, x2, x3, x4, x5, x6))
# gram = pairwise_distances(xs, metric='l2')
# gram = gram / np.max(gram)
# print(gram)
# print()
# gammaGram = gram[[0,1,5]]
# print(gammaGram)
# n1 = np.argsort(gammaGram, axis=0)[0]
# n2 = np.argmin(gammaGram, axis=0)
# print(n1)
# print(n2)
# for i in range(1):
#     print(greedyConstructEpsilonNetWithGram(xs, gram, 0.125))
#     print(hieracConstructEpsilonNet(xs, gram, 0.125))
#     print(optmizedHieracConstructEpsilonNet(xs, gram, 0.125))

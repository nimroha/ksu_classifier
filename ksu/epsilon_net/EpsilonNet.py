"""
Implementation of Algorithm 3 from [Near-optimal sample compression for nearest neighbors](https://www.cs.bgu.ac.il/~karyeh/condense-journal.pdf)

variable names sadly avoid normal convention to correspond to the paper notations
"""
import numpy as np
from math import log, ceil


def greedyConstructEpsilonNetWithGram(points, gram, epsilon):
    idx     = np.random.randint(0, len(points))
    net     = np.array(points[idx])
    netGram = np.array(np.expand_dims(gram[idx], axis=0))
    for i, p in enumerate(points): #iterate rows
        if np.min(netGram[:,i]) >= epsilon:
            net = np.vstack((net, p))
            netGram = np.vstack((netGram, gram[i]))

    return net

def collectPotentialNeighbors(point, i, N, P, C):
    return {C[r, i] for p in P[point, i] for r in N[p, i]}

def buildLevel(p, i, radius, gram, S, N, P, C):
    T = collectPotentialNeighbors(p, i, N, P, C)

    for r in T:
        if gram[r, p] < radius:
            P[p, i - 1] = set(r)
            return

    S[i - 1].add(p)
    [C[r, i].add(p) for r in P[p, i]]

    for r in T:
        if gram[r, p] < 4 * radius:
            N[p, i - 1].add(r)
            N[r, i - 1].add(p)




def hieracConstructEpsilonNet(points, gram, epsilon):
    n = len(points)
    levels = range(1, int(ceil(log(epsilon, 2))), -1)

    #arbitrary starting point
    startIdx = np.random.randint(0, n)

    #init S - nets
    S = {i:set() for i in levels}
    S[1].add(startIdx)

    #init P - parents
    P = {(i, 1): set(startIdx) for i in range(n)}

    #init N - neighbors
    N = {(startIdx, 1): set(startIdx)}

    #init C - covered #TODO is this init correct? should it be empty?
    C = {(startIdx, 1): set(np.unique(np.argwhere(gram[startIdx] < epsilon)).tolist())}

    for i in levels:
        radius = pow(2, i - 1)
        for p in S[i]:
            buildLevel(p, i, radius, gram, S, N, P, C)
        for p in set(range(n)) - S[i]:
            buildLevel(p, i, radius, gram, S, N, P, C)

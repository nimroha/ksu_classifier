import numpy as np


def greedyConstructEpsilonNetWithGram(points, gram, epsilon):
    idx     = np.random.randint(0,len(points))
    net     = np.array(points[idx])
    netGram = np.array(np.expand_dims(gram[idx], axis=0))
    for i, p in enumerate(points): #iterate rows
        if np.min(netGram[:,i]) >= epsilon:
            net = np.vstack((net, p))
            netGram = np.vstack((netGram, gram[i]))

    return net

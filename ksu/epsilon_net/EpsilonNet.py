import numpy as np

def minDistFromGroup(points, point, dist):
    minDist = np.inf
    for p in points:
        d = dist(p, point)
        if d < minDist:
            minDist = d

    return minDist
    
def greedyConstructEpsilonNet(points, dist, epsilon):
    net = [np.random.choice(points)]
    for p in points:
        if minDistFromGroup(net, p, dist) >= epsilon:
            net.append(p)

    return net

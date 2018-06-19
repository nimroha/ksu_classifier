import pytest
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from ksu.epsilon_net.EpsilonNet import greedyConstructEpsilonNetWithGram as e_net

x0   = np.array([0, 0])
x1   = np.array([0, 1])
x2   = np.array([1, 0])
x3   = np.array([1, 1])
xs   = np.vstack((x0, x1, x2, x3))
gram = pairwise_distances(xs, metric='l2')

@pytest.mark.parametrize('eps,shape', [(0.9, (4, 2)),
                                       (1.1, (2, 2)),
                                       (1.3, (2, 2)),
                                       (1.5, (1,2))])
def testGreedyConstruct(eps, shape):
    assert np.shape(e_net(xs, gram, eps)[0]) == shape

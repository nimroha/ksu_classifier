import numpy as np
import editdistance
from scipy.stats import wasserstein_distance

#TODO add more

def makeLn(n):
    return lambda a, b: np.linalg.norm(a - b, ord=n)

def editDistance(a, b):
    return editdistance.eval(a, b)

def earthMoverDistance(u, v, u_weights=None, v_weights=None):
    """
    Compute the first Wasserstein distance between two 1D distributions.
    :param u, v: (Array-like) Values observed in the (empirical) distribution.
    :param u_weights, v_weights: (Array-like, optional) Weight for each value. If left unspecified,
    each value is assigned the same weight. u_weights (resp. v_weights) must have the same length
    as u_values (resp. v_values). If the weight sum differs from 1,
    it must still be positive and finite so that the weights can be normalized to sum to 1.
    :return: (float) The computed distance between the distributions.
    """
    return wasserstein_distance(u, v, u_weights, v_weights)

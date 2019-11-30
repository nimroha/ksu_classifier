import numpy as np
import editdistance

from numba import jit

#TODO add more

def makeLn(n):
    return lambda a, b: np.linalg.norm(a - b, ord=n)

def editDistance(a, b):
    return editdistance.eval(a, b)

@jit(nopython=True)
def earthMoverDistance(u, v):
    """
    Compute the first Wasserstein distance between two 1D distributions.
    :param u, v: (Array-like) Values observed in the (empirical) distribution.
    :return: (float) The computed distance between the distributions.
    """
    # Calculate the CDFs of u and v
    u_cumweights = np.concatenate((np.array([0.0]), np.cumsum(u)))
    u_cdf = u_cumweights / u_cumweights[-1]

    v_cumweights = np.concatenate((np.array([0.0]), np.cumsum(v)))
    v_cdf = v_cumweights / v_cumweights[-1]

    return np.sum(np.abs(u_cdf - v_cdf))

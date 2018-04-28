import numpy as np

from math import sqrt, \
                 log2 as log


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

    secondTerm = (m * log(n) - log(delta)) / (n - m)

    thirdTerm = sqrt(((n * m * alpha * log(n)) / (n - m) - log(delta)) / (n - m))

    return firstTerm + secondTerm + thirdTerm

def  computeAlpha():
import numpy as np

#TODO add more

def l1(a, b):
    return np.linalg.norm(a - b, ord=1)

def l2(a, b):
    return np.linalg.norm(a - b, ord=2)
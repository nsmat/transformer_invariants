import numpy as np


def euclidean_distance(x, y):
    diff = x - y
    squ = (diff ** 2).sum()
    return np.sqrt(squ)

import numpy as np


def entropy(prob: np.array, logarithm):
    return (-prob * logarithm(prob)).sum(axis=-1)

import numpy as np


def entropy(prob: np.ndarray, logarithm):
    return (-prob * logarithm(prob)).sum(axis=-1)


def dmi(prob: np.ndarray, classes: np.ndarray, n: int):
    L = np.eye(n)[classes]
    O = prob.T / prob.shape[0]
    return -np.log(np.linalg.det(O @ L))

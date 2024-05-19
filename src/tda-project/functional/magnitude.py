import numpy as np


def magnitude(d: np.ndarray):
    d = np.exp(-d)
    return np.linalg.inv(d).sum()

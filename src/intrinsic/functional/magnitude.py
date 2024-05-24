import numpy as np
from scipy.spatial import distance_matrix


def magnitude(X: np.ndarray, distances: bool = False):
    d = X if distances else distance_matrix(X, X)
    d = np.exp(-d)
    return np.linalg.inv(d).sum()

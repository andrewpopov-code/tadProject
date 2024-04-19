import numpy as np
from utils.math import compute_unique_distances


def magnitude(X: np.array, distances: bool = False):
    if not distances:
        X = compute_unique_distances(X)

    d = np.exp(-X)
    return np.linalg.inv(d).sum()

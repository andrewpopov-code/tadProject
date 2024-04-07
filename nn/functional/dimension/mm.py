import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils import unique_points


def mm(X: np.array, k: int):
    # TODO: figure out if this is correct

    X = unique_points(X)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors()

    w = distances[:, -1]
    m1 = distances.mean(axis=1)
    return -m1 / (m1 - w)

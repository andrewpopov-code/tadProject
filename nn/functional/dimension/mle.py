import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import unique_points


def mle(X: np.array, k: int = 5):
    X = unique_points(X)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors()

    return 1 / np.log(np.expand_dims(distances[:, k - 1], 1) / distances[:, :k - 1]).mean()

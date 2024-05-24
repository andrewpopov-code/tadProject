import numpy as np
from sklearn.neighbors import NearestNeighbors

from .dimension import mle
from .kernel import gaussian_kernel_one
from src.intrinsic.utils.math import unit_ball_volume, mle_aggregate


def density(X: np.ndarray, k: int, dim: float, distances: bool = False) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed'if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()
    h = np.power(np.arange(1, k + 1), -1 / (dim + 4))
    return np.sum(1 / np.power(h, k) / k * gaussian_kernel_one(dist / h, dim, 1) / k, axis=-1)


def mean_density(X: np.ndarray, k: int, dim: float, distances: bool = False) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, ix = nn.kneighbors()
    rho = density(X, k, dim, distances)
    return np.arange(1, k + 1) / np.cumsum(1 / rho[ix], axis=-1)


def volume(X: np.ndarray, k: int, dim: float, distances: bool = False) -> np.ndarray:
    mu = mean_density(X, k, dim, distances)
    return np.arange(1, k + 1) / mu / (k - 1)


def quadratic_fit(X: np.ndarray, k: int, dim: float, distances: bool = False) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()
    vol = volume(X, k, dim, distances)
    y = vol / unit_ball_volume(dim) / np.power(dist, dim)

    return 5 * np.sum(np.square(dist) * (y - 1) * np.diff(dist, prepend=0), axis=-1) / (dist[:, -1] ** 5 - dist[:, 0] ** 5)


def curvature(X: np.ndarray, k: int, dim: float = None, distances: bool = False) -> np.ndarray:
    dim = dim or mle_aggregate(mle(X, k, distances))
    return -6 * (dim + 2) * quadratic_fit(X, k, dim, distances)

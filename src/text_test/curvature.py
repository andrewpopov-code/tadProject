import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma


def mle(X: np.ndarray, k: int = 5, distances: bool = False):
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()

    return (k - 1) / np.log(np.expand_dims(dist[:, -1], 1) / dist).sum(axis=-1)


def mle_aggregate(dim: np.ndarray):
    return 1 / np.mean(1 / dim)


def gaussian_kernel_one(x: float, n: float, sigma: float) -> float:
    return 1 / np.power(np.sqrt(2 * np.pi) * sigma, n) * np.exp(-x*x / 2 / sigma / sigma)


def unit_ball_volume(n: float) -> float:
    return np.sqrt(np.power(np.pi, n)) / gamma(n / 2 + 1)


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

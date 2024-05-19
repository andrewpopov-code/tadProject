import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseEstimator
from .dimension import DimensionEstimator
from functional.kernel import gaussian_kernel_one
from utils.math import unit_ball_volume, mle_aggregate


class CurvatureEstimator(BaseEstimator):
    def __init__(self, aggregate=np.mean, k: int = 5, treat_as_distances: bool = False, dim_est: DimensionEstimator = None):
        super().__init__(aggregate=aggregate)
        self.dist = treat_as_distances
        self.k = k
        self.dim_est = dim_est or DimensionEstimator('mle', mle_aggregate, distances=treat_as_distances)

    def fit_transform(self, X: np.ndarray, y=None):
        nn = NearestNeighbors(n_neighbors=self.k, metric='precomputed' if self.dist else 'minkowski').fit(X)
        dist, ix = nn.kneighbors()
        dim = self.dim_est.fit_transform(X)

        h = np.power(np.arange(1, self.k + 1), -1 / (dim + 4))
        rho = 1 / np.power(h, self.k) / self.k * gaussian_kernel_one(dist / h, dim, 1) / self.k
        mu = np.arange(1, self.k + 1) / np.cumsum(1 / rho[ix], axis=-1)
        vol = np.arange(1, self.k + 1) / mu / (self.k - 1)
        y = vol / unit_ball_volume(dim) / np.power(dist, dim)
        q = 5 * np.sum(np.square(dist) * (y - 1) * np.diff(dist, prepend=0), axis=-1) / (dist[:, -1] ** 5 - dist[:, 0] ** 5)
        c = -6 * (dim + 2) * q

        if self.aggregate is not None:
            return self.aggregate(c)
        return c

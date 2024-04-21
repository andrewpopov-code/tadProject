import numpy as np

from .base import BaseEstimator
from functional.homology import diagrams, betti, persistence_entropy, persistence_norm, mtd, rtd, cross_barcode, r_cross_barcode, pairwise_dist


class Barcode(BaseEstimator):
    def __init__(self, maxdim: int = 1, treat_as_distances: bool = False):
        super().__init__()
        self.maxdim = maxdim
        self.treat_as_distances = treat_as_distances

    def fit_transform(self, X: np.ndarray, y=None):
        return diagrams(X, maxdim=self.maxdim, distances=self.treat_as_distances)


class CrossBarcode(BaseEstimator):
    def __init__(self, maxdim: int = 1):
        super().__init__()
        self.maxdim = maxdim

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        return cross_barcode(X, y, maxdim=self.maxdim)


class RCrossBarcode(BaseEstimator):
    def __init__(self, maxdim: int = 1):
        super().__init__()
        self.maxdim = maxdim

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        return r_cross_barcode(X, y, maxdim=self.maxdim)


class MTD(BaseEstimator):
    def __init__(self, maxdim: int = 1):
        super().__init__()
        self.maxdim = maxdim

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        return mtd(X, y, maxdim=self.maxdim)


class RTD(BaseEstimator):
    def __init__(self, maxdim: int = 1):
        super().__init__()
        self.maxdim = maxdim

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        return rtd(X, y, maxdim=self.maxdim)


class BettiCurves(BaseEstimator):
    def __init__(self, n_bins: int = 100):
        super().__init__()
        self.n_bins = n_bins

    def fit_transform(self, X: np.ndarray, y=None):
        return betti(X, n_bins=self.n_bins)


class BettiDistance(BaseEstimator):
    def __init__(self, n_bins: int = 100):
        super().__init__()
        self.n_bins = n_bins

    def fit_transform(self, X: list[np.ndarray], y=None):
        bX = [betti(X[b], n_bins=self.n_bins) for b in range(len(X))]
        return pairwise_dist(bX)


class PersistenceEntropy(BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None):
        return persistence_entropy(X)


class PersistenceNorm(BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X, y=None):
        return persistence_norm(X)

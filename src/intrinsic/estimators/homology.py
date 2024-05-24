import numpy as np

from .base import BaseEstimator
from src.intrinsic.functional.homology import vr_diagrams, betti, persistence_entropy, total_persistence


class Barcode(BaseEstimator):
    def __init__(self, maxdim: int = 1, treat_as_distances: bool = False):
        super().__init__()
        self.maxdim = maxdim
        self.treat_as_distances = treat_as_distances

    def fit_transform(self, X: np.ndarray, y=None):
        return [vr_diagrams(X[b], maxdim=self.maxdim, distances=self.treat_as_distances) for b in range(X.shape[0])]


class BettiCurves(BaseEstimator):
    def __init__(self, n_bins: int = 100):
        super().__init__()
        self.n_bins = n_bins

    def fit_transform(self, X: list[list[np.ndarray]], y=None):
        return np.array([betti(X[b], n_bins=self.n_bins) for b in range(len(X))])


class PersistenceEntropy(BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit_transform(self, X: list[list[np.ndarray]], y=None):
        return np.array([persistence_entropy(X[b]) for b in range(len(X))])


class TotalPersistence(BaseEstimator):
    def __init__(self, q: float = 1):
        super().__init__()
        self.q = q

    def fit_transform(self, X: list[list[np.ndarray]], y=None):
        return np.array([total_persistence(X[b], self.q) for b in range(len(X))])

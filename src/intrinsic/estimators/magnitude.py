import numpy as np

from .base import BaseEstimator
from src.intrinsic.utils.math import compute_unique_distances
from src.intrinsic.functional.magnitude import magnitude


class MagnitudeEstimator(BaseEstimator):
    def __init__(self, treat_as_distances: bool = False):
        super().__init__()
        self.treat_as_distances = treat_as_distances

    def fit(self, X: np.ndarray, y=None):
        if not self.treat_as_distances:
            X = compute_unique_distances(X)
        return magnitude(X)

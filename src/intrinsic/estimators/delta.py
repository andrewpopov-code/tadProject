import numpy as np
from .base import BaseEstimator
from src.intrinsic.functional.delta import delta_hyperbolicity
from src.intrinsic.utils.math import compute_unique_distances


class DeltaHyperbolicity(BaseEstimator):
    def __init__(self, treat_as_distances: bool = False):
        super().__init__()
        self.treat_as_distances = treat_as_distances

    def fit_transform(self, X: np.ndarray, y=None):
        d = compute_unique_distances(X) if not self.treat_as_distances else X
        return delta_hyperbolicity(d)

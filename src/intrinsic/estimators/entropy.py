import numpy as np
from .base import BaseEstimator
from src.intrinsic.utils.math import entropy


class EntropyEstimator(BaseEstimator):
    def __init__(self, aggregate=np.mean, base: str = 'nat'):
        super().__init__(aggregate=aggregate)
        self.base = np.e if base == 'nat' else 2 if base == 'bits' else 10

    def fit_transform(self, X, y=None):
        if self.aggregate is not None:
            return self.aggregate(entropy(X, base=self.base))
        return entropy(X, base=self.base)

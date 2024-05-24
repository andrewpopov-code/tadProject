import numpy as np
from .base import BaseEstimator
from src.intrinsic.functional import entropy


class EntropyEstimator(BaseEstimator):
    def __init__(self, aggregate=np.mean, base: str = 'nat'):
        super().__init__(aggregate=aggregate)
        self.log = np.log if base == 'nat' else np.log2 if base == 'bits' else np.log10

    def fit_transform(self, X, y=None):
        if self.aggregate is not None:
            return self.aggregate(entropy(X, logarithm=self.log))
        return entropy(X, logarithm=self.log)

import numpy as np

from sklearn.base import BaseEstimator as sklearnBase
from sklearn.linear_model import LinearRegression, Ridge as RidgeRegression


class BaseEstimator(sklearnBase):
    def __init__(self, aggregate=np.mean, slope_estimator_penalty: str = None, eps: float = None):
        self.aggregate = aggregate

        if slope_estimator_penalty is None:
            self.slope_estimator = LinearRegression
        elif slope_estimator_penalty == "l2":
            self.slope_estimator = RidgeRegression
        else:
            raise ValueError("Slope estimator penalty should be 'None' or 'l2'")

        self.eps = eps

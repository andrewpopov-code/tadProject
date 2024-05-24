from .base import BaseEstimator


class DimensionEstimator(BaseEstimator):
    def __init__(self, name: str, aggregate=np.mean, slope_estimator_penalty: str = None, eps: float = None, **kwargs):
        super().__init__(aggregate, slope_estimator_penalty, eps)
        self.est = name
        self.kwargs = kwargs

    def fit_transform(self, X: np.ndarray, y=None):
        if self.est == 'mom':
            est = mm(X, **self.kwargs)
        elif self.est == 'mle':
            est = mle(X, **self.kwargs)
        elif self.est == 'ols':
            est = ols(X, slope_estimator=self.slope_estimator, **self.kwargs)
        elif self.est == 'two_nn':
            est = two_nn(X)
        elif self.est == 'pca':
            est = pca(X, **self.kwargs)
        elif self.est == 'local_pca':
            est = local_pca(X, **self.kwargs)
        else:
            raise ValueError(f"Not found an appropriate estimator for name: '{self.est}'")

        if self.aggregate:
            return self.aggregate(est)
        return est

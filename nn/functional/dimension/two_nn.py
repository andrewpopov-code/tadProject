import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

from utils import unique_points


def two_nn(X: np.array):
    X = unique_points(X)
    n = X.shape[0]

    nn = NearestNeighbors(n_neighbors=2).fit(X)
    distances, _ = nn.kneighbors()

    mu = np.sort(distances[:, 1] / distances[:, 0])
    cdf = np.linspace(0, 1 - 1/n, n)
    lr = LinearRegression(fit_intercept=False)
    return lr.fit(np.log(mu), -np.log(1 - cdf)).coef_[0, 0]

import numpy as np
from sklearn.decomposition import PCA

from utils import unique_points


def pca(X: np.array, explained_variance: float = 0.95):
    X = unique_points(X)
    pca = PCA(n_components=explained_variance)
    pca.fit(X)

    return pca.n_components_

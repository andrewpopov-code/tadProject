import numpy as np
import torch
from scipy.spatial import distance_matrix


def image_to_cloud(X: np.ndarray, channel_first: bool = False):
    if channel_first:
        X = X.T
    return X.reshape(X.shape[0] * X.shape[1], X.shape[2])


def unique_points(x: np.ndarray):
    return np.unique(x, axis=-2)


def compute_unique_distances(X: np.ndarray):
    X = unique_points(X)
    return distance_matrix(X, X)


def mle_aggregate(dim: np.ndarray):
    return 1 / np.mean(1 / dim)


def mle_aggregate_torch(dim: torch.Tensor):
    return 1 / torch.mean(1 / dim)

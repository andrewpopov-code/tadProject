import torch
import numpy as np


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    return torch.cdist(x, y)


def image_distance(x: torch.Tensor):
    """
    :param x: (B) x H x W x C
    :return: distances between pixels
    """
    if x.ndim == 3:
        x = x.flatten(0, 1)
    else:
        x = x.flatten(1, 2)

    return euclidean_dist(x, x)


def unique_points(x: np.array):
    return np.unique(x, axis=-2)


def compute_unique_distances(x: torch.Tensor):
    if x.ndim == 3:  # H x W x C
        x = x.flatten(0, 1)
    x = x.detach().numpy()
    x = torch.tensor(unique_points(x))  # N x C
    return euclidean_dist(x, x)

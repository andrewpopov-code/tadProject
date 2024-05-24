import numpy as np
import torch
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist


def min_max_prod(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.minimum(A.reshape(*A.shape, 1), B.reshape(1, *B.shape)).max(axis=-2)


def min_max_prod_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.minimum(x.unsqueeze(-1), y.unsqueeze(0)), dim=-2).values


def gromov_product(dist: np.ndarray, p: int) -> np.ndarray:
    row, col = np.expand_dims(dist[p, :], axis=-1), np.expand_dims(dist[:, p], axis=0)
    return (row + col - dist) / 2


def delta_hyperbolicity(X: np.ndarray, distances: bool = False, p: int = 0) -> float:
    d = X if distances else distance_matrix(X, X)
    A = gromov_product(d, p)
    maxmin = min_max_prod(A, A)

    return np.max(np.max(maxmin - A, axis=-1), axis=-1)


def relative_delta_hyperbolicity(X: np.ndarray, distances: bool = False) -> float:
    d = delta_hyperbolicity(X, distances)
    diam = X.max() if distances else pdist(X).max()
    return d / diam


def delta_hyperbolicity_torch(x: torch.Tensor, distances: bool = False) -> torch.Tensor:
    p = 0
    d = x if distances else torch.cdist(x, x)
    row, col = d[:, p, :].unsqueeze(0), d[:, :, p].unsqueeze(-1)
    A = (row + col - d) / 2

    return torch.max(torch.max(min_max_prod_torch(A, A) - A, dim=-1).values, dim=-1).values


def mobius_addition(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    num = (1 + 2 * c * np.sum(x * y) + c * np.sum(y * y)) * x + (1 - c * np.sum(x * x)) * y
    den = 1 + 2 * c * np.sum(x * y) + c * c * np.sum(y * y) * np.sum(x * x)

    return num / den


def hyperbolic_distance(x: np.ndarray, y: np.ndarray, c: float) -> float:
    return 2 * np.arctanh(np.sqrt(c) * np.linalg.norm(mobius_addition(-x, y, c))) / np.sqrt(c)


def exponential_map(x: np.ndarray, c: float, v: np.ndarray) -> np.ndarray:
    return mobius_addition(
        x,
        np.tanh(np.sqrt(c) * conformal(x, c) * np.linalg.norm(v) / 2) * v / np.sqrt(c) / np.linalg.norm(v),
        c
    )


def exponential_map_torch(x: torch.Tensor, c: float, v: torch.Tensor) -> torch.Tensor:
    return mobius_addition_torch(
        x.unsqueeze(-2),
        torch.tanh(torch.sqrt(c) * conformal_torch(x, c).unsqueeze(-1) * torch.norm(v, dim=-1) / 2).unsqueeze(-1) * v / torch.sqrt(c) / torch.norm(v, dim=-1).unsqueeze(-1),
        c
    )


def logarithmic_map(x: np.ndarray, c: float, y: np.ndarray) -> np.ndarray:
    ma = mobius_addition(-x, y, c)
    norm = np.linalg.norm(ma)
    return 2 * np.arctanh(np.sqrt(c) * norm) * ma / norm / np.sqrt(c) / conformal(x, c)


def hyp_ave(X: np.ndarray, c: float) -> np.ndarray:
    gamma = conformal(X, c)
    return np.sum(gamma * X) / np.sum(gamma)


def mobius_addition_torch(x: torch.Tensor, y: torch.Tensor, c: [torch.Tensor, float]) -> torch.Tensor:
    x = x.unsqueeze(-2)
    num = (1 + 2 * c * torch.sum(x * y, dim=-1, keepdim=True) + c * x * torch.sum(y * y, dim=-1, keepdim=True)) + (1 - c * torch.sum(x * x, dim=-1, keepdim=True)) * y
    den = 1 + 2 * c * torch.sum(x * y, dim=-1) + c * c * torch.sum(x * x, dim=-1) * torch.sum(y * y, dim=-1)

    return num / den.unsqueeze(-1)


def conformal(x: np.ndarray, c: float) -> [float, np.ndarray]:
    return 2 / (1 - c * np.sum(x * x, axis=-1))


def conformal_torch(x: torch.Tensor, c: [torch.Tensor, float]) -> torch.Tensor:
    return 2 / (1 - c * torch.sum(x * x, dim=-1))


def c(x: np.ndarray, distances: bool = False):
    return np.square(0.144 / delta_hyperbolicity(x, distances))


def c_torch(x: torch.Tensor, distances: bool = False):
    return torch.square(0.144 / delta_hyperbolicity_torch(x, distances))

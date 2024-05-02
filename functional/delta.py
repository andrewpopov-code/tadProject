import numpy as np
import torch
from scipy.spatial import distance_matrix


def min_max_prod(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.minimum(A.reshape(*A.shape, 1), B.reshape(1, *B.shape)).max(axis=-2)


def min_max_prod_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.minimum(x.unsqueeze(-1), y.unsqueeze(0)), dim=-2).values


def delta_hyperbolicity(X: np.ndarray, distances: bool = True) -> float:
    p = 0
    d = X if distances else distance_matrix(X, X)
    row, col = d[p, :].reshape(X.shape[0], 1), d[:, p].reshape(1, X.shape[0])
    A = 0.5 * (row + col - d)
    maxmin = min_max_prod(A, A)

    return np.max(np.max(maxmin - A, axis=-1), axis=-1)


def delta_hyperbolicity_torch(x: torch.Tensor, distances: bool = True) -> torch.Tensor:
    p = 0
    d = x if distances else torch.cdist(x, x)
    row, col = d[:, p, :].unsqueeze(0), d[:, :, p].unsqueeze(-1)
    A = (row + col - d) / 2

    return torch.max(torch.max(min_max_prod_torch(A, A) - A, dim=-1).values, dim=-1).values


def mobius_addition(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    num = (1 + 2 * c * np.sum(x * y) + c * np.sum(y * y)) * x + (1 - c * np.sum(x * x)) * y
    den = 1 + 2 * c * np.sum(x * y) + c * c * np.sum(y * y) * np.sum(x * x)

    return num / den


def mobius_addition_torch(x: torch.Tensor, y: torch.Tensor, c: [torch.Tensor, float]) -> torch.Tensor:
    num = (1 + 2 * c * torch.sum(x * y, dim=-1) + c * torch.sum(y * y, dim=-1)) * x + (1 - c * torch.sum(x * x, dim=-1)) * y
    den = 1 + 2 * c * torch.sum(x * y, dim=-1) + c * c * torch.sum(y * y, dim=-1) * torch.sum(x * x, dim=-1)

    return num / den


def conformal(x: np.ndarray, c: float) -> np.ndarray:
    return 2 / (1 - c * np.sum(x * x))


def conformal_torch(x: torch.Tensor, c: [torch.Tensor, float]) -> torch.Tensor:
    return 2 / (1 - c * torch.sum(x * x, dim=-1))

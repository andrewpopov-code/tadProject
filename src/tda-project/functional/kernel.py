import numpy as np
from scipy.special import gamma, jv
from scipy.spatial import distance_matrix
from .homology import landscapes, drop_inf


def landscape_kernel(diagX: list[np.ndarray], diagY: list[np.ndarray], n_points: int = 100) -> float:
    lX, lY = landscapes(drop_inf(diagX), n_points), landscapes(drop_inf(diagY), n_points)
    return np.sqrt(np.square(lX - lY).sum())


def scale_kernel(F: list[np.ndarray], G: list[np.ndarray], sigma: float) -> float:
    F, G = np.vstack(drop_inf(F)), np.vstack(drop_inf(G))

    diff = np.exp(-np.square(distance_matrix(F, G)).sum(dim=-1) / 8 / sigma) - np.exp(-np.square(F - G.reshape(-1, 1, 2)[:, :, ::-1]).sum(dim=-1) / 8 / sigma)
    return diff.sum() / (8 * np.pi * sigma)


def heat_kernel(F: np.ndarray, G: np.ndarray, t: float) -> float:
    return np.exp(-np.square(distance_matrix(F, G)).sum(dim=-1) / 4 / t).sum() / (4 * np.pi * t)


def multinomial_kernel(P: np.ndarray, Q: np.ndarray, t: float) -> float:
    d = P.size - 1
    return np.power(4 * np.pi * t, -d / 2) * np.exp(-np.square(np.arccos(np.sqrt(P * Q).sum())) / t)


def bessel_kernel2(F: np.ndarray, G: np.ndarray) -> float:
    n = F.shape[1]
    k0 = np.power(np.pi, n / 2) / gamma(n / 2 + 1)
    return np.exp(-2 * np.pi * np.square(distance_matrix(F, G) / np.sqrt(2 * n))) * k0


def bessel_kernel3(F: np.ndarray, G: np.ndarray, c: float = 2) -> float:
    n = F.shape[1]
    r = distance_matrix(F, G)
    return np.power(c / 2, n / 2) * jv(n / 2, np.pi * c * r) / np.power(r, n / 2)


def bessel_kernel(x: np.ndarray, y: np.ndarray, v: float) -> float:
    norm = np.linalg.norm(x - y)
    return np.power(2 / norm, v) * gamma(v + 1) * jv(v, norm)


def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(x * y)


def poly_kernel(x: np.ndarray, y: np.ndarray, theta: float, d: float) -> float:
    return np.power(np.sum(x * y) + theta, d)


def sigmoid_kernel(x: np.ndarray, y: np.ndarray, eta: float, theta: float) -> float:
    return np.tanh(eta * np.sum(x * y) + theta)


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    return 1 / np.power(np.sqrt(2 * np.pi) * sigma, x.size) * np.exp(-np.linalg.norm(x - y) / 2 / sigma / sigma)


def gaussian_kernel_one(x: float, n: float, sigma: float) -> float:
    return 1 / np.power(np.sqrt(2 * np.pi) * sigma, n) * np.exp(-x*x / 2 / sigma / sigma)

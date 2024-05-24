import torch
from .vr import VietorisRips
from src.intrinsic.utils.math import batch_select, neighbors, set_ops


def pq_loss(dgms_tensorXb: torch.Tensor, dgms_tensorXd: torch.Tensor, left: float, right: float):
    left = torch.pow(dgms_tensorXd - dgms_tensorXb, left)
    right = torch.pow((dgms_tensorXb + dgms_tensorXd) / 2, right)
    return torch.sum(left * right)


def signature_loss(X: torch.Tensor, Z: torch.Tensor):  # FIXME
    _, gens_tensorXd = VietorisRips.apply(X)[2:]
    _, gens_tensorZd = VietorisRips.apply(Z)[2:]
    distXX = torch.norm(batch_select(X, gens_tensorXd[:, :, :, 1]) - batch_select(X, gens_tensorXd[:, :, :, 0]))
    distZX = torch.norm(batch_select(Z, gens_tensorXd[:, :, :, 1]) - batch_select(Z, gens_tensorXd[:, :, :, 0]))
    distZZ = torch.norm(batch_select(Z, gens_tensorZd[:, :, :, 1]) - batch_select(Z, gens_tensorZd[:, :, :, 0]))
    distXZ = torch.norm(batch_select(X, gens_tensorZd[:, :, :, 1]) - batch_select(X, gens_tensorZd[:, :, :, 0]))

    return torch.sqrt(torch.square(distXX - distZX).sum()) / 2 + torch.sqrt(torch.square(distXZ - distZZ).sum()) / 2


def ph_dimension_loss(X: torch.Tensor):
    n = torch.log(torch.arange(2, X.shape[1] + 1, X.shape[1] // 10))
    e = torch.zeros((X.shape[0], n.shape[0]))
    for ni in range(2, X.shape[1] + 1, X.shape[1] // 10):
        dgms_tensorXb, dgms_tensorXd = VietorisRips.apply(X[:, :ni])[:2]
        dgms_tensorXb = torch.nan_to_num(dgms_tensorXb, 0, 0)
        dgms_tensorXd = torch.nan_to_num(dgms_tensorXd, 0, 0)

        e[:, (ni - 1) // 10] = torch.sum(torch.abs(dgms_tensorXd[:, 0] - dgms_tensorXb[:, 0]), dim=1)
    return torch.sum(
        torch.abs(
            (n.shape[0] * torch.sum(n * e, dim=1) - n.sum() * e.sum(dim=1)) / (X.shape[1] * torch.square(n).sum() - torch.square(n.sum()))
        )
    )


def stress_loss(X: torch.Tensor, Z: torch.Tensor):
    dX, dZ = torch.cdist(X, X), torch.cdist(Z, Z)
    return torch.sqrt(torch.square(dX - dZ).sum() / torch.square(dZ).sum())


def trustworthiness_loss(X: torch.Tensor, Z: torch.Tensor, k: int):
    dX, iX = neighbors(X, k)
    dZ, iZ = neighbors(Z, k)

    onlyX = set_ops(iX, iZ)[3]
    return onlyX.sum()


def continuity_loss(X: torch.Tensor, Z: torch.Tensor, k: int):
    return trustworthiness_loss(Z, X, k)


def intersection_loss(X: torch.Tensor, Z: torch.Tensor, k: int):
    dX, iX = neighbors(X, k)
    dZ, iZ = neighbors(Z, k)

    return 1 - torch.sum(set_ops(iX, iZ)[1] / k) / X.shape[1]

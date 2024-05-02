import torch
from .vr import VietorisRips
from functional.homology import drop_inf


def pq_loss(dgms_tensorXb: torch.Tensor, dgms_tensorXd: torch.Tensor, left: float, right: float):
    left = torch.pow(dgms_tensorXd - dgms_tensorXb, left)
    right = torch.pow((dgms_tensorXb + dgms_tensorXd) / 2, right)
    return torch.sum(left * right)


def signature_loss(X: torch.Tensor, Z: torch.Tensor):  # FIXME
    gens_tensorXb, gens_tensorXd = VietorisRips.apply(X)[2:]
    gens_tensorZb, gens_tensorZd = VietorisRips.apply(Z)[2:]
    dX, dZ = torch.cdist(X, X), torch.cdist(Z, Z)
    Lxz = Lzx = 0

    for b in range(X.shape[0]):
        for dim in range(gens_tensorXb.shape[1]):
            gensb, gensd = drop_inf(gens_tensorXb[b, dim, :dim + 1]), drop_inf(gens_tensorXd[b, dim])
            distXX = dX[b, gensb, gensd]
            distZX = dZ[b, gensb, gensd]
            Lxz += torch.square(distXX - distZX)

            gensb, gensd = drop_inf(gens_tensorZb[b, dim, :dim + 1]), drop_inf(gens_tensorZd[b, dim])
            distZZ = dZ[b, gensb, gensd]
            distXZ = dX[b, gensb, gensd]
            Lzx += torch.square(distXZ - distZZ)

    Lxz = torch.sqrt(Lxz) / 2
    Lzx = torch.sqrt(Lzx) / 2
    return Lxz + Lzx


def ph_dimension_loss(X: torch.Tensor):
    n = torch.log(torch.arange(1, X.shape[1] + 1, X.shape[1] // 10))
    e = torch.zeros((X.shape[0], n.shape[0]))
    for ni in range(1, X.shape[1] + 1, X.shape[1] // 10):
        dgms_tensorXb, dgms_tensorXd = VietorisRips.apply(X[:, :ni])[:2]
        dgms_tensorXb = torch.nan_to_num(dgms_tensorXb, 0, 0)
        dgms_tensorXd = torch.nan_to_num(dgms_tensorXd, 0, 0)

        e[:, (ni - 1) // 10] = torch.sum(dgms_tensorXd[:, 0] - dgms_tensorXb[:, 0], dim=1)
    return torch.sum(
        torch.abs(
            (n.shape[0] * torch.sum(n * e, dim=1) - n.sum() * e.sum(dim=1)) / (X.shape[1] * torch.square(n).sum() - torch.square(n.sum()))
        )
    )

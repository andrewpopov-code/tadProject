import numpy as np
import torch
from torch.nn.functional import normalize
from src.intrinsic.functional.homology import vr_diagrams, drop_inf
from src.intrinsic.utils.math import diagrams_to_tensor, gens_to_tensor, boundary_matrix, batch_select


def filtration_return(ret: list):
    dgms, gens = [drop_inf(r[0]) for r in ret], [[r[1][0], *r[1][1]] for r in ret]
    dgms_tensor, gens_tensor = diagrams_to_tensor(dgms, requires_grad=True), gens_to_tensor(gens, fill_value=0)

    gens_birth, gens_death = gens_tensor[:, :, :, :-2], gens_tensor[:, :, :, -2:]
    return dgms_tensor[:, :, :, 0], dgms_tensor[:, :, :, 1], gens_birth, gens_death


def grad_subroutine(X: torch.Tensor, gens: torch.Tensor) -> torch.Tensor:
    boundary1 = boundary_matrix(gens[:, :, :, :1].numpy(force=True), np.arange(X.shape[1]))
    boundary0 = boundary_matrix(gens[:, :, :, 1:].numpy(force=True), np.arange(X.shape[1]))
    norm = normalize(batch_select(X, gens[:, :, :, 1]) - batch_select(X, gens[:, :, :, 0]), dim=-1).unsqueeze(-2)
    J0 = -norm * torch.tensor(boundary0).unsqueeze(-1)
    J1 = norm * torch.tensor(boundary1).unsqueeze(-1)
    return J0 + J1


def cross_grad_subroutine(X: torch.Tensor, Y: torch.Tensor, gens: torch.Tensor) -> torch.Tensor:
    Z = torch.cat((X, Y), dim=-2)
    boundary1 = boundary_matrix(gens[:, :, :, :1].numpy(force=True), np.arange(Z.shape[1] + 1))
    boundary0 = boundary_matrix(gens[:, :, :, 1:].numpy(force=True), np.arange(Z.shape[1] + 1))

    XX = gens % X.shape[0]
    X_normed = normalize(batch_select(Z, XX[:, :, :, 1]) - batch_select(Z, XX[:, :, :, 0]), dim=-1)
    YY = gens % X.shape[0] + X.shape[0]
    Y_normed = normalize(batch_select(Z, YY[:, :, :, 1]) - batch_select(Z, YY[:, :, :, 0]), dim=-1)
    Xnorm = torch.norm(batch_select(Z, XX[:, :, :, 1]) - batch_select(Z, XX[:, :, :, 0]))
    Ynorm = torch.norm(batch_select(Z, YY[:, :, :, 1]) - batch_select(Z, YY[:, :, :, 0]))
    JYY = ((Xnorm <= Ynorm) * X_normed + (X_normed > Y_normed) * Y_normed) * torch.all((X.shape[1] <= gens) & (gens < 2 * X.shape[1]), dim=-1).unsqueeze(-1)
    JXXY = X_normed * torch.all((X.shape[1] > gens[:, :, :, :1]) & (gens[:, :, :, 1:] < 2 * X.shape[1]), dim=-1).unsqueeze(-1)

    J0 = -(JYY + JXXY).unsqueeze(-2) * torch.tensor(boundary0).unsqueeze(-1)
    J1 = (JYY + JXXY).unsqueeze(-2) * torch.tensor(boundary1).unsqueeze(-1)
    return J0 + J1


class VietorisRips(torch.autograd.Function):
    @staticmethod
    def forward(X: torch.Tensor):
        ret = [vr_diagrams(X[b].numpy(force=True), gens=True) for b in range(X.shape[0])]
        return filtration_return(ret)

    @staticmethod
    def backward(ctx, grad_output_birth, grad_output_death, *args):
        shape, gens_birth, gens_death, X = ctx.saved_tensors
        Jb = grad_subroutine(X, gens_birth) * grad_output_birth.unsqueeze(-1).unsqueeze(-1)
        Jd = grad_subroutine(X, gens_death) * grad_output_death.unsqueeze(-1).unsqueeze(-1)

        return (Jb + Jd).sum(dim=1).sum(dim=1)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(torch.tensor(inputs[0].shape), *outputs[2:], inputs[0])


class CrossBarcode(torch.autograd.Function):
    @staticmethod
    def forward(X: torch.Tensor, Y: torch.Tensor):
        XX, YY = torch.cdist(X, X), torch.cdist(Y, Y)
        inf_block = torch.triu(torch.full_like(XX, torch.inf), 1) + XX
        M = torch.cat([
            torch.cat([XX, inf_block.transpose(1, 2), torch.zeros((XX.shape[0], XX.shape[1], 1))], dim=-1),
            torch.cat([inf_block, torch.minimum(XX, YY), torch.full((XX.shape[0], XX.shape[1], 1), torch.inf)], dim=-1),
            torch.cat([torch.zeros((XX.shape[0], 1, XX.shape[1])), torch.full((XX.shape[0], 1, XX.shape[1]), torch.inf), torch.zeros((XX.shape[0], 1, 1))], dim=-1)
        ], dim=1)
        ret = [vr_diagrams(M[b].numpy(force=True), distances=True, gens=True) for b in range(X.shape[0])]
        return filtration_return(ret)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(*outputs[2:], *inputs)

    @staticmethod
    def backward(ctx, grad_output_birth, grad_output_death, *args):
        gens_birth, gens_death, X, Y = ctx.saved_tensors
        gens_birth.sort()
        gens_death.sort()
        Jb = cross_grad_subroutine(X, Y, gens_birth) * grad_output_birth.unsqueeze(-1).unsqueeze(-1)
        Jd = cross_grad_subroutine(X, Y, gens_death) * grad_output_death.unsqueeze(-1).unsqueeze(-1)

        J = (Jb + Jd).sum(dim=1).sum(dim=1)
        return J[:, :X.shape[0]], J[:, X.shape[0]:-1]

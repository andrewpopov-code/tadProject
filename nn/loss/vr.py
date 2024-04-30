import numpy as np
import torch
from functional.homology import diagrams, drop_inf
from utils.math import diagrams_to_tensor


def filtration_return(ctx, ret: list):
    dgms, gens = [drop_inf(r[0]) for r in ret], [[r[1][0], *r[1][1]] for r in ret]
    dgms_tensor, gens_tensor = diagrams_to_tensor(dgms, requires_grad=True), diagrams_to_tensor(gens, fill_value=-1)

    gens_birth, gens_death = gens_tensor[:, :, :, :-2], gens_tensor[:, :, :, -2:]
    ctx.save_for_backward(gens_birth, gens_death)

    return dgms_tensor[:, :, :, 0], dgms_tensor[:, :, :, 1], gens_birth, gens_death


def boundary_matrix(gens: np.ndarray, x: np.ndarray):
    return np.logical_or.reduce(gens.reshape(*gens.shape, 1) == x.reshape(1, -1), axis=3).astype(int)


class VietorisRips(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor):
        # Return birth and death times separately
        ctx.save_for_backward(torch.tensor(X.shape))
        ret = [diagrams(X[b].numpy(force=True), gens=True) for b in range(X.shape[0])]
        return filtration_return(ctx, ret)

    @staticmethod
    def backward(ctx, grad_output_birth, grad_output_death):
        shape, gens_birth, gens_death = ctx.saved_tensors
        gens_birth, gens_death = gens_birth.numpy(force=True), gens_death.numpy(force=True)
        x = np.arange(shape[1])
        Jb, Jd = boundary_matrix(gens_birth, x), boundary_matrix(gens_death, x)

        return torch.sum(
            torch.tensor(Jb) * grad_output_birth.unsqueeze(-1) + torch.tensor(Jd) * grad_output_death.unsqueeze(-1),
            dim=1
        ).sum(dim=1).unsqueeze(-1).expand(-1, -1, shape[-1])


class CrossBarcode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, Y: torch.Tensor):
        XX, YY = torch.cdist(X, X), torch.cdist(Y, Y)
        inf_block = torch.triu(torch.full_like(XX, torch.inf), 1) + XX
        M = torch.cat([
            torch.cat([torch.zeros_like(XX), inf_block.transpose(1, 2), torch.zeros((XX.shape[0], XX.shape[1], 1))], dim=-1),
            torch.cat([inf_block, torch.minimum(XX, YY), torch.full((XX.shape[0], XX.shape[1], 1), torch.inf)], dim=-1),
            torch.cat([torch.zeros((XX.shape[0], 1, XX.shape[1])), torch.full((XX.shape[0], 1, XX.shape[1]), torch.inf), torch.zeros((XX.shape[0], 1, 1))], dim=-1)
        ], dim=1)
        ctx.save_for_backward(torch.tensor(X.shape), torch.tensor(Y.shape))
        ret = [diagrams(M[b].numpy(force=True), distances=True, gens=True) for b in range(X.shape[0])]
        return filtration_return(ctx, ret)

    @staticmethod
    def backward(ctx, grad_output_birth, grad_output_death):
        shapeX, shapeY, gens_birth, gens_death = ctx.saved_tensors
        gens_birth, gens_death = gens_birth.numpy(force=True), gens_death.numpy(force=True)
        x, y = np.arange(shapeX[1]), shapeX[1] + np.arange(shapeY[1])
        JbX, JdX = boundary_matrix(gens_birth, x), boundary_matrix(gens_death, x)
        JbY, JdY = boundary_matrix(gens_birth, y), boundary_matrix(gens_death, y)

        return (
            torch.sum(
                torch.tensor(JbX) * grad_output_birth.unsqueeze(-1) + torch.tensor(JdX) * grad_output_death.unsqueeze(-1),
                dim=1
            ).sum(dim=1).unsqueeze(-1).expand(-1, -1, shapeX[-1]),
            torch.sum(
                torch.tensor(JbY) * grad_output_birth.unsqueeze(-1) + torch.tensor(JdY) * grad_output_death.unsqueeze(-1),
                dim=1
            ).sum(dim=1).unsqueeze(-1).expand(-1, -1, shapeY[-1])
        )

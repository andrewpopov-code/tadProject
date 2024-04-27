from typing import Any

import torch
from scipy.spatial import distance_matrix
from gph import ripser_parallel


class SignatureLoss(torch.autograd.Function):
    # TODO: input persistence pairings (grad output is computed wrt them)
    @staticmethod
    def forward(ctx, X: torch.Tensor, Z: torch.Tensor):
        dX = torch.tensor([distance_matrix(X[b].detach().numpy(), X[b].detach().numpy()) for b in range(X.shape[0])])
        dZ = torch.tensor([distance_matrix(Z[b].detach().numpy(), Z[b].detach().numpy()) for b in range(X.shape[0])])
        retX, retZ = ripser_parallel(X.detach().numpy(), return_generators=True), ripser_parallel(Z.detach().numpy(), return_generators=True)
        distXX = dX[retX.gens[0][:, 1], retX.gens[0][:, 2]]
        distZZ = dZ[retZ.gens[0][:, 1], retZ.gens[0][:, 2]]
        distXZ = dX[retZ.gens[0][:, 1], retZ.gens[0][:, 2]]
        distZX = dZ[retX.gens[0][:, 1], retX.gens[0][:, 2]]

        ctx.save_for_backward(distXX, distZX)

        Lxz = torch.norm(distXX - distZX) / 2
        Lzx = torch.norm(distXZ - distZZ) / 2
        return Lxz + Lzx

    @staticmethod
    def backward(ctx: Any, grad_output: Any):
        distXX, distZX = ctx.saved_tensors
        return (distXX - distZX) * grad_output

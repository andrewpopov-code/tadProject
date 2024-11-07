import numpy as np
import torch
from src.intrinsic.functional.kernel import rbf_kernel_torch
from src.intrinsic.utils.math import matrix_power_torch


class Entropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, sx: float, alpha: float):
        K = rbf_kernel_torch(X, sx)
        A = K / K.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        E = torch.linalg.eigvals(A).real
        ctx.save_for_backward(A, E, alpha)
        return 1 / (1 - alpha) * np.log2(np.power(E, alpha).sum(axis=-1))

    @staticmethod
    def backward(ctx, *grad_outputs):
        A, E, alpha = ctx.saved_tensors
        P = matrix_power_torch(A, alpha - 1)
        return alpha / (1 - alpha) * P / torch.pow(E, alpha).sum(dim=-1, keepdim=True)


class MutualEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, Y: torch.Tensor, sx: float, sy: float, alpha: float):
        KX, KY = rbf_kernel_torch(X, sx), rbf_kernel_torch(Y, sy)
        A = KX * KY
        A = A / A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        E = torch.linalg.eigvals(A).real
        ctx.save_for_backward(KX, KY, E, alpha)
        return 1 / (1 - alpha) * np.log2(np.power(E, alpha).sum(axis=-1))


    @staticmethod
    def backward(ctx, *grad_outputs):
        KX, KY, E, alpha = ctx.saved_tensors
        lhs = matrix_power_torch(KX * KY, alpha - 1) * KY / torch.pow(E, alpha).sum(dim=-1, keepdim=True)
        tr = (KX * KY).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        rhs = torch.eye(KY.shape[1]).unsqueeze(0) * KY / tr
        return alpha / (1 - alpha) * (lhs - rhs)

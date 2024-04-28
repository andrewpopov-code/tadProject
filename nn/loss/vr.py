import numpy as np
import torch
from functional.homology import diagrams, drop_inf
from utils.math import diagrams_to_tensor


class VietorisRips(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor):
        # Return birth and death times separately
        ret = [diagrams(X[b].numpy(force=True), gens=True) for b in range(X.shape[0])]
        dgms, gens = [drop_inf(r[0]) for r in ret], [[r[1][0], r[1][1][0]] for r in ret]
        dgms_tensor, gens_tensor = diagrams_to_tensor(dgms, requires_grad=True), diagrams_to_tensor(gens, fill_value=-1)

        gens_birth, gens_death = gens_tensor[:, :, :, :-2], gens_tensor[:, :, :, -2:]
        ctx.save_for_backward(torch.tensor(X.shape), gens_birth, gens_death)

        return dgms_tensor[:, :, :, 0], dgms_tensor[:, :, :, 1]

    @staticmethod
    def backward(ctx, grad_output_birth, grad_output_death):
        shape, gens_birth, gens_death = ctx.saved_tensors
        gens_birth, gens_death = gens_birth.numpy(force=True), gens_death.numpy(force=True)
        x = np.arange(shape[1])
        Jb = np.logical_or.reduce(gens_birth.reshape(*gens_birth.shape, 1) == x.reshape(1, -1), axis=3).astype(int)
        Jd = np.logical_or.reduce(gens_death.reshape(*gens_death.shape, 1) == x.reshape(1, -1), axis=3).astype(int)

        return torch.sum(
            torch.tensor(Jb) * grad_output_birth.unsqueeze(-1) + torch.tensor(Jd) * grad_output_death.unsqueeze(-1),
            dim=1
        ).sum(dim=1).unsqueeze(-1).expand(-1, -1, shape[-1])

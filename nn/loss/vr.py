import numpy as np
import torch
from functional.homology import diagrams


class VietorisRips(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor):
        # Return birth and death times separately
        ret = [diagrams(X[b].numpy(force=True), gens=True) for b in range(X.shape[0])]
        dgms, gens = [r[0] for r in ret], [(r[1][0], r[1][1][0]) for r in ret]
        m_dgm = max(max(len(d[0]), len(d[1])) for b in dgms for d in b)

        dgms_tensor = torch.tensor(np.stack(
            [
                np.stack(
                    [
                        np.pad(dgms[b][0], ((0, m_dgm - dgms[b][0].shape[0]), (0, 0)), constant_values=np.inf),
                        np.pad(dgms[b][1], ((0, m_dgm - dgms[b][1].shape[0]), (0, 0)), constant_values=np.inf)
                    ]
                ) for b in range(X.shape[0])
            ]
        ))

        gens_tensor = torch.tensor(np.stack(
            [
                np.stack(
                    [
                        np.pad(gens[b][0], ((0, m_dgm - dgms[b][0].shape[0]), (0, 1)), constant_values=-1),
                        np.pad(gens[b][1], ((0, m_dgm - dgms[b][1].shape[0]), (0, 0)), constant_values=-1)
                    ]
                ) for b in range(X.shape[0])
            ]
        ))

        gens_birth, gens_death = gens_tensor[:, :, :-2], gens_tensor[:, :, -2:]
        ctx.save_for_backward(X.shape, gens_birth, gens_death)

        return dgms_tensor[:, :, 0], dgms_tensor[:, :, 1]

    @staticmethod
    def backward(ctx, grad_output_birth, grad_output_death):
        shape, gens_birth, gens_death = ctx.saved_tensors
        gens_birth, gens_death = gens_birth.numpy(force=True), gens_death.numpy(force=True)
        x = np.arange(shape[1])
        Jd = (x.reshape(-1, 1) == gens_death[:, :, 0]) | (x.reshape(-1, 1) == gens_death[:, :, 1])
        Jb = (x.reshape(-1, 1) == gens_birth[:, :, 0]) | (x.reshape(-1, 1) == gens_birth[:, :, 1])

        return torch.tensor(Jb) * grad_output_birth, torch.tensor(Jd) * grad_output_death

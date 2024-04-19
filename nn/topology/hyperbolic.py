import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from nn.base import IntrinsicBase
from nn.module import IntrinsicModule
from utils.math import compute_unique_distances


class DeltaHyperbolicity(IntrinsicModule):
    DISTANCES = False

    def __init__(self, tag: str = None, parents: list[IntrinsicBase] = (), writer: SummaryWriter = None):
        super().__init__(tag=tag or f'Delta Hyperbolicity {id(self)}', parents=parents, writer=writer)

    def _forward(self, x: torch.Tensor, *, label: str = IntrinsicModule.LABEL, logging: bool = IntrinsicModule.LOGGING, channel_first: bool = IntrinsicModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        delta = np.zeros(x.shape[0])
        for b in range(x.shape[0]):
            d = (compute_unique_distances(x[b]) if not distances else x[b]).detach().numpy()
            p = 0
            row = d[p, :][np.newaxis, :]
            col = d[:, p][:, np.newaxis]
            XY_p = 0.5 * (row + col - d)
            maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)

            delta[b] = np.max(maxmin - XY_p)
        delta = torch.tensor(delta)

        return delta

    def log(self, args: tuple, kwargs: dict, delta, tag: str, writer: SummaryWriter):
        writer.add_scalar('/'.join((kwargs['label'], tag)), delta.mean(), self.step)

        return delta

    def get_tag(self):
        return self.tag

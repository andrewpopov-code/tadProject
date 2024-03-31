import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .base import TopologyBase
from .module import TopologyModule
from utils import compute_unique_distances


class DeltaHyperbolicity(TopologyModule):
    def __init__(self, tag: str = None, parent: TopologyBase = None, writer: SummaryWriter = None):
        super().__init__(tag=tag or f'Delta Hyperbolicity {id(self)}', parent=parent, writer=writer)

    def forward(self, x: torch.Tensor, *, label: str = '', logging: bool = True, batches: bool = False, channel_first: bool = True, distances: bool = True):
        if not batches:
            x = x.unsqueeze(0)

        if channel_first:
            if x.ndim == 2 + batches:
                x = x.transpose(0 + batches, 1 + batches)
            else:
                x = x.transpose(0 + batches, 1 + batches).transpose(1 + batches, 2 + batches)

        y = np.zeros(x.shape[0])
        for b in range(x.shape[0]):
            d = (compute_unique_distances(x[b]) if not distances else x[b]).detach().numpy()
            p = 0
            row = d[p, :][np.newaxis, :]
            col = d[:, p][:, np.newaxis]
            XY_p = 0.5 * (row + col - d)
            maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)

            y[b] = np.max(maxmin - XY_p)
        y = torch.tensor(y)

        return y if batches else y[0]

    def log(self, args: tuple, kwargs: dict, delta, tag: str, writer: SummaryWriter):
        if kwargs.get('batches', False):
            delta = delta[0]
        writer.add_scalar('/'.join((kwargs['label'], tag)), delta, self.step)

        return delta

    def get_tags(self):
        if self.parent() is not None:
            return self.parent().get_tags() + [self.tag]
        return [self.tag]

import torch
import numpy as np

from topology import TopologyBase
from utils import compute_unique_distances


class DeltaHyperbolicity(TopologyBase):
    def __init__(self, parent: ['TopologyBase', None] = None):
        super().__init__(tag=f'Delta Hyperbolicity {id(self)}', parent=parent)

    def forward(self, x: torch.Tensor, *, label: str = '', distances: bool = True, logging: bool = True, batches: bool = False):
        if not batches:
            x = x.unsqueeze(0)

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

    @staticmethod
    def log(self: 'DeltaHyperbolicity', args: tuple, kwargs: dict, delta):
        if kwargs.get('batches', False):
            delta = delta[0]

        if kwargs.get('logging', True):
            kwargs['writer'].add_scalar(kwargs['tag'], delta, self.step)

        return delta

    def get_tags(self):
        if self.parent() is not None:
            return self.parent().get_tags() + [self.tag]
        return [self.tag]

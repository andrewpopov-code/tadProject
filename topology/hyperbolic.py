import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from topology import TopologyBase
from utils import image_distance, euclidean_dist


class DeltaHyperbolicity(TopologyBase):
    def __init__(self, parent: ['TopologyBase', None] = None):
        super().__init__(tag=f'Delta Hyperbolicity {id(self)}', parent=parent)

    def forward(self, x: torch.Tensor, *args, label: str = '', distances: bool = True, **kwargs):
        x = x.numpy()
        if not distances:
            x = np.unique(x, axis=-2)
            if x.ndim == 3:  # image
                x = image_distance(x)
            else:  # T x D
                x = euclidean_dist(x, x)

        p = 0
        row = x[p, :][np.newaxis, :]
        col = x[:, p][:, np.newaxis]
        XY_p = 0.5 * (row + col - x)
        maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)

        return label, np.max(maxmin - XY_p)

    def log(self, tag: str, delta: float = None, *args, writer: SummaryWriter):
        if self.logging and delta is not None:
            writer.add_scalar(tag, delta)

        return tag, delta

    def get_tags(self):
        if self.parent is not None:
            return self.parent.get_tags() + [self.tag]
        return [self.tag]

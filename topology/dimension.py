import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dadapy import data

from utils import image_distance, euclidean_dist
from .topologybase import TopologyBase


class IntrinsicDimension(TopologyBase):
    def __init__(self, parent: [TopologyBase, None] = None):
        super().__init__(parent)
        self.tag = f'ID Profile {id(self)}'

    def forward(self, x: torch.Tensor, *args, tag: str = '', distances: bool = True, **kwargs):
        x = x.numpy()
        if not distances:
            x = np.unique(x, axis=-2)
            if x.ndim == 3:  # image
                x = image_distance(x)
            else:  # T x D
                x = euclidean_dist(x, x)
        dim, err, _ = data.Data(distances=x).compute_id_2NN()
        return tag, dim, err

    def log(self, tag: str, dim: int = None, err: float = None, *args, writer: SummaryWriter):
        if not self.logging:
            return tag, dim, err

        writer.add_scalar(tag + '/ID Estimate', dim)
        writer.add_scalar(tag + '/ID Estimate Error', err)
        return tag, dim, err

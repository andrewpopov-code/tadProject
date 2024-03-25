import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dadapy import data

from utils import image_distance, euclidean_dist
from .topologybase import TopologyBase


class IntrinsicDimension(TopologyBase):
    def __init__(self, parent: [TopologyBase, None] = None):
        super().__init__(tag=f'ID Profile {id(self)}', parent=parent)

    def forward(self, x: torch.Tensor, *args, label: str = '', distances: bool = True, **kwargs):
        x = x.numpy()
        if not distances:
            x = np.unique(x, axis=-2)
            if x.ndim == 3:  # image
                x = image_distance(x)
            else:  # T x D
                x = euclidean_dist(x, x)
        dim, err, _ = data.Data(distances=x).compute_id_2NN()
        return label, dim, err

    def get_tags(self):
        if self.parent is not None:
            return self.parent.get_tags() + [self.tag]
        return [self.tag]

    def log(self, tag: str, dim: int = None, err: float = None, *args, writer: SummaryWriter):
        if self.logging and dim is not None and err is not None:
            writer.add_scalar(tag + '/ID Estimate', dim)
            writer.add_scalar(tag + '/ID Estimate Error', err)

        return tag, dim, err

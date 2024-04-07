import torch
from torch.utils.tensorboard import SummaryWriter
from dadapy import data

from ..base import TopologyBase
from ..module import TopologyModule
from utils import compute_unique_distances


class IntrinsicDimension(TopologyModule):
    # TODO: integrate estimators from functional
    DISTANCES = False

    def __init__(self, tag: str = None, parents: [TopologyBase] = (), writer: SummaryWriter = None):
        super().__init__(tag=tag or f'ID Profile {id(self)}', parents=parents, writer=writer)

    def forward(self, x: torch.Tensor, *, label: str = TopologyModule.LABEL, logging: bool = TopologyModule.LOGGING, channel_first: bool = TopologyModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)
        dim, err = torch.zeros(x.shape[0]), torch.zeros(x.shape[0])
        for b in range(x.shape[0]):
            d = compute_unique_distances(x[b]) if not distances else x[b]
            dim[b], err[b], _ = data.Data(distances=d.detach().numpy()).compute_id_2NN()
        return dim, err

    def get_tag(self):
        return self.tag

    def log(self, args: tuple, kwargs: dict, result, tag: str, writer: SummaryWriter):
        dim, err = result
        dim, err = dim[0], err[0]

        writer.add_scalar('/'.join((kwargs['label'] + ' (ID Estimate)', tag)), dim, self.step)
        writer.add_scalar('/'.join((kwargs['label'] + ' (ID Estimate Error)', tag)), err, self.step)

        return result

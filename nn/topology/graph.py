import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

import torch
from torch.utils.tensorboard import SummaryWriter

from nn.base import TopologyBase


class Curvature(TopologyBase):
    def __init__(self, writer: SummaryWriter = None):
        super().__init__(self.log if writer is not None else None)

    def forward(self, d: torch.Tensor):
        g = nx.Graph()
        e = []
        for i in range(d.shape[0]):
            for j in range(d.shape[0]):
                e.append((i, j, d[i][j]))
        g.add_weighted_edges_from(e)

        # Start a Ricci flow with Lin-Yau's probability distribution setting with 4 process.
        orf = OllivierRicci(g, alpha=0.5, base=1, exp_power=0, proc=4, verbose="INFO")

        # Do Ricci flow for 2 iterations
        orf.compute_ricci_flow(iterations=25)

    def log(self):
        ...

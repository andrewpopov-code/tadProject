import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

import torch
from torch.utils.tensorboard import SummaryWriter

from src.intrinsic.nn import IntrinsicModule
from ..base import IntrinsicBase
from src.intrinsic.utils.math import compute_unique_distances
from src.intrinsic.utils.tensorboard import show_results_ricci


class Curvature(IntrinsicModule):
    DISTANCES = False

    def __init__(self, tag: str = None, parents: list[IntrinsicBase] = (), writer: SummaryWriter = None, ricci_iter: int = 3):
        super().__init__(tag=tag or f'', parents=parents, writer=writer)
        self.ricci_iter = ricci_iter

    def _forward(self, x: torch.Tensor, *, label: str = IntrinsicModule.LABEL, logging: bool = IntrinsicModule.LOGGING, channel_first: bool = IntrinsicModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        graphs = []

        for b in range(x.shape[0]):
            d = compute_unique_distances(x[b]) if not distances else x[b]

            G = nx.Graph()

            e = []
            s = set()
            for i in range(d.shape[0]):
                for j in range(i + 1, d.shape[0]):
                    e.append((i, j))
                    s.add(i)
                    s.add(j)
            s = list(s)
            ee = []
            for i, j in e:
                ee.append((s.index(i), s.index(j), d[i][j]))
            G.add_weighted_edges_from(ee)

            # Start a Ricci flow with Lin-Yau's probability distribution setting with 4 process.
            orc = OllivierRicci(G, alpha=0.5, verbose="TRACE")

            # Do Ricci flow for 2 iterations
            orc.compute_ricci_flow(iterations=self.ricci_iter)

            orc.compute_ricci_curvature()
            G_orc = orc.G.copy()

            try:
                cc = orc.ricci_community()
                nx.set_node_attributes(G_orc, cc[1], "community")
            except AssertionError:
                nx.set_node_attributes(G_orc, 0, "community")

            graphs.append(G_orc)
        return graphs

    def log(self, args: tuple, kwargs: dict, result: list[nx.Graph], tag: str, writer: SummaryWriter):
        G = result[0]
        fig = show_results_ricci(G, 'community')
        writer.add_figure(kwargs['label'] + '/' + tag, fig)

        return result

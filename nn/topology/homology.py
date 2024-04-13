import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gtda.diagrams import BettiCurve, PersistenceEntropy, PairwiseDistance
from gtda.homology import VietorisRipsPersistence

from dataclasses import dataclass

from utils.math import compute_unique_distances
from utils.tensorboard import draw_heatmap, plot_persistence
from ..base import TopologyBase
from ..module import TopologyModule
from ..functional.homology import diagrams, betti, persistence_metric, persistence_entropy, pairwise_dist


@dataclass
class PersistenceInformation:
    diagram: np.array
    betti: np.array
    entropy: np.array
    persistence: np.array


class Persistence(TopologyModule):
    DISTANCES = False

    def __init__(self, tag: str = None, parents: list[TopologyBase] = (), writer: SummaryWriter = None, homology_dim: int = 1):
        super().__init__(tag=tag or f'Persistence Profile {id(self)}', parents=parents, writer=writer)
        self.maxdim = homology_dim
        self.PD = PairwiseDistance(metric='wasserstein')

        self.diagrams: list[PersistenceInformation] = []  # to compute the heatmap

    def forward(self, x: torch.Tensor, *, label: str = TopologyModule.LABEL, logging: bool = TopologyModule.LOGGING, channel_first: bool = TopologyModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        pi, bc, pe, persistence = [], [], [], []
        for b in range(x.shape[0]):
            d = (compute_unique_distances(x[b]) if not distances else x[b]).unsqueeze(0)

            pi.append(diagrams(d, maxdim=self.maxdim))
            bc.append(betti(pi[-1]))
            pe.append(persistence_entropy(pi[-1]))
            persistence.append(persistence_metric(pi[-1]))

        self.diagrams.append(PersistenceInformation(diagram=pi, betti=bc, entropy=pe, persistence=persistence))
        return self.diagrams[-1]

    def heatmap(self):
        # d = torch.zeros((len(self.diagrams), len(self.diagrams)))
        # for i in range(len(self.diagrams)):
        #     for j in range(i + 1, len(self.diagrams)):
        #         d[i, j] = d[j, i] = self.PD.fit(
        #             np.expand_dims(self.diagrams[i].diagram[0], 0)
        #         ).transform(
        #             np.expand_dims(self.diagrams[j].diagram[0], 0)
        #         )[0, 0]
        d = pairwise_dist([dgrm.betti[0] for dgrm in self.diagrams])
        return d

    def log(self, args: tuple, kwargs: dict, result: PersistenceInformation, tag: str, writer: SummaryWriter):
        pi, bc, entropy, persistence = result.diagram[0], result.betti[0], result.entropy[0], result.persistence[0]

        for j in range(bc.shape[1]):
            writer.add_scalars(
                '/'.join((kwargs['label'] + ' (Betti Curves)', tag)), {
                    f'Dimension {i}': bc[i, j] for i in range(bc.shape[0])
                }, j
            )

        writer.add_figure(
            '/'.join((kwargs['label'] + ' (Persistence Diagrams)', tag)),
            plot_persistence(pi)
        )

        writer.add_scalars('/'.join((kwargs['label'] + ' (Persistence Metric)', tag)), {
            f'H{dim}': persistence[dim] for dim in range(self.maxdim + 1)
        }, self.step)

        writer.add_scalars('/'.join((kwargs['label'] + ' (Persistence Entropy)', tag)), {
            f'H{dim}': entropy[dim] for dim in range(self.maxdim + 1)
        }, self.step)

        return result

    def flush(self):
        if self.diagrams:
            tags = self.get_tags()
            for ws, ts in tags:
                for w in ws:
                    w.add_figure(
                        '/'.join(['RTD'] + ts),
                        draw_heatmap(self.heatmap())
                    )
        self.diagrams = []

        return TopologyModule.flush(self)

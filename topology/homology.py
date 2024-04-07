import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from gtda.diagrams import PairwiseDistance, BettiCurve, PersistenceEntropy
from gtda.homology import VietorisRipsPersistence

from dataclasses import dataclass

from utils import draw_heatmap, compute_unique_distances
from .base import TopologyBase
from .module import TopologyModule


@dataclass
class PersistenceInformation:
    diagram: np.array
    betti: np.array
    entropy: np.array
    persistence: np.array


class Persistence(TopologyModule):
    def __init__(self, tag: str = None, parent: TopologyBase = None, writer: SummaryWriter = None, homology_dim: int = 1):
        super().__init__(tag=tag or f'Persistence Profile {id(self)}', parent=parent, writer=writer)
        self.VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=tuple(range(homology_dim + 1)))
        self.PE = PersistenceEntropy(normalize=True)
        # self.PD = PairwiseDistance(metric='wasserstein')
        self.Betti = BettiCurve()

        self.homology_dim = homology_dim
        # self.diagrams = []  # to compute the heatmap

    def forward(self, x: torch.Tensor, *, label: str = '', logging: bool = True, channel_first: bool = True, distances: bool = False):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        pi, bc, pe, persistence = [], [], [], []
        for b in range(x.shape[0]):
            d = (compute_unique_distances(x[b]) if not distances else x[b]).unsqueeze(0)
            pi.append(self.VR.fit_transform(d)[0])
            bc.append(self.Betti.fit_transform(pi[-1].reshape(-1, *pi[-1].shape))[0])
            pe.append(self.PE.fit_transform(pi[-1].reshape(-1, *pi[-1].shape))[0])

            z = torch.zeros(self.homology_dim + 1)
            for start, end, dim in pi[-1]:
                z[int(dim)] += end - start
            persistence.append(z)

        return PersistenceInformation(diagram=pi, betti=bc, entropy=pe, persistence=persistence)

    # def heatmap(self):
    #     d = torch.zeros((len(self.diagrams), len(self.diagrams)))
    #     for i in range(len(self.diagrams)):
    #         for j in range(i + 1, len(self.diagrams)):
    #             d[i, j] = d[j, i] = self.PD.fit(self.diagrams[i]).transform(self.diagrams[j])[0]
    #     return d

    def log(self, args: tuple, kwargs: dict, result: PersistenceInformation, tag: str, writer: SummaryWriter):
        pi, bc, entropy, persistence = result.diagram, result.betti, result.entropy, result.persistence
        pi, bc, entropy, persistence = pi[0], bc[0], entropy[0], persistence[0]

        for j in range(bc.shape[1]):
            writer.add_scalars(
                '/'.join((kwargs['label'] + ' (Betti Curves)', tag)), {
                    f'Dimension {i}': bc[i, j] for i in range(bc.shape[0])
                }, j
            )

        # TODO: add persistence diagrams, persistence entropy, persistence metric
        return result

    # def finalize(self, encoder_step: int):
    #     if self.display_heatmap is not None and encoder_step % self.display_heatmap == 0:
    #         draw_heatmap(self.heatmap(), self.writer, 'RTD', 'Pairwise RTD')
    #     self.diagrams = []

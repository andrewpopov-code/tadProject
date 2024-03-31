import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from gtda.diagrams import PairwiseDistance, BettiCurve
from gtda.homology import VietorisRipsPersistence

from utils import draw_heatmap, compute_unique_distances
from .base import TopologyBase
from .module import TopologyModule


class Persistence(TopologyModule):
    def __init__(self, tag: str = None, parent: TopologyBase = None, writer: SummaryWriter = None, homology_dim: int = 1):
        super().__init__(tag=tag or f'Persistence Profile {id(self)}', parent=parent, writer=writer)
        self.VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=tuple(range(homology_dim + 1)))
        self.PD = PairwiseDistance(metric='wasserstein')
        self.Betti = BettiCurve()
        # self.diagrams = []  # to compute the heatmap

    def forward(self, x: torch.Tensor, *, label: str = '', logging: bool = True, batches: bool = False, distances: bool = True):
        if batches:
            pi, bc = [], []
            for b in range(x.shape[0]):
                d = (compute_unique_distances(x[b]) if not distances else x[b]).unsqueeze(0)
                pi.append(self.VR.fit_transform(d)[0])
                bc.append(self.Betti.fit_transform(pi[-1].reshape(-1, *pi[-1].shape))[0])
        else:
            d = (compute_unique_distances(x) if not distances else x).unsqueeze(0)
            pi = self.VR.fit_transform(d)
            bc = self.Betti.fit_transform(pi)

        return pi, bc

    # def heatmap(self):
    #     d = torch.zeros((len(self.diagrams), len(self.diagrams)))
    #     for i in range(len(self.diagrams)):
    #         for j in range(i + 1, len(self.diagrams)):
    #             d[i, j] = d[j, i] = self.PD.fit(self.diagrams[i]).transform(self.diagrams[j])[0]
    #     return d

    @staticmethod
    def log(self: 'Persistence', args: tuple, kwargs: dict, result):
        pi, bc = result

        if kwargs.get('batches', False):
            pi, bc = pi[0], bc[0]

        if kwargs.get('logging', True):
            for j in range(bc.shape[1]):
                kwargs['writer'].add_scalars(
                    '/'.join((kwargs['label'] + ' (Betti Curves)', kwargs['tag'])), {
                        f'Dimension {i}': bc[i, j] for i in range(bc.shape[0])
                    }, j
                )

        # TODO: add persistence diagrams
        return result

    # def finalize(self, encoder_step: int):
    #     if self.display_heatmap is not None and encoder_step % self.display_heatmap == 0:
    #         draw_heatmap(self.heatmap(), self.writer, 'RTD', 'Pairwise RTD')
    #     self.diagrams = []

import torch
from torch.utils.tensorboard import SummaryWriter
from gtda.diagrams import PairwiseDistance, BettiCurve
from gtda.homology import VietorisRipsPersistence

from utils import draw_heatmap, betti_curves_str
from .topologybase import TopologyBase


class Persistence(TopologyBase):
    def __init__(self, homology_dim: int = 1, parent: [TopologyBase, None] = None):
        super().__init__(tag=f'Persistence {id(self)}', parent=parent)
        self.VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=tuple(range(homology_dim + 1)))
        self.PD = PairwiseDistance(metric='wasserstein')
        self.Betti = BettiCurve()
        self.diagrams = []  # to compute the heatmap

    def forward(self, x: torch.Tensor, *, label: str = '', distances: bool = True, logging: bool = True):
        if not distances:
            ...

        pi = self.VR.fit_transform(x)
        self.diagrams.append(pi)
        bc = self.Betti.fit_transform(pi)

        return pi, bc

    def heatmap(self):
        d = torch.zeros((len(self.diagrams), len(self.diagrams)))
        for i in range(len(self.diagrams)):
            for j in range(i + 1, len(self.diagrams)):
                d[i, j] = d[j, i] = self.PD.fit(self.diagrams[i]).transform(self.diagrams[j])[0]
        return d

    @staticmethod
    def log(self: 'Persistence', args: tuple, kwargs: dict, result):
        if kwargs['logging']:
            for j in range(result[1].shape[2]):
                kwargs['writer'].add_scalars(
                    kwargs['tag'] + '/Betti Curves', {
                        f'Dimension {i}': result[1][0, i, j] for i in range(result[1].shape[1])
                    }
                )

        # TODO: add persistence diagrams
        return result

    def finalize(self, encoder_step: int):
        if self.display_heatmap is not None and encoder_step % self.display_heatmap == 0:
            draw_heatmap(self.heatmap(), self.writer, 'RTD', 'Pairwise RTD')
        self.diagrams = []

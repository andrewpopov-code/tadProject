import torch
from torch.utils.tensorboard import SummaryWriter
from gtda.diagrams import PairwiseDistance, BettiCurve
from gtda.homology import VietorisRipsPersistence

from utils import draw_heatmap, betti_curves_str
from .topologybase import TopologyBase


class Persistence(TopologyBase):
    def __init__(self, homology_dim: int = 1, parent: [TopologyBase, None] = None):
        super().__init__(parent)
        self.VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=tuple(range(homology_dim + 1)))
        self.PD = PairwiseDistance(metric='wasserstein')
        self.Betti = BettiCurve()
        self.diagrams = []  # to compute the heatmap
        self.tag = f'Persistence {id(self)}'

    def forward(self, x: torch.Tensor, *args, tag: str = '', **kwargs):
        pi = self.VR.fit_transform(x)
        self.diagrams.append(pi)
        bc = self.Betti.fit_transform(pi)

        return tag, pi, bc

    def heatmap(self):
        d = torch.zeros((len(self.diagrams), len(self.diagrams)))
        for i in range(len(self.diagrams)):
            for j in range(i + 1, len(self.diagrams)):
                d[i, j] = d[j, i] = self.PD.fit(self.diagrams[i]).transform(self.diagrams[j])[0]
        return d

    def log(self, tag: str, pi=None, bc=None, *args, writer: SummaryWriter):
        if not self.logging:
            return tag, pi, bc, *args

        for j in range(bc.shape[2]):
            writer.add_scalars(
                tag + '/Betti Curves', {
                    f'Dimension {i}': bc[0, i, j] for i in range(bc.shape[1])
                }
            )

        # TODO: add persistence diagrams

    def finalize(self, encoder_step: int):
        if self.display_heatmap is not None and encoder_step % self.display_heatmap == 0:
            draw_heatmap(self.heatmap(), self.writer, 'RTD', 'Pairwise RTD')
        self.diagrams = []

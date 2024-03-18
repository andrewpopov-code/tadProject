import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from gtda.diagrams import PairwiseDistance, BettiCurve
from gtda.homology import VietorisRipsPersistence

from utils import draw_heatmap


class Persistence(nn.Module):
    def __init__(self, writer: SummaryWriter, homology_dim=1):
        super().__init__()
        self.writer = writer
        self.VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=tuple(range(homology_dim + 1)))
        self.PD = PairwiseDistance(metric='wasserstein')
        self.Betti = BettiCurve()
        self.diagrams = []  # to compute the heatmap
        self.step = 0

    def __call__(self, d: torch.Tensor):
        self.step += 1
        pi = self.VR.fit_transform(d)
        self.diagrams.append(pi)
        bc = self.Betti.fit_transform(pi)

        self.log(bc)

    def heatmap(self):
        d = torch.zeros((len(self.diagrams), len(self.diagrams)))
        for i in range(len(self.diagrams)):
            for j in range(i + 1, len(self.diagrams)):
                d[i, j] = d[j, i] = self.PD.fit(self.diagrams[i]).transform(self.diagrams[j])[0]
        return d

    def log(self, bc):
        for i in range(bc.shape[0]):
            self.writer.add_scalar(f'Betti Curves/Layer {self.step} Betti Curve', bc[i], i)

    def finalize(self):
        draw_heatmap(self.heatmap(), self.writer, 'RTD', 'Pairwise RTD')

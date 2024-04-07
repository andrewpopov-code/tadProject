import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gtda.diagrams import BettiCurve, PersistenceEntropy, PairwiseDistance
from gtda.homology import VietorisRipsPersistence

from dataclasses import dataclass

from utils import compute_unique_distances, draw_heatmap
from ..base import TopologyBase
from ..module import TopologyModule


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
        self.VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=tuple(range(homology_dim + 1)))
        self.PE = PersistenceEntropy(normalize=True)
        self.PD = PairwiseDistance(metric='wasserstein')
        self.Betti = BettiCurve()

        self.homology_dim = homology_dim
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
            pi.append(self.VR.fit_transform(d)[0])
            bc.append(self.Betti.fit_transform(pi[-1].reshape(-1, *pi[-1].shape))[0])
            pe.append(self.PE.fit_transform(pi[-1].reshape(-1, *pi[-1].shape))[0])

            z = torch.zeros(self.homology_dim + 1)
            for start, end, dim in pi[-1]:
                z[int(dim)] += end - start
            persistence.append(z)

        self.diagrams.append(PersistenceInformation(diagram=pi, betti=bc, entropy=pe, persistence=persistence))
        return self.diagrams[-1]

    def heatmap(self):
        d = torch.zeros((len(self.diagrams), len(self.diagrams)))
        for i in range(len(self.diagrams)):
            for j in range(i + 1, len(self.diagrams)):
                d[i, j] = d[j, i] = self.PD.fit(
                    np.expand_dims(self.diagrams[i].diagram[0], 0)
                ).transform(
                    np.expand_dims(self.diagrams[j].diagram[0], 0)
                )[0, 0]
        return d

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

    def flush(self):
        if self.diagrams:
            tags = self.get_tags()  # TODO: not with every call
            for ws, ts in tags:
                for w in ws:
                    w.add_figure(
                        '/'.join(['RTD'] + ts),
                        draw_heatmap(self.heatmap())
                    )
        self.diagrams = []

        return TopologyModule.flush(self)

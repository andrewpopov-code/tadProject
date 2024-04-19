import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass

from utils.math import compute_unique_distances
from utils.tensorboard import draw_heatmap, plot_persistence
from ..base import IntrinsicBase
from ..module import IntrinsicModule
from ..functional.homology import diagrams, betti, persistence_norm, persistence_entropy, pairwise_dist, divergence


@dataclass
class PersistenceInformation:
    diagram: np.array
    betti: np.array
    entropy: np.array
    persistence: np.array
    sample: np.array


class Persistence(IntrinsicModule):
    DISTANCES = False

    def __init__(self, tag: str = None, parents: list[IntrinsicBase] = (), writer: SummaryWriter = None, homology_dim: int = 1):
        super().__init__(tag=tag or f'Persistence Profile {id(self)}', parents=parents, writer=writer)
        self.maxdim = homology_dim
        self.diagrams: list[PersistenceInformation] = []  # to compute the heatmap

    def _forward(self, x: torch.Tensor, *, label: str = IntrinsicModule.LABEL, logging: bool = IntrinsicModule.LOGGING, channel_first: bool = IntrinsicModule.CF, distances: bool = DISTANCES):
        if channel_first:
            if x.ndim == 3:
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).transpose(2, 3)

        pi, bc, pe, persistence = [], [], [], []
        for b in range(x.shape[0]):
            d = compute_unique_distances(x[b]).detach().numpy() if not distances else x[b].detach().numpy()

            pi.append(diagrams(d, maxdim=self.maxdim))
            bc.append(betti(pi[-1]))
            pe.append(persistence_entropy(pi[-1]))
            persistence.append(persistence_norm(pi[-1]))

        self.diagrams.append(PersistenceInformation(diagram=pi, betti=bc, entropy=pe, persistence=persistence, sample=x.detach().numpy()))
        return self.diagrams[-1]

    def heatmap(self):
        return pairwise_dist(np.array([dgrm.betti[0] for dgrm in self.diagrams]))

    def mtd(self):
        d = np.zeros((len(self.diagrams), len(self.diagrams), self.maxdim + 1))
        for i in range(len(self.diagrams)):
            for j in range(len(self.diagrams)):
                l = self.diagrams[i].sample.shape[0]
                d[i][j] = np.mean([
                    divergence(self.diagrams[i].sample[k], self.diagrams[j].sample[k]) for k in range(l)
                ], axis=0)
        return d

    def log(self, args: tuple, kwargs: dict, result: PersistenceInformation, tag: str, writer: SummaryWriter):
        pi, bc, entropy, persistence = result.diagram[0], result.betti[0], result.entropy[0], result.persistence[0]

        for j in range(bc.shape[1]):
            writer.add_scalars(
                '/'.join((kwargs['label'] + ' (Betti Curves)', tag + f' (Call {self.step})')), {
                    f'Dimension {i}': bc[i, j] for i in range(bc.shape[0])
                }, j
            )

        writer.add_figure(
            '/'.join((kwargs['label'] + ' (Persistence Diagrams)', tag + f' (Call {self.step})')),
            plot_persistence(pi)
        )

        writer.add_scalars('/'.join((kwargs['label'] + ' (Persistence Metric)', tag)), {
            f'H{dim}': persistence[dim] for dim in range(self.maxdim + 1)
        }, self.step)

        writer.add_scalars('/'.join((kwargs['label'] + ' (Persistence Entropy)', tag)), {
            f'H{dim}': entropy[dim] for dim in range(self.maxdim + 1)
        }, self.step)

        return result

    def get_tag(self):
        return self.tag

    def flush(self):
        if self.diagrams:
            maps = self.heatmap()
            div = self.mtd().T

            tags = self.get_tags()
            for ws, ts, logging in tags:
                if logging:
                    for w in ws:
                        for dim in range(len(maps)):
                            w.add_figure(
                                '/'.join(['Betti Distance'] + ts),
                                draw_heatmap(torch.tensor(maps[dim])),
                                dim
                            )

                            w.add_figure(
                                '/'.join(['Manifold Topology Divergence'] + ts),
                                draw_heatmap(torch.tensor(div[dim])),
                                dim
                            )
        self.diagrams = []

        return IntrinsicModule.flush(self)

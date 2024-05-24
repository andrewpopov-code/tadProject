import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass

from src.intrinsic.utils.math import unique_points
from src.intrinsic.utils.tensorboard import draw_heatmap, plot_persistence, plot_persistence_each, plot_betti_each
from .base import IntrinsicBase
from .module import IntrinsicModule
from src.intrinsic.functional.homology import vr_diagrams, diagrams_barycenter, betti, persistence_norm, persistence_entropy, pairwise_dist, mtd, rtd


@dataclass
class PersistenceInformation:
    diagrams: list
    betti: np.ndarray
    entropy: np.ndarray
    norm: np.ndarray
    sample: np.ndarray


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

        pi, bc, pe, norm = [], [], [], []
        for b in range(x.shape[0]):
            if distances:
                pi.append(vr_diagrams(x[b].numpy(force=True), maxdim=self.maxdim, distances=True))
            else:
                pi.append(vr_diagrams(unique_points(x[b].numpy(force=True)), maxdim=self.maxdim, distances=False))
            bc.append(betti(pi[-1]))
            pe.append(persistence_entropy(pi[-1]))
            norm.append(persistence_norm(pi[-1]))

        self.diagrams.append(PersistenceInformation(diagrams=pi, betti=np.array(bc), entropy=np.array(pe), norm=np.array(norm), sample=x.numpy(force=True)))
        return self.diagrams[-1]

    @staticmethod
    def heatmap(diag: list[PersistenceInformation]):
        return pairwise_dist(np.array([dgrm.betti[0] for dgrm in diag]))

    def divergence(self, diag: list[PersistenceInformation]):
        d_mtd = np.zeros((len(diag), len(diag), self.maxdim + 1))
        d_rtd = np.zeros((len(diag), len(diag), self.maxdim + 1))

        for i in range(len(diag)):
            for j in range(len(diag)):
                if diag[i].sample.shape[1] != diag[j].sample.shape[1]:
                    d_mtd[i][j] = np.inf
                    continue

                batch_size = diag[i].sample.shape[0]
                d_mtd[i][j] = np.mean([
                    mtd(diag[i].sample[k], diag[j].sample[k]) for k in range(batch_size)
                ], axis=0)
                d_rtd[i][j] = np.mean([
                    rtd(diag[i].sample[k], diag[j].sample[k]) for k in range(batch_size)
                ], axis=0)

        return d_mtd, d_rtd

    def log(self, args: tuple, kwargs: dict, result: PersistenceInformation, tag: str, writer: SummaryWriter):
        pi = diagrams_barycenter(result.diagrams)
        bc = betti(pi)

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

        return result

    def log_epoch(self, epoch: int, label: str, results: list[PersistenceInformation], writer: SummaryWriter):
        tag = self.tag + ': ' + label
        for i, result in enumerate(results):
            entropy, persistence = result.entropy.mean(), result.norm.mean()
            writer.add_scalars(tag + ' (Persistence Metric)', {
                f'H{dim}': persistence[dim] for dim in range(self.maxdim + 1)
            }, epoch + i)

            writer.add_scalars(tag + ' (Persistence Entropy)', {
                f'H{dim}': entropy[dim] for dim in range(self.maxdim + 1)
            }, epoch + i)

        maps = self.heatmap(results)
        div_mtd, div_rtd = self.divergence(results)
        div_mtd, div_rtd = div_mtd.T, div_rtd.T

        writer.add_figure(
            tag + ' (Persistence Diagrams)',
            plot_persistence_each([res.diagrams[0] for res in results]),
            epoch
        )
        writer.add_figure(
            tag + f' (Betti Curves)',
            plot_betti_each([res.betti[0] for res in results]),
            epoch
        )

        for dim in range(self.maxdim + 1):
            writer.add_figure(
                tag + f' (Betti Distance for H{dim})',
                draw_heatmap(maps[dim]),
                epoch
            )

            writer.add_figure(
                tag + f' (Manifold Topology Divergence for H{dim})',
                draw_heatmap(div_mtd[dim]),
                epoch
            )

            writer.add_figure(
                tag + f' (Representation Topology Divergence for H{dim})',
                draw_heatmap(div_rtd[dim]),
                epoch
            )

    def get_tag(self):
        return self.tag

    def plot_diagrams(self, diag, tags):
        maps = self.heatmap(diag)
        div_mtd, div_rtd = self.divergence(diag)
        div_mtd, div_rtd = div_mtd.T, div_rtd.T

        for ws, ts, logging in tags:
            if logging:
                for w in ws:
                    for dim in range(len(maps)):
                        w.add_figure(
                            '/'.join(['Betti Distance'] + ts),
                            draw_heatmap(maps[dim]),
                            dim
                        )

                        w.add_figure(
                            '/'.join(['Manifold Topology Divergence'] + ts),
                            draw_heatmap(div_mtd[dim]),
                            dim
                        )

                        w.add_figure(
                            '/'.join(['Representation Topology Divergence'] + ts),
                            draw_heatmap(div_rtd[dim]),
                            dim
                        )

    def flush(self):
        if self.diagrams:
            self.plot_diagrams(self.diagrams, self.get_tags())
        self.diagrams = []

        return IntrinsicModule.flush(self)

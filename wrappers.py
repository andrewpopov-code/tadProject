import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_topological.nn as tnn
from dadapy import data
from gtda.diagrams import PairwiseDistance, BettiCurve
from gtda.homology import VietorisRipsPersistence
from gtda.images import ImageToPointCloud, RadialFiltration
import matplotlib.pyplot as plt


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    return torch.cdist(x, y)


class AttentionWrapper(nn.Module):
    def __init__(self, attn, homology_dim=1):
        super().__init__()
        self.attn_module = attn
        self.VR = tnn.VietorisRipsComplex(dim=homology_dim)

    def __call__(self, *args, **kwargs):
        x, attn_weights = self.attn_module(*args, **kwargs)
        print(attn_weights)  # Attention matrices
        # TODO: compute distance matrix, curvature, magnitude
        d = AttentionWrapper.distance(attn_weights)
        pi = self.VR(d, treat_as_distances=True)
        id_profile = data.Data(distances=d.numpy()).compute_id_2NN()

        return x, attn_weights

    @staticmethod
    def distance(attn_weights: torch.Tensor):
        d = torch.zeros_like(attn_weights)
        d1 = torch.zeros_like(attn_weights)
        for i in range(attn_weights.shape[0]):
            for j in range(attn_weights.shape[1]):
                d[i][j] = torch.abs(attn_weights[i] - attn_weights[j]).sum() / 2
                d1[i][i] = torch.sqrt(torch.square(torch.sqrt(attn_weights[i]) - torch.sqrt(attn_weights[j])).sum() / 2)
        return d


class FFTLayerWrapper(nn.Module):
    def __init__(self, fft_module):
        super().__init__()
        self.fft_module = fft_module

    def __call__(self, x, *args, **kwargs):
        print(x)  # Batch representation

        return self.fft_module(x, *args, **kwargs)


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
            self.writer.add_scalar(f'Layer {self.step} Betti Curve', bc[i], i)

    def finalize(self):
        d = self.heatmap()

        fig, ax = plt.subplots()
        ax.imshow(d)
        for i in range(len(self.diagrams)):
            for j in range(len(self.diagrams)):
                ax.text(j, i, d[i, j], ha='center', va='center', color='w')
        ax.set_title('Pairwise RTD')
        fig.tight_layout()

        self.writer.add_figure('RTD', fig)


class IntrinsicDimension(nn.Module):
    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self.writer = writer

    def __call__(self, d: torch.Tensor):
        dim = data.Data(distances=d.numpy()).compute_id_2NN()[0]
        self.log(dim)

    def log(self, dim):
        self.writer.add_scalar('ID Profile', dim)


class Mapper(nn.Module):
    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self.writer = writer

    def __call__(self, *args, **kwargs):
        ...

    def log(self):
        ...


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.TransformerEncoder, writer: SummaryWriter, homology_dim=1, display_heads=False):
        # TODO: compute distance matrix, curvature, magnitude, PCA, MLE, PackingNumbers, CorrelationDimension
        # TODO: GeometryScore, MTD, RTD, OllivierRicciCurvature, FormanRicciCurvature
        # TODO: add display_heads support

        super().__init__()
        self.encoder = encoder
        self.Filtration = Persistence(writer, homology_dim)
        self.ID = IntrinsicDimension(writer)
        self.display_heads = display_heads

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        res = x
        x = self.encoder(x, *args, **kwargs)

        """
        src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False
            
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x
        """

        for i in range(self.encoder.num_layers):
            res, attn_weights = self.encoder.layers[i].self_attn(
                res, res, res, need_weigths=True, average_attn_weights=self.display_heads
            )
            # TODO: manipulate the weights, or obtain a distance matrix / matrices
            d = attn_weights
            self.Filtration(d)
            self.ID(d)
        self.Filtration.finalize()

        return x

    # self-attention block
    def _sa_block(
            self, x: torch.Tensor, attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor], is_causal: bool = False
    ) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

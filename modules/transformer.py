import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from typing import Optional
from functools import partial
from json import dumps

from utils import draw_heatmap, id_str
from topology import Persistence, IntrinsicDimension


def _sa_block(
        self, wrapper: 'TransformerEncoderWrapper', x: torch.Tensor, attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor], is_causal: bool = False
) -> torch.Tensor:
    x, attn_weights = self.self_attn(x, x, x,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=True, is_causal=is_causal,
                                     average_attention_weights=not wrapper.display_heads)
    d = attn_weights  # TODO: replace it with metric
    # TODO: 1 x N x N into filtration; N x N into dimension
    wrapper.Filtration(d, outer_step=wrapper.encoder_step, inner_step=wrapper.global_step % wrapper.encoder.num_layers)
    wrapper.ID(d, id_str(wrapper.encoder_step, wrapper.global_step % wrapper.encoder.num_layers))
    wrapper.log(d)
    wrapper.global_step += 1

    return self.dropout1(x)


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.TransformerEncoder, writer: SummaryWriter,
                 homology_dim=1, display_heads=False, display_heatmap: int = None):
        # TODO: compute distance matrix, curvature, magnitude, PCA, MLE, PackingNumbers, CorrelationDimension
        # TODO: GeometryScore, MTD, RTD, OllivierRicciCurvature, FormanRicciCurvature
        # TODO: add display_heads support

        super().__init__()
        self.encoder = encoder
        self.Filtration = Persistence(writer, homology_dim, display_heatmap)
        self.ID = IntrinsicDimension(writer)
        self.display_heads = display_heads
        self.writer = writer
        self.encoder_step = 0  # Incremented with each call of the encoder
        self.global_step = 0  # Incremented with each call of self-attn
        self.num_layers = self.encoder.num_layers
        self.display_heatmap = display_heatmap

        for i in range(self.num_layers):
            self.encoder.layers[i]._sa_block = partial(_sa_block, wrapper=self)

        hp = {
            'Number of Layers': self.num_layers,
            'Dimension of Input/Output Tokens': self.encoder.layers[0].linear1.in_features,
            'Number of Attention Heads': self.encoder.layers[0].self_attn.num_heads
        }
        self.writer.add_text('Encoder', str(self.encoder))
        self.writer.add_text('Encoder Hyperparameters', dumps(hp))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.encoder(x, *args, **kwargs)
        self.Filtration.finalize(self.encoder_step)
        self.encoder_step += 1

        return x

    def log(self, d: torch.Tensor):
        if self.display_heatmap is not None and self.encoder_step % self.display_heatmap == 0:
            draw_heatmap(
                d, self.writer, 'Distance',
                f'Distance at Step {self.encoder_step}, Layer {self.global_step % self.num_layers}'
            )


class TransformerObserver(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()

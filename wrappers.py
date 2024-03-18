import torch
import torch.nn as nn
from torch.nn.modules.transformer import _detect_is_causal_mask, _get_seq_len
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_topological.nn as tnn
from dadapy import data
from gtda.diagrams import PairwiseDistance, BettiCurve
from gtda.homology import VietorisRipsPersistence
from gtda.images import ImageToPointCloud, RadialFiltration
import matplotlib.pyplot as plt
from typing import Optional


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    return torch.cdist(x, y)


def draw_heatmap(d: torch.Tensor, writer: SummaryWriter, tag: str, title: str):
    fig, ax = plt.subplots()
    ax.imshow(d)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, d[i, j], ha='center', va='center', color='w')
    ax.set_title(title)
    fig.tight_layout()

    writer.add_figure(tag, fig)


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
        draw_heatmap(self.heatmap(), self.writer, 'RTD', 'Pairwise RTD')


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
        self.writer = writer

    def __call__(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                 src_key_padding_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        res = src
        x = self.encoder(src, mask, src_key_padding_mask, is_causal)

        self.simulate_encoder(res, mask, src_key_padding_mask, is_causal)
        self.Filtration.finalize()

        return x

    def simulate_encoder(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                         src_key_padding_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.encoder.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
              and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for step, layer in enumerate(self.encoder.layers, 1):
            output = self.simulate_layer(step, layer, output, mask, src_key_padding_mask_for_layers, is_causal)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output

    def simulate_layer(self, step: int, layer: nn.TransformerEncoderLayer, src: torch.Tensor,
                       src_mask: Optional[torch.Tensor] = None,
                       src_key_padding_mask: Optional[torch.Tensor] = None,
                       is_causal: bool = False):

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

        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif layer.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not layer.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (layer.norm1.eps == layer.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.linear1.weight,
                layer.linear1.bias,
                layer.linear2.weight,
                layer.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = layer.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    layer.self_attn.embed_dim,
                    layer.self_attn.num_heads,
                    layer.self_attn.in_proj_weight,
                    layer.self_attn.in_proj_bias,
                    layer.self_attn.out_proj.weight,
                    layer.self_attn.out_proj.bias,
                    layer.activation_relu_or_gelu == 2,
                    layer.norm_first,
                    layer.norm1.eps,
                    layer.norm1.weight,
                    layer.norm1.bias,
                    layer.norm2.weight,
                    layer.norm2.bias,
                    layer.linear1.weight,
                    layer.linear1.bias,
                    layer.linear2.weight,
                    layer.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        x = src
        if layer.norm_first:
            x = x + self._sa_block(layer, layer.norm1(x), src_mask, src_key_padding_mask, step, is_causal=is_causal)
            x = x + layer._ff_block(layer.norm2(x))
        else:
            x = layer.norm1(x + self._sa_block(layer, x, src_mask, src_key_padding_mask, step, is_causal=is_causal))
            x = layer.norm2(x + layer._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
            self, layer: nn.TransformerEncoderLayer, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor], step: int, is_causal: bool = False
    ) -> torch.Tensor:
        x, attn_weights = layer.self_attn(x, x, x,
                                          attn_mask=attn_mask,
                                          key_padding_mask=key_padding_mask,
                                          need_weights=True, is_causal=is_causal,
                                          average_attention_weights=not self.display_heads)
        # TODO: manipulate the weights, or obtain a distance matrix / matrices
        d = attn_weights
        self.Filtration(d)
        self.ID(d)
        self.log(d, step)

        return layer.dropout1(x)

    def log(self, d: torch.Tensor, step: int):
        draw_heatmap(d, self.writer, 'Distance', f'Distance at Step {step}')

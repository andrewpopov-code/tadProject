import torch.nn as nn


class AttentionWrapper(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn_module = attn

    def __call__(self, *args, **kwargs):
        x, A = self.attn_module(*args, **kwargs)
        print(A)  # Attention matrices

        return x, A


class FFTLayerWrapper(nn.Module):
    def __init__(self, fft_module):
        super().__init__()
        self.fft_module = fft_module

    def __call__(self, x, *args, **kwargs):
        print(x)  # Batch representation

        return self.fft_module(x, *args, **kwargs)

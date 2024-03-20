import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dadapy import data


class IntrinsicDimension(nn.Module):
    def __init__(self, writer: SummaryWriter, num_layers: int):
        super().__init__()
        self.writer = writer
        self.num_layers = num_layers

    def __call__(self, d: torch.Tensor, global_step: int):
        dim = data.Data(distances=d.numpy()).compute_id_2NN()[0]
        self.log(dim, global_step)

    def log(self, dim, global_step):
        self.writer.add_scalar(f'ID Profile/Layer {global_step % self.num_layers}', dim)

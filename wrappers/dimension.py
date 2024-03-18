import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dadapy import data


class IntrinsicDimension(nn.Module):
    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self.writer = writer

    def __call__(self, d: torch.Tensor):
        dim = data.Data(distances=d.numpy()).compute_id_2NN()[0]
        self.log(dim)

    def log(self, dim):
        self.writer.add_scalar('ID Profile', dim)

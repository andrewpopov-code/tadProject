import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Mapper(nn.Module):
    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self.writer = writer

    def __call__(self, *args, **kwargs):
        ...

    def log(self):
        ...

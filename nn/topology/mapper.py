import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ..base import TopologyBase


class Mapper(TopologyBase):
    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self.writer = writer

    def forward(self, *args, **kwargs):
        ...

    def log(self):
        ...

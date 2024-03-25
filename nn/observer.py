import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from topology import TopologyBase


class TopologyObserver(TopologyBase):
    # Hooks to a network and fires when it is called | Hooks to modules assuming they're in the same network
    def __init__(self, net: nn.Module = None, writer: SummaryWriter = None, *topology_modules: TopologyBase):
        super().__init__(tag=f'Topological Network {id(net)}', *topology_modules, logging=writer is not None)
        self.writer = writer

        if net is not None:
            net.register_forward_hook(self.increment)
            net.apply(self.register)

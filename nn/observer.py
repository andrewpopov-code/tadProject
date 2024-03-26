import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from topology import TopologyBase


class TopologyObserver(TopologyBase):
    # Hooks to a network and fires when it is called | Hooks to modules assuming they're in the same network
    def __init__(self, *topology_modules: TopologyBase, net: nn.Module = None, writer: SummaryWriter = None):
        super().__init__(
            tag=f'Topology Observer {id(net or self)}',
            *topology_modules
        )
        self.writer = writer

        if net is not None:
            net.register_forward_hook(self.increment)
            net.apply(self.register)

        for m in topology_modules:
            if m.parent is None and hasattr(m, 'log'):
                m.add_log_hook()  # Added once (closest to the master object)
            m.parent = self  # Set by the immediate parent: last forward call is performed closest to the module

    @property
    def parent(self):
        return None

    @parent.setter
    def parent(self, value):
        raise PermissionError("Observer objects can't have parents. Place it outside of other Topology modules.")

    def register(self, m: nn.Module):
        if isinstance(m, TopologyBase):
            self.layers[id(m)] = m
            if m.parent is None and hasattr(m, 'log'):
                m.add_log_hook()  # Added once (closest to the master object)
            m.parent = self  # Set by the immediate parent: last forward call is performed closest to the module

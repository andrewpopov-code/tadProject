import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from topology import TopologyBase, TopologyModule


class TopologyObserver(TopologyBase):
    # Hooks to a network and fires when it is called | Hooks to modules assuming they're in the same network
    def __init__(self, net: nn.Module, writer: SummaryWriter = None, topology_modules: list[TopologyModule] = ()):
        super().__init__(
            tag=f'Topology Observer {id(net or self)}',
            writer=writer,
            layers=topology_modules
        )
        net.register_forward_hook(self.increment)
        net.apply(self.register)

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule):
            self.topology_modules[id(m)] = m
            if m.parent() is None:
                m.add_log_hook()  # Added once (closest to the master object)
            m.set_parent(self)  # Set by the immediate parent: last forward call is performed closest to the module

    def increment(self, m: nn.Module, args: tuple, result):
        self.step += 1
        for k in self.topology_modules:
            self.topology_modules[k].flush()
        return result

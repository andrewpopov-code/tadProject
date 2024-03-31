import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from typing import Callable, Union

from topology import TopologyBase, TopologyModule


class TopologyObserver(TopologyBase):
    # Hooks to a network and fires when it is called | Hooks to modules assuming they're in the same network
    def __init__(
            self, net: nn.Module, *, topology_modules: list[TopologyModule] = (),
            writer: SummaryWriter = None, reset: bool = True,
            pre_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = (),
            post_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = ()
    ):
        super().__init__(
            tag=f'Topology Observer {id(net or self)}',
            writer=writer,
            layers=topology_modules
        )
        self.reset = reset
        self.net = net

        for m, tms in post_topology:
            for tm, kwargs, *f in tms:
                if f:
                    m.register_forward_hook(
                        lambda s, args, result: TopologyObserver._second_arg(tm(f[0](result), **kwargs), result)
                    )
                else:
                    m.register_forward_hook(
                        lambda s, args, result: TopologyObserver._second_arg(tm(result, **kwargs), result)
                    )
                self.register(tm)
        for m, tms in pre_topology:
            for tm, kwargs, *f in tms:
                if f:
                    m.register_forward_pre_hook(
                        lambda s, args, kw: TopologyObserver._second_arg(tm(f[0](*args, **kw), **kwargs), args),
                        with_kwargs=True
                    )
                else:
                    m.register_forward_pre_hook(
                        lambda s, args: TopologyObserver._second_arg(tm(args[0], **kwargs), args)
                    )
                self.register(tm)

        net.register_forward_hook(self.increment)
        for m in topology_modules:
            self.register(m)
        net.apply(self.register)

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule):
            self.topology_modules[id(m)] = m
            m.add_or_skip_log_hook()
            if m.parent() is None or self.reset:
                m.set_parent(self)  # Set to be the observer

    def increment(self, m: nn.Module, args: tuple, result):
        self.step += 1
        for k in self.topology_modules:
            self.topology_modules[k].flush()
        return result

    @staticmethod
    def _second_arg(x, y):
        return y


class TopologyTrainingObserver(TopologyObserver):
    def __init__(
            self, net: nn.Module, *, topology_modules: list[TopologyModule] = (),
            writer: SummaryWriter = None, reset: bool = True,
            pre_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = (),
            post_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[TopologyModule, dict], tuple[TopologyModule, dict, Callable]]]]
            ]] = (),
            log_every_train: int = 1, log_every_val: int = 1
    ):
        super().__init__(net, topology_modules=topology_modules, writer=writer, reset=reset, pre_topology=pre_topology, post_topology=post_topology)
        self.log_every_train = log_every_train
        self.log_every_val = log_every_val
        self.val_step = 0

    def register(self, m: nn.Module):
        if isinstance(m, TopologyModule) and id(m) not in self.topology_modules:
            m.register_forward_pre_hook(self.set_logging, with_kwargs=True)
        super().register(m)

    def increment(self, m: nn.Module, args: tuple, result):
        if m.training:
            self.step += 1
            self.val_step = 0
        else:
            self.val_step += 1
        for k in self.topology_modules:
            self.topology_modules[k].flush()
        return result

    def set_logging(self, m: TopologyModule, args: tuple, kwargs: dict):
        kwargs['logging'] = (self.step % self.log_every_train if m.training else self.val_step % self.log_every_val) == 0
        return args, kwargs

    def flush(self):
        self.val_step = 0
        super().flush()

    def get_tags(self):
        if self.net.training:
            tag = f' (Training Call {self.step})'
        else:
            tag = f' (Validation Call {self.step} + {self.val_step}'
        return [self.tag + tag]

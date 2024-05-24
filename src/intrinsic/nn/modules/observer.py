import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from typing import Callable, Union
import os

from .homology import Persistence
from .dimension import Dimension
from .hyperbolic import DeltaHyperbolicity
from .base import IntrinsicBase
from .module import IntrinsicModule


class _Hook:
    def __init__(self, f: tuple[Callable], tm: IntrinsicModule, kwargs: dict):
        self.f = f
        self.tm = tm
        self.kwargs = kwargs

    def __call__(self, s, args, result):
        if self.f:
            self.tm(self.f[0](result), **self.kwargs)
        else:
            self.tm(result, **self.kwargs)
        return result


class _PreHook:
    def __init__(self, f: tuple[Callable], tm: IntrinsicModule, kwargs: dict):
        self.f = f
        self.tm = tm
        self.kwargs = kwargs

    def __call__(self, s, args, kwargs):
        if self.f:
            self.tm(self.f[0](*args, **kwargs), **self.kwargs)
        else:
            self.tm(args[0], **self.kwargs)


class IntrinsicObserver(IntrinsicBase):
    # Hooks to a network and fires when it is called | Hooks to modules assuming they're in the same network
    def __init__(
            self, net: nn.Module, *, topology_modules: list[IntrinsicModule] = (),
            writer: SummaryWriter = None, find: bool = False,
            pre_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[IntrinsicModule, dict], tuple[IntrinsicModule, dict, Callable]]]]
            ]] = (),
            post_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[IntrinsicModule, dict], tuple[IntrinsicModule, dict, Callable]]]]
            ]] = ()
    ):
        super().__init__(
            tag=f'Intrinsic Observer {id(net or self)}',
            writer=writer,
            topology_children=topology_modules
        )
        self.net = net
        self.forward_information: list[dict[tuple[int, str], list]] = []
        self.registered = set()

        for m, tms in post_topology:
            for tm, kwargs, *f in tms:
                m.register_forward_hook(_Hook(f, tm, kwargs))
                self.register(tm)
        for m, tms in pre_topology:
            for tm, kwargs, *f in tms:
                m.register_forward_pre_hook(_PreHook(f, tm, kwargs), with_kwargs=True)
                self.register(tm)

        net.register_forward_pre_hook(self.info)
        net.register_forward_hook(self.increment)
        for m in topology_modules:
            self.register(m)
        if find:
            net.apply(self.register)

    def register(self, m: nn.Module):
        if isinstance(m, IntrinsicModule) and m is not self.net and id(m) not in self.registered:
            m.register_forward_hook(self.accumulate, with_kwargs=True)

            self.topology_children[id(m)] = m
            if not m.parents():  # No parents previously, or a high level module, so we connect to it
                m.add_parent(self)
            self.registered.add(id(m))

    def increment(self, m: nn.Module, args: tuple, result):
        self.step += 1
        for k in self.topology_children:
            self.topology_children[k].flush()
        return result

    def info(self, m: nn.Module, args: tuple):
        self.forward_information.append({})

    def accumulate(self, m: IntrinsicModule, args: tuple, kwargs: dict, result):
        if id(m) in self.forward_information[-1]:
            self.forward_information[-1][id(m), kwargs['label']].append(result)
        else:
            self.forward_information[-1][id(m), kwargs['label']] = [result]
        return result


class IntrinsicTrainingObserver(IntrinsicObserver):
    def __init__(
            self, net: nn.Module, *, topology_modules: list[IntrinsicModule] = (),
            writer: SummaryWriter = None,
            pre_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[IntrinsicModule, dict], tuple[IntrinsicModule, dict, Callable]]]]
            ]] = (),
            post_topology: list[Union[
                tuple[nn.Module, list[Union[tuple[IntrinsicModule, dict], tuple[IntrinsicModule, dict, Callable]]]]
            ]] = (),
            log_every_train: int = 1, log_every_val: int = 1
    ):
        super().__init__(net, topology_modules=topology_modules, writer=writer, pre_topology=pre_topology, post_topology=post_topology)

        self.log_every_train = log_every_train
        self.log_every_val = log_every_val
        self.val_step = 0
        self.train_epoch_information: list[dict[tuple[int, str], list]] = []  # information yielded at each step
        self.val_epoch_information: list[dict[tuple[int, str], list]] = []  # information yielded at each step
        self.training = False
        self.net.register_forward_hook(self.accumulate_epoch)
        self.net.register_forward_pre_hook(self.set_train)

    def accumulate(self, m: IntrinsicModule, args: tuple, kwargs: dict, result):
        if result is None:
            return result
        return IntrinsicObserver.accumulate(self, m, args, kwargs, result)

    def accumulate_epoch(self, m: nn.Module, args: tuple, result):
        if not self.forward_information[-1]:
            return result

        if self.training:
            self.train_epoch_information.append(self.forward_information[-1])
        else:
            self.val_epoch_information.append(self.forward_information[-1])
        return result

    def info(self, m: nn.Module, args: tuple):
        self.forward_information = [{}]

    def increment(self, m: nn.Module, args: tuple, result):
        if m.training:
            self.step += 1
            self.val_step = 0
        else:
            self.val_step += 1

        for k in self.topology_children:
            self.topology_children[k].flush()

        return result

    def set_train(self, m: nn.Module, args):
        self.training = m.training

    def logging(self):
        return (self.step % self.log_every_train if self.training else self.val_step % self.log_every_val) == 0

    def flush(self):
        self.val_step = 0
        IntrinsicObserver.flush(self)

    def get_tag(self):
        if self.net.training:
            tag = f' (Training Call {self.step})'
        else:
            tag = f' (Validation Call {self.step - 1} + {self.val_step})'

        return self.tag + tag

    def report(self):
        d_train = os.path.join(self.writer.log_dir, self.tag + f': training report #{self.step}')
        train_writer = SummaryWriter(log_dir=d_train)
        d_val = os.path.join(self.writer.log_dir, self.tag + f': validation report #{self.val_step}')
        val_writer = SummaryWriter(log_dir=d_val)

        for epoch, inf in enumerate(self.train_epoch_information):
            for (module_id, label), results in inf.items():
                if isinstance(self.topology_children[module_id], IntrinsicModule):
                    self.topology_children[module_id].log_epoch(epoch * self.log_every_train, label, results, train_writer)
        for epoch, inf in enumerate(self.val_epoch_information):
            for (module_id, label), results in inf.items():
                if isinstance(self.topology_children[module_id], IntrinsicModule):
                    self.topology_children[module_id].log_epoch(epoch * self.log_every_val, label, results, val_writer)


class AttentionTopologyObserver(IntrinsicObserver):
    ...


class AttentionTopologyTrainingObserver(IntrinsicTrainingObserver):
    ...


class EmbeddingIntrinsicTrainingObserver(IntrinsicTrainingObserver):
    def __init__(
            self, net: nn.Module, *, embedding_modules: list[nn.Embedding] = (),
            writer: SummaryWriter = None, log_every_train: int = 1, log_every_val: int = 1
    ):
        self.Filtration = Persistence()
        self.Dimension = Dimension()
        self.DeltaHyperbolicity = DeltaHyperbolicity()

        IntrinsicTrainingObserver.__init__(
            self, net, topology_modules=[self.Filtration, self.Dimension, self.DeltaHyperbolicity],
            writer=writer, log_every_train=log_every_train, log_every_val=log_every_val
        )

        self.embedding_modules = embedding_modules
        net.register_forward_hook(self.embedding_topology)

    def embedding_topology(self, s: nn.Module, args: tuple, result):
        # TODO: this should wrt epoch
        for em in self.embedding_modules:
            w = em.weight.unsqueeze(0)
            self.Filtration(w, label=f'Embedding Module {id(em)}')
            self.Dimension(w, label=f'Embedding Module {id(em)}')
            self.DeltaHyperbolicity(w, label=f'Embedding Module {id(em)}')

        return result

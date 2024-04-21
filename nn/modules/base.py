from torch.utils.tensorboard import SummaryWriter


class _ParentContainer:
    def __init__(self, parent: ['IntrinsicBase', None]):
        self.obj = parent


class IntrinsicBase:
    def __init__(self, tag: str = '', parents: list['IntrinsicBase'] = (), writer: SummaryWriter = None,
                 topology_children: list['IntrinsicBase'] = ()):
        self.step = 0
        self.tag = tag
        self.writer = writer
        self._parents = [_ParentContainer(p) for p in parents]
        self.topology_children: dict[int, 'IntrinsicBase'] = {id(x): x for x in topology_children}

    def parents(self) -> list['IntrinsicBase']:
        return [p.obj for p in self._parents]

    def parents_iter(self):
        for p in self._parents:
            yield p.obj

    def add_parent(self, parent: 'IntrinsicBase'):
        self._parents.append(_ParentContainer(parent))

    def get_tags(self):
        # Idea: get writers and tags from parents and extend them with ours; returns variants x writers x tags x logging
        tags = [
            [ws | {self.writer} if self.writer is not None else ws, ts + [self.get_tag()], logging and self.logging()] for p in self.parents_iter()
            for ws, ts, logging in p.get_tags()
        ]
        return tags or [[{self.writer} if self.writer is not None else set(), [self.get_tag()], self.logging()]]

    def get_tag(self):
        return self.tag + f' (Call {self.step})'

    def logging(self):
        return True

    def flush(self):
        self.step = 0
        for k in self.topology_children:
            self.topology_children[k].flush()
